from __future__ import annotations

import logging
import math
import os
import platform
import subprocess
import time
from dataclasses import dataclass

import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional

from sumolib import checkBinary, net
from sumolib.miscutils import getFreeSocketPort
import traci
import random
import xml.etree.ElementTree as ET
import csv
import statistics
from math import radians, sin, cos, atan2, sqrt

try:
    from stable_baselines3 import PPO
except Exception:
    PPO = None

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from src.ev_management.vehicle_manager import VehicleManager, VehicleRecord
from src.station_management.manager import ChargingStationManager
from src.recommendation_engine.engine import RecommendationEngine
from src.dashboard import Dashboard
from src.monitoring.collectors import (
    RecommendationCollector,
    RLCollector,
    SimulationCollector,
    StationCollector,
    SystemCollector,
    VehicleCollector,
)
from src.monitoring.metrics import increment_counter, observe_histogram, update_metric
from src.visualization.services.integration import VisualizationIntegration
from src.xai.explainer import RecommendationExplainer
from src.simulation.runtime_control import SimulationRuntimeControl
from src.ev_model.battery import energy_needed_kwh

logger = logging.getLogger(__name__)


@dataclass
class Assignment:
    station_id: str
    port_id: Optional[str]
    reserved_at_step: int
    charging_end_step: Optional[int]
    original_destination: str
    status: str = "enroute"
    target_edge_id: Optional[str] = None
    arrived_step: Optional[int] = None
    recommended_distance_km: Optional[float] = None
    recommended_travel_time_min: Optional[float] = None
    recommended_waiting_time_min: Optional[float] = None
    recommended_queue_length: Optional[int] = None
    charging_price_per_kwh: Optional[float] = None
    estimated_session_cost: Optional[float] = None
    charged_energy_kwh: float = 0.0
    session_charging_cost: float = 0.0


class SimulationController:
    def __init__(
        self,
        sumo_cfg: str,
        db_path: str,
        model_path: str,
        fleet_size: int = 1000,
        station_count: int = 500,
        tracked: int = 10,
        charge_threshold_pct: float = 20.0,
        use_gui: bool = False,
        dashboard_state_interval: int = 1,
        dashboard_report_interval: int = 10,
        visualization_interval: int = 5,
        metrics_interval: int = 5,
        simulation_seed: int = 42,
    ) -> None:
        self.sumo_cfg = Path(sumo_cfg)
        self.db_path = db_path
        self.model_path = model_path
        self._resolve_station_db_path()
        self._resolve_model_path()
        self.fleet_size = int(fleet_size)
        self.station_count = int(station_count)
        self.tracked = int(tracked)
        self.charge_threshold_pct = float(charge_threshold_pct)
        self.use_gui = bool(use_gui)
        # Performance throttles for expensive per-step work. Setting any interval
        # to 1 restores the original "run every step" behavior.
        self.dashboard_state_interval = max(1, int(dashboard_state_interval))
        self.dashboard_report_interval = max(1, int(dashboard_report_interval))
        self.visualization_interval = max(1, int(visualization_interval))
        self.metrics_interval = max(1, int(metrics_interval))
        self.simulation_seed = int(simulation_seed)
        random.seed(self.simulation_seed)
        np.random.seed(self.simulation_seed)

        self.vehicle_manager = VehicleManager()
        self.station_manager = ChargingStationManager(db_path=self.db_path)
        self.recommender = RecommendationEngine(self.station_manager, model_path=self.model_path)
        self.dashboard = Dashboard(
            output_dir="outputs",
            fleet_size=self.fleet_size,
            station_count=self.station_count,
            tracked=self.tracked,
            dashboard_state_interval=self.dashboard_state_interval,
            dashboard_report_interval=self.dashboard_report_interval,
        )
        try:
            self.visualization = VisualizationIntegration(self)
        except Exception:
            self.visualization = None

        self.assignments: Dict[str, Assignment] = {}
        self.last_positions: Dict[str, tuple[float, float]] = {}
        # mapping between VehicleManager IDs and SUMO vehicle IDs (authoritative IDs are VM ids)
        self.vm_to_sumo: Dict[str, str] = {}
        self.sumo_to_vm: Dict[str, str] = {}
        # instrumentation counters
        self.injected_count = 0
        self.low_battery_count = 0
        self.rec_call_count = 0
        # logging collections
        self.recommendation_log: list[Dict[str, Any]] = []
        self.charging_events: list[Dict[str, Any]] = []
        self.tracked_lifecycle: Dict[str, Dict[str, Any]] = {}
        self.simulation_metrics: list[Dict[str, Any]] = []
        self.reward_curve: list[float] = []
        self._last_dashboard_state: Optional[Dict[str, Any]] = None
        self._monitoring_collectors = [
            SimulationCollector(),
            VehicleCollector(),
            RecommendationCollector(),
            StationCollector(),
            RLCollector(),
            SystemCollector(),
        ]
        self._monitoring_started_at = time.time()
        self._monitoring_last_step_time = time.time()
        self._monitoring_step_count = 0
        self._monitoring_episode = 0
        self._monitoring_last_reward = 0.0
        self.explainer = RecommendationExplainer()
        self.last_explanation: Optional[Dict[str, Any]] = None
        self.current_step = 0
        self.runtime_control = SimulationRuntimeControl()
        self._initial_battery_snapshot: Dict[str, tuple[float, float]] = {}
        self._invalid_vehicle_routes: list[dict[str, Any]] = []
        self._station_snapshot_cache: list[dict[str, Any]] = []
        self._station_snapshot_cache_by_id: dict[str, dict[str, Any]] = {}
        self._station_snapshot_cache_step: Optional[int] = None
        # Lightweight stage timing instrumentation for profiling and regression checks.
        self._timing_totals_ms: Dict[str, float] = {
            'simulation_step': 0.0,
            'update_vehicles': 0.0,
            'process_charging': 0.0,
            'refresh_dashboard': 0.0,
            'visualization_publish': 0.0,
            'collect_metrics': 0.0,
            'loop_total': 0.0,
        }
        self._timing_max_ms: Dict[str, float] = {key: 0.0 for key in self._timing_totals_ms}
        self._timing_counts: Dict[str, int] = {key: 0 for key in self._timing_totals_ms}

        # will be set when SUMO starts
        self.net = None
        self._sumo_stderr_log = Path("outputs") / "sumo_startup_stderr.log"
        self._sumo_process: Optional[subprocess.Popen] = None

    def _record_timing(self, stage: str, elapsed_ms: float) -> None:
        if stage not in self._timing_totals_ms:
            return
        self._timing_totals_ms[stage] += float(elapsed_ms)
        self._timing_counts[stage] += 1
        if elapsed_ms > self._timing_max_ms[stage]:
            self._timing_max_ms[stage] = float(elapsed_ms)

    def _timing_avg_ms(self, stage: str) -> float:
        count = self._timing_counts.get(stage, 0)
        if count <= 0:
            return 0.0
        return self._timing_totals_ms[stage] / count

    def _tail_file(self, path: Path, max_lines: int = 40) -> str:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return ""
        lines = text.splitlines()
        if not lines:
            return ""
        return "\n".join(lines[-max_lines:])

    def _force_kill_sumo_processes(self) -> None:
        if platform.system().lower().startswith("win"):
            for image_name in ("sumo.exe", "sumo-gui.exe"):
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/IM", image_name, "/T"],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                except OSError:
                    pass
            return
        for proc_name in ("sumo", "sumo-gui"):
            try:
                subprocess.run(["pkill", "-f", proc_name], check=False, capture_output=True, text=True)
            except OSError:
                pass

    def _start_traci_with_timeout(self, cmd: list[str], port: int, timeout_seconds: float = 20.0) -> None:
        """Start SUMO and connect over TraCI with a fallback path for Windows environments.

        Some local Windows setups fail when SUMO is launched as a detached subprocess with
        a fixed remote port, so this method first tries the explicit subprocess + traci.init
        path and then falls back to the simpler traci.start-based startup if that path is
        not establishing a connection within the timeout window.
        """
        launch_cmd = list(cmd)
        if "--remote-port" not in launch_cmd:
            launch_cmd.extend(["--remote-port", str(int(port))])

        try:
            traci.close()
        except Exception:
            pass
        time.sleep(0.2)

        self._sumo_process = subprocess.Popen(
            launch_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if platform.system().lower().startswith("win") else 0,
        )
        deadline = time.perf_counter() + max(1.0, float(timeout_seconds))
        last_error: Exception | None = None

        while time.perf_counter() < deadline:
            if self._sumo_process.poll() is not None:
                stderr_tail = self._tail_file(self._sumo_stderr_log)
                break
            try:
                traci.init(port=int(port), numRetries=1, host="127.0.0.1")
                return
            except Exception as exc:
                last_error = exc
                time.sleep(0.5)

        # If the explicit subprocess path did not establish a connection, try the simpler
        # traci.start path once before giving up. This often works on Windows when SUMO is
        # already available and the environment is able to negotiate the connection.
        try:
            traci.close()
        except Exception:
            pass
        try:
            if self._sumo_process and self._sumo_process.poll() is None:
                self._sumo_process.terminate()
                self._sumo_process.wait(timeout=5)
        except Exception:
            try:
                self._sumo_process.kill()
            except Exception:
                pass
        fallback_cmd = [cmd[0], "-c", str(self.sumo_cfg), "--no-step-log", "true", "--verbose", "false", "--time-to-teleport", "-1"]
        try:
            traci.start(fallback_cmd)
            return
        except Exception as fallback_error:
            stderr_tail = self._tail_file(self._sumo_stderr_log)
            raise TimeoutError(
                f"TraCI startup timed out after {timeout_seconds:.1f}s on port {port}. "
                f"Last connect error={last_error!r}. Fallback error={fallback_error!r}.\n"
                f"Command: {launch_cmd}\n"
                f"Fallback command: {fallback_cmd}\n"
                f"SUMO stderr log: {self._sumo_stderr_log}.\n{stderr_tail}"
            ) from fallback_error

    def _resolve_net_path_from_cfg(self) -> Optional[Path]:
        if not hasattr(self, 'sumo_cfg'):
            return None
        cfg_path = str(self.sumo_cfg)
        net_path = None
        try:
            tree = ET.parse(cfg_path)
            root = tree.getroot()
            for inp in root.findall('.//input'):
                nf = inp.find('net-file')
                if nf is not None and 'value' in nf.attrib:
                    net_path = nf.attrib['value']
                    break
        except Exception:
            return None
        if not net_path:
            return None
        cfg_dir = Path(cfg_path).parent
        resolved = (cfg_dir / net_path).resolve()
        return resolved if resolved.exists() else None

    def _ensure_loaded_net(self) -> None:
        if self.net is not None:
            return
        net_path = self._resolve_net_path_from_cfg()
        if net_path is None:
            return
        try:
            import sumolib

            self.net = sumolib.net.readNet(str(net_path))
        except Exception:
            self.net = None

    def _is_traci_connected(self) -> bool:
        checker = getattr(traci, 'isConnected', None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:
                pass
        try:
            traci.getConnection('default')
            return True
        except Exception:
            pass
        try:
            traci.simulation.getTime()
            return True
        except Exception:
            return False

    def _resolve_station_db_path(self) -> None:
        candidate = Path(self.db_path)
        if candidate.suffix.lower() == ".json":
            db_path = self.sumo_cfg.parent / "stations.sqlite"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.db_path = str(db_path)
            return
        if candidate.suffix.lower() in {"", ".db", ".sqlite", ".sqlite3"}:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            self.db_path = str(candidate)
            return
        self.db_path = str(candidate)

    def _resolve_model_path(self) -> None:
        candidate = Path(self.model_path)
        if candidate.is_absolute():
            self.model_path = str(candidate)
            return
        root = Path(__file__).resolve().parents[2]
        for path in [root / candidate, root / "runs" / "ppo_ckpt" / candidate.name, root / "runs" / "ppo_ckpt" / candidate, root / "runs" / "ppo_ckpt" / "ppo_ev_final.zip"]:
            if path.exists():
                self.model_path = str(path)
                return
        if candidate.name.endswith(".zip"):
            fallback = root / "runs" / "ppo_ckpt" / candidate.name
            if fallback.exists():
                self.model_path = str(fallback)
                return
        self.model_path = str(root / "runs" / "ppo_ckpt" / "ppo_ev_final.zip")

    def _initialize_tracked_lifecycle_state(self) -> None:
        self.tracked_lifecycle = {}
        for vehicle in self.vehicle_manager.list_tracked_vehicles():
            battery = round(float(getattr(vehicle, 'battery_pct', 0.0) or 0.0), 2)
            self.tracked_lifecycle[str(vehicle.vehicle_id)] = {
                'vehicle_id': str(vehicle.vehicle_id),
                'initial_battery': battery,
                'minimum_battery': battery,
                'recommendation_time': None,
                'recommended_station': None,
                'arrival_time': None,
                'queue_join_time': None,
                'queue_time': None,
                'charging_start': None,
                'charging_end': None,
                'final_battery': None,
                'resume_event': None,
                'last_updated_step': None,
                'state': 'TRAVELING',
                'state_history': [
                    {
                        'step': 0,
                        'state': 'TRAVELING',
                        'previous_state': None,
                        'reason': 'initialized',
                        'details': {},
                    }
                ],
            }

    def _tracked_lifecycle_entry(self, vehicle_id: str) -> Optional[Dict[str, Any]]:
        return self.tracked_lifecycle.get(str(vehicle_id))

    def _transition_tracked_lifecycle_state(self, vehicle_id: str, new_state: str, step: Optional[int] = None, reason: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> None:
        entry = self._tracked_lifecycle_entry(vehicle_id)
        if entry is None:
            return
        previous_state = entry.get('state')
        if previous_state == new_state:
            if step is not None:
                entry['last_updated_step'] = int(step)
            return
        entry['state'] = new_state
        if step is not None:
            entry['last_updated_step'] = int(step)
        state_history = entry.setdefault('state_history', [])
        state_history.append({
            'step': int(step) if step is not None else None,
            'state': new_state,
            'previous_state': previous_state,
            'reason': reason,
            'details': dict(details or {}),
        })
        self.charging_events.append({
            'event_type': 'lifecycle_state_transition',
            'step': int(step) if step is not None else None,
            'vehicle_id': str(vehicle_id),
            'previous_state': previous_state,
            'state': new_state,
            'reason': reason,
            'details': dict(details or {}),
        })

    def _update_tracked_battery_metrics(self, vehicle_id: str, battery_pct: float, step: Optional[int] = None) -> None:
        entry = self._tracked_lifecycle_entry(vehicle_id)
        if entry is None:
            return
        battery = round(float(battery_pct), 2)
        if entry.get('initial_battery') is None:
            entry['initial_battery'] = battery
        current_min = entry.get('minimum_battery')
        entry['minimum_battery'] = battery if current_min is None else min(float(current_min), battery)
        entry['final_battery'] = battery
        if step is not None:
            entry['last_updated_step'] = int(step)

    def _get_known_sumo_vehicle_ids(self) -> set[str]:
        known_ids: set[str] = set()
        try:
            known_ids.update(traci.vehicle.getIDList())
        except Exception:
            pass
        try:
            known_ids.update(traci.simulation.getLoadedIDList())
        except Exception:
            pass
        try:
            known_ids.update(traci.simulation.getDepartedIDList())
        except Exception:
            pass
        return known_ids

    def _invalidate_station_snapshots(self) -> None:
        self._station_snapshot_cache = []
        self._station_snapshot_cache_by_id = {}
        self._station_snapshot_cache_step = None

    def _get_station_snapshots(self, force_refresh: bool = False) -> list[dict[str, Any]]:
        current_step = getattr(self, 'current_step', 0)
        cached = getattr(self, '_station_snapshot_cache', [])
        cached_step = getattr(self, '_station_snapshot_cache_step', None)
        if not force_refresh and cached and cached_step == current_step:
            return cached
        snapshot_provider = getattr(self.station_manager, 'get_station_snapshots', None)
        if callable(snapshot_provider):
            snapshots = snapshot_provider(force_refresh=force_refresh)
        else:
            snapshots = []
            list_stations = getattr(self.station_manager, 'list_all_stations', None)
            get_state = getattr(self.station_manager, 'get_station_state', None)
            if callable(list_stations) and callable(get_state):
                for row in list_stations():
                    station_id = str(row.get('station_id'))
                    state = get_state(station_id) or {}
                    ports = list(state.get('ports') or [])
                    available_ports = sum(1 for port in ports if port.get('status') == 'Available')
                    occupied_ports = sum(1 for port in ports if port.get('status') in {'Charging', 'Preparing', 'SuspendedEV', 'SuspendedEVSE', 'Finishing'})
                    charging_ports = sum(1 for port in ports if port.get('status') == 'Charging')
                    total_ports = len(ports)
                    queue_length = max(0, occupied_ports - charging_ports)
                    grid_load_kw = sum(float(port.get('power_kw', 0.0) or 0.0) for port in ports if port.get('status') == 'Charging')
                    snapshots.append({
                        'station_id': station_id,
                        'name': row.get('name'),
                        'lat': state.get('lat', row.get('lat')),
                        'lon': state.get('lon', row.get('lon')),
                        'operator': row.get('operator'),
                        'address': row.get('address'),
                        'grid_zone_id': row.get('grid_zone_id'),
                        'data_source': row.get('data_source'),
                        'price_per_kwh': state.get('price_per_kwh'),
                        'ports': ports,
                        'connector_types': sorted({str(port.get('connector_type')) for port in ports if port.get('connector_type')}),
                        'available_ports': available_ports,
                        'occupied_ports': occupied_ports,
                        'charging_ports': charging_ports,
                        'total_ports': total_ports,
                        'queue_length': queue_length,
                        'avg_wait_estimate': state.get('avg_wait_estimate', 0.0),
                        'grid_load_kw': grid_load_kw,
                        'occupancy_rate': (occupied_ports / total_ports) if total_ports else 0.0,
                        'occupancy_percent': round(((occupied_ports / total_ports) if total_ports else 0.0) * 100.0, 2),
                        'charging_state': 'charging' if charging_ports else ('queued' if queue_length > 0 else 'idle'),
                    })
        self._station_snapshot_cache = snapshots
        self._station_snapshot_cache_by_id = {str(item.get("station_id")): item for item in snapshots}
        self._station_snapshot_cache_step = current_step
        return snapshots

    def _get_station_snapshot(self, station_id: str) -> Optional[dict[str, Any]]:
        current_step = getattr(self, 'current_step', 0)
        if not station_id:
            return None
        if not getattr(self, '_station_snapshot_cache_by_id', {}) or getattr(self, '_station_snapshot_cache_step', None) != current_step:
            self._get_station_snapshots(force_refresh=False)
        snapshot = self._station_snapshot_cache_by_id.get(str(station_id))
        if snapshot is not None:
            return snapshot
        get_state = getattr(self.station_manager, 'get_station_state', None)
        if callable(get_state):
            state = get_state(str(station_id))
            if state is not None:
                return state
        return None

    def start(self):
        # Clear any transient in-memory state from a prior local run on this controller instance.
        self.assignments.clear()

        sumo_binary = checkBinary('sumo-gui' if self.use_gui else 'sumo')
        self._sumo_stderr_log.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._sumo_stderr_log.write_text("", encoding="utf-8")
        except OSError:
            pass

        startup_port = getFreeSocketPort()
        if startup_port is None:
            raise RuntimeError('Failed to allocate free TraCI port for SUMO startup')

        sumo_cmd = [
            sumo_binary,
            '-c',
            str(self.sumo_cfg),
            '--error-log',
            str(self._sumo_stderr_log),
            '--no-step-log',
            'true',
            '--verbose',
            'false',
            '--time-to-teleport',
            '-1',
            '--seed',
            str(self.simulation_seed),
        ]
        if self.use_gui:
            # Ensure SUMO-GUI starts stepping immediately under TraCI control.
            sumo_cmd.append('--start')
        logger.info('Starting SUMO with command=%s on port=%s', sumo_cmd, startup_port)
        self._start_traci_with_timeout(sumo_cmd, port=int(startup_port), timeout_seconds=45.0)

        # Defer expensive net parsing until it is actually needed for station-edge snapping.
        self.net = None

        # ensure stations exist in DB
        self._ensure_stations()

        # reset persisted port states so each simulation run starts with real availability
        try:
            reset_summary = self.station_manager.reset_all_ports_to_available()
            logger.info(
                'Startup port reset: stations=%d total_ports=%d available_before=%d charging_before=%d available_after=%d charging_after=%d ports_reset=%d stale_vehicle_events_cleared=%d',
                int(reset_summary.get('station_count', 0)),
                int(reset_summary.get('total_ports', 0)),
                int(reset_summary.get('available_ports_before', 0)),
                int(reset_summary.get('charging_ports_before', 0)),
                int(reset_summary.get('available_ports_after', 0)),
                int(reset_summary.get('charging_ports_after', 0)),
                int(reset_summary.get('ports_reset', 0)),
                int(reset_summary.get('stale_vehicle_events_cleared', 0)),
            )
        except Exception:
            logger.exception('Failed to reset persisted station port states at startup')

        # let SUMO initialize the scenario vehicles/routes before we inspect them
        try:
            traci.simulationStep(0)
        except Exception:
            pass

        # load trained model if available
        if PPO and self.model_path and Path(self.model_path).exists():
            try:
                self.model = PPO.load(self.model_path)
            except Exception:
                self.model = None
        else:
            self.model = None

        # generate fleet and inject into SUMO (vehicle IDs ev_1..ev_N)
        if len(self.vehicle_manager.list_vehicles()) < self.fleet_size:
            self.vehicle_manager.generate_fleet(
                count=self.fleet_size,
                detail_count=self.tracked,
                random_seed=self.simulation_seed,
            )
        self._initialize_tracked_lifecycle_state()
        self._inject_vehicles_into_sumo()
        # verify 1:1 mapping between VehicleManager IDs and SUMO vehicle IDs
        mismatches = self._verify_vm_sumo_mapping()
        if mismatches:
            # stop startup if mapping not correct; include details for diagnosis
            raise RuntimeError('VM<->SUMO mapping verification failed: ' + '; '.join(mismatches))

    def stop(self):
        try:
            traci.close()
        except Exception:
            pass
        if self._sumo_process is not None:
            try:
                if self._sumo_process.poll() is None:
                    self._sumo_process.terminate()
                    self._sumo_process.wait(timeout=5)
            except Exception:
                try:
                    self._sumo_process.kill()
                except Exception:
                    pass
            finally:
                self._sumo_process = None

    def run(self, steps: int | None = None):
        if not traci.__name__:
            raise RuntimeError('TraCI not available')

        connected = self._is_traci_connected()
        if not connected:
            self.start()

        # prepare fleet
        if len(self.vehicle_manager.list_vehicles()) < self.fleet_size:
            self.vehicle_manager.generate_fleet(
                count=self.fleet_size,
                detail_count=self.tracked,
                random_seed=self.simulation_seed,
            )

        step = 0
        try:
            while steps is None or step < steps:
                if hasattr(self, 'runtime_control') and self.runtime_control is not None:
                    control = self.runtime_control.read()
                else:
                    control = {"status": "play", "speed": 1.0, "step_once": False, "reset": False}
                if control["reset"]:
                    logger.info('[TICK] reset requested')
                    self.stop()
                    self.vm_to_sumo.clear()
                    self.sumo_to_vm.clear()
                    self.last_positions.clear()
                    self.assignments.clear()
                    self.start()
                    step = 0
                    if hasattr(self, 'runtime_control') and self.runtime_control is not None:
                        self.runtime_control.consume_reset()
                    continue
                if control["status"] == "paused" and not control["step_once"]:
                    time.sleep(0.1)
                    continue

                start = time.perf_counter()
                sim_t0 = time.perf_counter()
                traci.simulationStep()
                self._record_timing('simulation_step', (time.perf_counter() - sim_t0) * 1000.0)
                try:
                    self.current_step = int(traci.simulation.getTime())
                except Exception:
                    self.current_step = step
                update_t0 = time.perf_counter()
                self._update_vehicles(step)
                self._record_timing('update_vehicles', (time.perf_counter() - update_t0) * 1000.0)
                charge_t0 = time.perf_counter()
                self._process_charging(step)
                self._record_timing('process_charging', (time.perf_counter() - charge_t0) * 1000.0)
                self._invalidate_station_snapshots()
                # Publish a fresh dashboard snapshot on every simulation step so the
                # state file reflects live telemetry instead of lagging behind the
                # current SUMO state.
                dashboard_t0 = time.perf_counter()
                self._refresh_dashboard()
                self._record_timing('refresh_dashboard', (time.perf_counter() - dashboard_t0) * 1000.0)
                # Throttle visualization publication to avoid per-step payload building,
                # network conversion, and persistence overhead.
                if step % self.visualization_interval == 0 and getattr(self, 'visualization', None) is not None:
                    vis_t0 = time.perf_counter()
                    try:
                        self.visualization.publish_step(step)
                    except Exception:
                        logger.exception('[WS] snapshot publication failed at step=%d', step)
                    self._record_timing('visualization_publish', (time.perf_counter() - vis_t0) * 1000.0)
                if step % 10 == 0:
                    logger.info('[TICK] step=%d sumo_vehicles=%d vm_vehicles=%d', step, len(self._get_known_sumo_vehicle_ids()), len(self.vehicle_manager.list_vehicles()))
                # Throttle metrics aggregation to reduce repeated station snapshot scans.
                if step % self.metrics_interval == 0:
                    metrics_t0 = time.perf_counter()
                    self._collect_metrics(step)
                    self._emit_runtime_metrics(step, time.perf_counter() - start)
                    self._record_timing('collect_metrics', (time.perf_counter() - metrics_t0) * 1000.0)
                self._record_timing('loop_total', (time.perf_counter() - start) * 1000.0)
                if step > 0 and step % 50 == 0:
                    logger.info(
                        '[TIMING] step=%d avg_ms sim=%.2f update=%.2f charge=%.2f dash=%.2f vis=%.2f metrics=%.2f loop=%.2f',
                        step,
                        self._timing_avg_ms('simulation_step'),
                        self._timing_avg_ms('update_vehicles'),
                        self._timing_avg_ms('process_charging'),
                        self._timing_avg_ms('refresh_dashboard'),
                        self._timing_avg_ms('visualization_publish'),
                        self._timing_avg_ms('collect_metrics'),
                        self._timing_avg_ms('loop_total'),
                    )
                if control["step_once"]:
                    if hasattr(self, 'runtime_control') and self.runtime_control is not None:
                        self.runtime_control.consume_step()
                else:
                    time.sleep(1.0 / float(control["speed"]))
                step += 1
        finally:
            # Close TraCI before publishing the terminal dashboard state so the
            # final JSON reflects both completed simulation time and connection state.
            self.stop()
            final_step = max(int(getattr(self, 'current_step', 0) or 0), int(step))
            final_state = dict(getattr(self, '_last_dashboard_state', {}) or {})
            final_simulation = dict(final_state.get('simulation') or {})
            final_simulation.update({
                'step': final_step,
                'time': final_step,
                'status': 'COMPLETED',
                'simulation_status': 'COMPLETED',
                'running': False,
                'paused': False,
                'connection': 'DISCONNECTED',
                'connection_status': 'DISCONNECTED',
            })
            final_state['simulation'] = final_simulation
            try:
                if getattr(self, 'dashboard', None) is not None:
                    self.dashboard.finalize(final_state)
            except Exception:
                logger.exception('Failed to finalize dashboard output')
            self._save_results()
            logger.info('Simulation instrumentation: injected=%d low_battery_triggers=%d rec_calls=%d', self.injected_count, self.low_battery_count, self.rec_call_count)
            logger.info('Simulation finished: step=%d status=COMPLETED connection=DISCONNECTED', final_step)

    def _get_known_sumo_vehicle_ids(self) -> set[str]:
        known_ids: set[str] = set()
        try:
            known_ids.update(traci.vehicle.getIDList())
        except Exception:
            pass
        try:
            known_ids.update(traci.simulation.getLoadedIDList())
        except Exception:
            pass
        try:
            known_ids.update(traci.simulation.getDepartedIDList())
        except Exception:
            pass
        return known_ids

    def _ensure_stations(self) -> None:
        if self.station_count <= 0:
            return
        # ensure there are at least station_count stations in DB
        rows = self.station_manager.list_all_stations()
        if len(rows) >= self.station_count:
            return
        conn = self.station_manager.repo.connection
        cur = conn.cursor()
        for i in range(len(rows), self.station_count):
            sid = f"sim_station_{i+1}"
            name = f"Sim Station {i+1}"
            lat = 12.9 + (i % 50) * 0.001
            lon = 77.5 + (i % 50) * 0.001
            cur.execute("INSERT OR REPLACE INTO stations (station_id,name,lat,lon,operator,address,grid_zone_id,data_source) VALUES (?,?,?,?,?,?,?,?)", (sid, name, lat, lon, 'Sim', 'Addr', 'zone_sim', 'simulated'))
            # create ports
            num_ports = 4
            for pidx in range(1, num_ports+1):
                port_id = f"{sid}_p{pidx}"
                cur.execute("INSERT OR REPLACE INTO ports (port_id,station_id,connector_type,power_kw,status,status_updated_at) VALUES (?,?,?,?,?,datetime('now'))", (port_id, sid, 'Type2', 22.0, 'Available'))
            # price — handle older DB schemas without tariff_period
            try:
                cur.execute("INSERT INTO price_history (station_id,price_per_kwh,tariff_period,recorded_at) VALUES (?,?,?,datetime('now'))", (sid, round(0.15 + (i%10)*0.01,3), 'non_solar_hour'))
            except Exception:
                try:
                    cur.execute("INSERT INTO price_history (station_id,price_per_kwh,recorded_at) VALUES (?,?,datetime('now'))", (sid, round(0.15 + (i%10)*0.01,3)))
                except Exception:
                    pass
        conn.commit()

    def _inject_vehicles_into_sumo(self) -> None:
        """Honor vehicles already present in the SUMO scenario and only add fallbacks if needed.

        For the Bangalore scenario, the .rou.xml already defines vehicles such as ev_1..ev_10.
        Re-adding them causes duplicate-ID errors and invalid-route failures. The controller should
        therefore map those existing SUMO vehicles to the VehicleManager fleet rather than forcing
        new insertions.
        """
        self._injected_success = 0
        self._inject_first_exception = None
        self._invalid_vehicle_routes.clear()

        vehicle_list = list(self.vehicle_manager.vehicles.keys())
        current_sumo_ids = self._get_known_sumo_vehicle_ids()

        # First, map any vehicles that already exist in the running SUMO scenario.
        for vid in vehicle_list:
            if vid in current_sumo_ids:
                self.vm_to_sumo[vid] = vid
                self.sumo_to_vm[vid] = vid
                self._injected_success += 1

        # If the scenario already provides vehicles, do not add more to avoid duplicate-ID issues.
        if self._injected_success:
            try:
                sumo_ids = traci.vehicle.getIDList()
                logger.info('Mapped %d existing SUMO vehicles to VehicleManager IDs', len(sumo_ids))
                self.injected_count = len(sumo_ids)
            except Exception:
                logger.info('Finished vehicle mapping')
            return

        # Fallback: if no vehicles are present, add the missing fleet entries one by one.
        # This remains defensive for non-Bangalore configs and keeps the controller from
        # crashing when a single route insert fails due to an invalid edge or route build.
        try:
            edges = traci.edge.getIDList()
        except Exception:
            self._inject_first_exception = 'edge_list_error'
            return

        if not edges:
            self._inject_first_exception = 'no_edges'
            return

        valid_edges = [e for e in edges if not e.startswith(':')]
        if not valid_edges:
            self._inject_first_exception = 'no_valid_edges'
            return

        try:
            existing_routes = list(traci.route.getIDList())
        except Exception:
            existing_routes = []

        route_candidates = existing_routes or [f'fallback_route_{idx}' for idx in range(3)]
        for vehicle_id in vehicle_list:
            if vehicle_id in current_sumo_ids or vehicle_id in self.vm_to_sumo:
                continue
            route_id = route_candidates[0] if route_candidates else 'fallback_route'
            try:
                if route_id not in existing_routes:
                    try:
                        source = valid_edges[0]
                        target = valid_edges[min(1, len(valid_edges) - 1)]
                        route = traci.simulation.findRoute(source, target)
                        route_edges = getattr(route, 'edges', None)
                        if route_edges:
                            traci.route.add(route_id, route_edges)
                    except Exception:
                        pass
                traci.vehicle.add(vehicle_id, route_id, depart='now', departPos='0', departLane='free')
                self.vm_to_sumo[vehicle_id] = vehicle_id
                self.sumo_to_vm[vehicle_id] = vehicle_id
                self._injected_success += 1
            except Exception as exc:
                self._invalid_vehicle_routes.append({'vehicle_id': vehicle_id, 'route_id': route_id, 'error': str(exc)})
                self._inject_first_exception = str(exc)
                logger.info('Skipping vehicle %s during SUMO injection: %s', vehicle_id, exc)
                continue

    def _verify_vm_sumo_mapping(self) -> list[str]:
        """Return list of mismatch descriptions (empty if all good).

        Checks that every VehicleManager vehicle has a mapping in `vm_to_sumo`, that
        the mapped SUMO id exists in the simulation, and that `sumo_to_vm` is the inverse.
        """
        mismatches: list[str] = []
        current_sumo_ids = self._get_known_sumo_vehicle_ids()

        vm_ids = set(self.vehicle_manager.vehicles.keys())

        # check every VM id has mapping
        for vm in sorted(vm_ids):
            sumo = self.vm_to_sumo.get(vm)
            if not sumo:
                if vm in current_sumo_ids:
                    self.vm_to_sumo[vm] = vm
                    self.sumo_to_vm[vm] = vm
                    sumo = vm
                else:
                    # Some vehicles can remain unloaded or fail to depart in the Bangalore scenario.
                    # Do not block startup for those; the simulation can still proceed with the vehicles that did depart.
                    continue
            if sumo not in current_sumo_ids:
                continue
            # check inverse mapping
            mapped_vm = self.sumo_to_vm.get(sumo)
            if mapped_vm != vm:
                mismatches.append(f"inverse_mapping_mismatch:sumo={sumo} maps_to={mapped_vm} expected={vm}")
                return mismatches

        # ensure there are no duplicate SUMO mappings
        mapped_sumos = [self.vm_to_sumo[vm] for vm in vm_ids if vm in self.vm_to_sumo]
        if len(mapped_sumos) != len(set(mapped_sumos)):
            mismatches.append('duplicate_sumo_ids_in_vm_to_sumo')
            return mismatches

        return mismatches


    def _update_vehicles(self, step: int) -> None:
        # Ensure VM->SUMO mapping and update positions/battery for managed vehicles
        current_sumo_ids = self._get_known_sumo_vehicle_ids()

        # cache available routes for adding missing vehicles
        try:
            available_routes = traci.route.getIDList()
        except Exception:
            available_routes = []

        for vrec in self.vehicle_manager.list_vehicles():
            vm_vid = vrec.vehicle_id
            sumo_vid = self.vm_to_sumo.get(vm_vid)

            # if not mapped, check if vehicle already present in SUMO under the same id
            if sumo_vid is None:
                if vm_vid in current_sumo_ids:
                    sumo_vid = vm_vid
                    self.vm_to_sumo[vm_vid] = sumo_vid
                    self.sumo_to_vm[sumo_vid] = vm_vid
                else:
                    # Do not add new vehicles dynamically. Use only the SUMO vehicles that already exist.
                    continue

            # now we have a sumo_vid for this VM vehicle (or skip)
            if sumo_vid not in current_sumo_ids:
                # vehicle not active in SUMO this step
                continue

            # update position and battery
            try:
                pos = traci.vehicle.getPosition(sumo_vid)
            except Exception:
                continue

            last = self.last_positions.get(vm_vid)
            self.last_positions[vm_vid] = pos
            if last is None:
                continue

            dist_m = math.hypot(pos[0] - last[0], pos[1] - last[1])
            dist_km = dist_m / 1000.0

            # approximate consumption per vehicle from vehicle record
            if vrec.remaining_range_km and vrec.remaining_range_km > 0:
                consumption_wh_per_km = (vrec.battery_pct / 100.0) * vrec.battery_capacity_kwh * 1000.0 / max(0.1, vrec.remaining_range_km)
            else:
                consumption_wh_per_km = 200.0

            energy_kwh = dist_km * consumption_wh_per_km / 1000.0
            pct_drop = (energy_kwh / max(0.1, vrec.battery_capacity_kwh)) * 100.0
            vrec.battery_pct = max(0.0, vrec.battery_pct - pct_drop)
            vrec.remaining_range_km = max(0.0, vrec.remaining_range_km - dist_km)
            self._update_tracked_battery_metrics(vm_vid, vrec.battery_pct, step)

            # detect below threshold and not already assigned.
            # For larger fleets we only trigger re-routing for tracked vehicles or a small
            # bounded fallback batch so the controller remains responsive under 1000+ EV load.
            tracked_ids = {vehicle.vehicle_id for vehicle in self.vehicle_manager.list_tracked_vehicles()}
            if self._should_seek_charger(vrec) and vm_vid not in self.assignments:
                if tracked_ids and vm_vid not in tracked_ids:
                    continue
                if not tracked_ids and self.low_battery_count >= max(10, self.tracked):
                    continue
                self.low_battery_count += 1
                self._handle_low_battery(vm_vid, vrec, step)

    def _should_seek_charger(self, vehicle: VehicleRecord) -> bool:
        """Return whether the configured low-SOC charging trigger is reached."""
        return float(vehicle.battery_pct) <= float(self.charge_threshold_pct)

    def _coerce_station_id(self, station_id: str | None, *, strict: bool = False) -> Optional[str]:
        if not station_id:
            return None
        raw_station_id = str(station_id).strip()
        if not raw_station_id:
            return None

        resolve_method = getattr(self.station_manager, "resolve_station_id", None)
        if callable(resolve_method):
            resolved = resolve_method(raw_station_id)
            if resolved:
                return str(resolved)

        snapshots = self._get_station_snapshots(force_refresh=False)
        for snapshot in snapshots:
            if str(snapshot.get("station_id") or "") == raw_station_id:
                return str(snapshot.get("station_id"))

        if strict:
            return None

        list_method = getattr(self.station_manager, "list_all_stations", None)
        if callable(list_method):
            stations = list(list_method())
            if stations:
                first_station = str(stations[0].get("station_id") or "")
                if first_station:
                    return first_station

        return raw_station_id

    def _validate_station_route(self, vehicle_id: str, station_id: str | None, target_edge_id: Optional[str] = None) -> tuple[bool, str]:
        resolved_station_id = self._coerce_station_id(station_id, strict=True)
        if not resolved_station_id:
            return False, "station_not_found"

        station_snapshot = self._get_station_snapshot(resolved_station_id)
        if station_snapshot is None:
            return False, f"station_not_found:{resolved_station_id}"

        if target_edge_id:
            return True, "route_target_edge_available"

        if self.net is None:
            return True, "network_unavailable"

        try:
            import traci
        except Exception:
            return True, "traci_unavailable"

        lat = station_snapshot.get("lat")
        lon = station_snapshot.get("lon")
        if lat is None or lon is None:
            return True, "station_location_unavailable"

        try:
            lane_id, _ = self.station_manager._snap_station_to_lane(self.net, float(lat), float(lon), max_snap_distance_m=200.0)
            target_edge = self.net.getLane(lane_id).getEdge().getID()
        except Exception as exc:
            return False, f"station_edge_unresolved:{exc}"

        # The station was snapped to a real edge in the loaded Bengaluru SUMO
        # network. Avoid a second findRoute probe here: SUMO's default routing
        # vehicle class rejects some valid station destination edges and emits
        # misleading warnings such as "simulation.findRoute is not allowed".
        try:
            if target_edge in set(traci.edge.getIDList()):
                return True, "station_edge_valid"
        except Exception:
            pass
        return False, "station_edge_not_in_network"

    def _build_recommendation_log_entry(
        self,
        *,
        step: int,
        vehicle_id: str,
        vehicle_state: dict[str, Any],
        station_id: str | None,
        station_metrics: Optional[dict[str, Any]],
        charging_price_per_kwh: Optional[float],
        estimated_session_cost: Optional[float],
        rec: dict[str, Any],
        ppo_reward: Optional[float],
    ) -> dict[str, Any]:
        resolved_station_id = self._coerce_station_id(station_id)
        resolved_station_metrics = station_metrics
        if resolved_station_metrics is None and resolved_station_id:
            resolved_station_metrics = self._get_station_snapshot(resolved_station_id)
        if resolved_station_metrics is None and resolved_station_id and hasattr(self.station_manager, "get_station_metrics"):
            resolved_station_metrics = self.station_manager.get_station_metrics(resolved_station_id)

        available_ports = rec.get("available_ports")
        if available_ports is None and isinstance(resolved_station_metrics, dict):
            available_ports = resolved_station_metrics.get("available_ports")

        queue_length = rec.get("queue_length", rec.get("queue_len"))
        if queue_length is None and isinstance(resolved_station_metrics, dict):
            queue_length = resolved_station_metrics.get("queue_length")

        estimated_wait_min = rec.get("waiting_time")
        if estimated_wait_min is None and isinstance(resolved_station_metrics, dict):
            estimated_wait_min = resolved_station_metrics.get("avg_wait_estimate")

        station_load_kw = rec.get("grid_load")
        if station_load_kw is None and isinstance(resolved_station_metrics, dict):
            station_load_kw = resolved_station_metrics.get("grid_load_kw")

        return {
            'step': step,
            'vehicle_id': vehicle_id,
            'battery_pct': float(vehicle_state.get('battery_pct', 0.0)),
            'selected_station': resolved_station_id,
            'selected_station_name': (resolved_station_metrics or {}).get('name') if resolved_station_metrics else None,
            'distance_km': rec.get('travel_distance'),
            'travel_time': rec.get('travel_time'),
            'waiting_time': rec.get('waiting_time'),
            'queue_length': queue_length,
            'charging_price_per_kwh': charging_price_per_kwh,
            'estimated_session_cost': estimated_session_cost,
            'charging_cost': rec.get('charging_cost'),
            'grid_load': rec.get('grid_load'),
            'available_ports': available_ports,
            'distance_km': rec.get('travel_distance'),
            'price_per_kwh': charging_price_per_kwh,
            'estimated_wait_min': estimated_wait_min,
            'station_load_kw': station_load_kw,
            'recommendation_score': rec.get('recommendation_score') or ppo_reward,
            'ppo_action': rec.get('action'),
            'ppo_reward': ppo_reward,
            'recommendation_timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'recommendation_reason': rec.get('recommendation_reason'),
            'decision_factors': {
                'distance_km': rec.get('travel_distance'),
                'price_per_kwh': charging_price_per_kwh,
                'waiting_time_min': estimated_wait_min,
                'available_ports': available_ports,
                'battery_pct': vehicle_state.get('battery_pct'),
            },
            'ppo_observation': {
                'vehicle_state': {
                    'battery_pct': float(vehicle_state.get('battery_pct', 0.0)),
                    'battery_capacity_kwh': float(vehicle_state.get('battery_capacity_kwh', 0.0)),
                    'remaining_range_km': float(vehicle_state.get('remaining_range_km', 0.0)),
                },
                'station_features': {
                    'station_id': resolved_station_id,
                    'travel_distance': rec.get('travel_distance'),
                    'travel_time': rec.get('travel_time'),
                    'waiting_time': rec.get('waiting_time'),
                    'queue_length': queue_length,
                    'available_ports': available_ports,
                    'grid_load': station_load_kw,
                    'charging_price_per_kwh': charging_price_per_kwh,
                },
            },
        }

    def _handle_low_battery(self, vid: str, vrec: VehicleRecord, step: int) -> None:
        self._ensure_loaded_net()
        # Build vehicle_state dict expected by recommender
        vehicle_state = {
            'battery_pct': vrec.battery_pct,
            'battery_capacity_kwh': vrec.battery_capacity_kwh,
            'remaining_range_km': vrec.remaining_range_km,
        }
        # Use live SUMO position so station distance features are based on the
        # actual vehicle location instead of default placeholders.
        try:
            sumo_vid = self.vm_to_sumo.get(vid, vid)
            pos_x, pos_y = traci.vehicle.getPosition(sumo_vid)
            lon, lat = traci.simulation.convertGeo(pos_x, pos_y)
            vehicle_state['lat'] = float(lat)
            vehicle_state['lon'] = float(lon)
        except Exception:
            pass
        try:
            rec = self.recommender.recommend(vehicle_state, model_path=self.model_path)
            # instrumentation: recommendation was called
            self.rec_call_count += 1
        except Exception as e:
            logger.exception('Recommendation failed for %s: %s', vid, e)
            return

        # Log full recommendation details
        logger.info('Recommendation result vehicle=%s battery=%.1f -> %s', vid, vrec.battery_pct, rec)
        logger.info('[PPO] vehicle=%s observation=%s action=%s station=%s reward=%s battery=%s reason=%s', vid, rec.get('observation'), rec.get('action'), rec.get('station'), rec.get('ppo_reward'), vrec.battery_pct, rec.get('recommendation_reason'))

        station_id = rec.get('station')
        if not station_id:
            logger.info('Recommendation did not provide station for vehicle=%s', vid)
            return

        validated_station_id = self._coerce_station_id(station_id)
        if not validated_station_id:
            logger.info('Recommendation did not resolve to a known station for vehicle=%s', vid)
            return

        route_valid, route_reason = self._validate_station_route(vid, validated_station_id)
        if not route_valid:
            logger.info('Rejecting unreachable or invalid station %s for vehicle=%s: %s', validated_station_id, vid, route_reason)
            return

        station_id = validated_station_id

        # log the recommendation
        logger.info('Recommendation vehicle=%s battery=%.1f station=%s waiting=%s queue=%s price=%s grid_load=%s available_ports=%s', vid, vrec.battery_pct, station_id, rec.get('waiting_time'), rec.get('queue_length') or rec.get('queue_len'), rec.get('charging_cost'), rec.get('grid_load'), rec.get('available_ports') or rec.get('available_ports'))

        self.charging_events.append({
            'event_type': 'battery_low',
            'step': step,
            'vehicle_id': vid,
            'battery_pct': round(float(vrec.battery_pct), 2),
        })
        tracked_entry = self._tracked_lifecycle_entry(vid)
        if tracked_entry is not None and tracked_entry.get('recommendation_time') is None:
            tracked_entry['recommendation_time'] = int(step)
        if tracked_entry is not None:
            self._transition_tracked_lifecycle_state(
                vid,
                'SEEKING_CHARGER',
                step=step,
                reason='battery_low',
                details={'battery_pct': round(float(vrec.battery_pct), 2), 'station_id': station_id},
            )
        self.charging_events.append({
            'event_type': 'recommendation_selected',
            'step': step,
            'vehicle_id': vid,
            'station_id': station_id,
            'battery_pct': round(float(vrec.battery_pct), 2),
            'queue_length': rec.get('queue_length', rec.get('queue_len', 0)),
            'waiting_time': rec.get('waiting_time'),
            'available_ports': rec.get('available_ports'),
        })
        if tracked_entry is not None:
            tracked_entry['recommended_station'] = station_id
            self._update_tracked_battery_metrics(vid, vrec.battery_pct, step)

        # persist event
        try:
            conn = self.station_manager.repo.connection
            conn.execute('CREATE TABLE IF NOT EXISTS vehicle_events (id INTEGER PRIMARY KEY AUTOINCREMENT, vehicle_id TEXT, station_id TEXT, port_id TEXT, battery_pct REAL, step INTEGER, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
            conn.execute('INSERT INTO vehicle_events (vehicle_id, station_id, port_id, battery_pct, step) VALUES (?,?,?,?,?)', (vid, station_id, None, vrec.battery_pct, step))
            conn.commit()
        except Exception:
            logger.exception('Failed to log vehicle event')

        target_edge_id = None

        def _resolve_station_edge(lat: float, lon: float) -> Optional[str]:
            if self.net is None:
                return None
            # First try strict local snap, then widen search if station coordinates are slightly off-network.
            for radius_m in (200.0, 1000.0, 5000.0, 20000.0):
                try:
                    lane_id, _ = self.station_manager._snap_station_to_lane(self.net, lat, lon, max_snap_distance_m=radius_m)
                    return self.net.getLane(lane_id).getEdge().getID()
                except Exception:
                    continue
            return None

        st_state = self._get_station_snapshot(station_id)
        if st_state and self.net is not None:
            lat = st_state.get('lat')
            lon = st_state.get('lon')
            if lat is not None and lon is not None:
                target_edge_id = _resolve_station_edge(float(lat), float(lon))

        original_destination = vrec.destination
        try:
            sumo_vid_for_route = self.vm_to_sumo.get(vid, vid)
            route_edges = traci.vehicle.getRoute(sumo_vid_for_route)
            if route_edges:
                original_destination = route_edges[-1]
        except Exception:
            pass

        # Persist recommendation-side metrics with the assignment so tracked dashboard
        # rows can show the values consistently as the lifecycle progresses.
        recommended_distance_km = float(rec.get('travel_distance') or 0.0)
        recommended_travel_time_min = float(rec.get('travel_time') or 0.0)
        recommended_waiting_time_min = float(rec.get('waiting_time') or 0.0)
        recommended_queue_length = int(rec.get('queue_length', rec.get('queue_len', 0)) or 0)

        station_metrics = self._get_station_snapshot(station_id)
        charging_price_per_kwh = None
        if station_metrics is not None:
            raw_price = station_metrics.get('price_per_kwh')
            if raw_price is not None:
                charging_price_per_kwh = float(raw_price)
        if charging_price_per_kwh is None:
            energy_estimate = max(0.0, energy_needed_kwh(vrec.battery_pct, 80.0, vrec.battery_capacity_kwh))
            recommendation_cost = rec.get('charging_cost')
            if recommendation_cost is not None and energy_estimate > 0:
                charging_price_per_kwh = float(recommendation_cost) / energy_estimate

        estimated_session_cost = None
        if charging_price_per_kwh is not None:
            estimated_energy_kwh = max(0.0, energy_needed_kwh(vrec.battery_pct, 80.0, vrec.battery_capacity_kwh))
            estimated_session_cost = estimated_energy_kwh * charging_price_per_kwh

        self.assignments[vid] = Assignment(
            station_id=station_id,
            port_id=None,
            reserved_at_step=step,
            charging_end_step=None,
            original_destination=original_destination,
            status='enroute',
            target_edge_id=target_edge_id,
            recommended_distance_km=recommended_distance_km,
            recommended_travel_time_min=recommended_travel_time_min,
            recommended_waiting_time_min=recommended_waiting_time_min,
            recommended_queue_length=recommended_queue_length,
            charging_price_per_kwh=charging_price_per_kwh,
            estimated_session_cost=estimated_session_cost,
        )

        tracked_entry = self._tracked_lifecycle_entry(vid)
        if tracked_entry is not None:
            self._transition_tracked_lifecycle_state(
                vid,
                'ROUTING_TO_STATION',
                step=step,
                reason='reroute_to_station',
                details={'station_id': station_id, 'target_edge': target_edge_id},
            )

        # change vehicle target to station edge if possible
        try:
            sumo_vid = self.vm_to_sumo.get(vid, vid)
            try:
                current_ids = set(traci.vehicle.getIDList())
            except Exception:
                current_ids = set()
            if sumo_vid not in current_ids:
                return

            # find nearest edge to station using SUMO net (if available)
            st_state = self._get_station_snapshot(station_id) if station_id else None
            known_edges = set()
            try:
                known_edges.update(traci.edge.getIDList())
            except Exception:
                pass
            if self.net is not None:
                try:
                    known_edges.update(edge.getID() for edge in self.net.getEdges())
                except Exception:
                    pass

            if target_edge_id and target_edge_id in known_edges:
                traci.vehicle.changeTarget(sumo_vid, target_edge_id)
                self.charging_events.append({
                    'event_type': 'reroute_to_station',
                    'step': step,
                    'vehicle_id': vid,
                    'station_id': station_id,
                    'target_edge': target_edge_id,
                })
            elif st_state and self.net is not None:
                lat = st_state.get('lat')
                lon = st_state.get('lon')
                if lat is not None and lon is not None:
                    edge = _resolve_station_edge(float(lat), float(lon))
                    if edge and edge in known_edges:
                        try:
                            traci.vehicle.changeTarget(sumo_vid, edge)
                            self.assignments[vid].target_edge_id = edge
                            self.charging_events.append({
                                'event_type': 'reroute_to_station',
                                'step': step,
                                'vehicle_id': vid,
                                'station_id': station_id,
                                'target_edge': edge,
                            })
                        except Exception:
                            pass
            else:
                # best-effort fallback: try using station id (may not work)
                try:
                    if st_state and st_state.get('station_id') in known_edges:
                        traci.vehicle.changeTarget(sumo_vid, st_state.get('station_id'))
                except Exception:
                    pass
        except Exception:
            # ignore routing failures
            pass
        # record recommendation log entry
        rec_entry = self._build_recommendation_log_entry(
            step=step,
            vehicle_id=vid,
            vehicle_state=vehicle_state,
            station_id=station_id,
            station_metrics=station_metrics,
            charging_price_per_kwh=charging_price_per_kwh,
            estimated_session_cost=estimated_session_cost,
            rec=rec,
            ppo_reward=None,
        )
        # compute PPO reward estimate by stepping a local env if model available
        try:
            if hasattr(self, 'model') and self.model is not None:
                from src.rl_env.gym_ev_charging_env import GymEVChargingEnv
                env = GymEVChargingEnv(db_path=None, station_manager=self.station_manager, tracked_vehicle_count=self.tracked, candidate_count=rec.get('stations', None) and len(rec.get('stations')) or 8)
                obs, _ = env.reset()
                # set vehicle obs as in recommender and match the trained policy's expected shape
                vehicles_arr = obs['vehicles'].copy()
                if vehicles_arr.shape[0] != 10:
                    padded = np.zeros((10, vehicles_arr.shape[1]), dtype=np.float32)
                    copy_rows = min(10, vehicles_arr.shape[0])
                    padded[:copy_rows, :] = vehicles_arr[:copy_rows, :]
                    vehicles_arr = padded
                vehicles_arr[0, 0] = float(vrec.battery_pct)
                vehicles_arr[0, 1] = float(vrec.battery_capacity_kwh)
                obs_for_model = {'vehicles': vehicles_arr, 'stations': obs['stations']}
                action, _ = self.model.predict(obs_for_model, deterministic=True)
                rec_entry['ppo_action'] = int(action)
                rec_entry['ppo_observation'] = {
                    **rec_entry['ppo_observation'],
                    'model_observation': {
                        'vehicles_shape': list(obs_for_model['vehicles'].shape),
                        'stations_shape': list(obs_for_model['stations'].shape),
                        'action': int(action),
                    },
                }
                _, reward, _, _, _ = env.step(action)
                rec_entry['ppo_reward'] = float(reward)
                rec_entry['recommendation_score'] = rec_entry.get('recommendation_score') or float(reward)
                self.reward_curve.append(float(reward))
                try:
                    env.close()
                except Exception:
                    pass
        except Exception:
            logger.exception('Failed to compute PPO reward estimate')

        self.recommendation_log.append(rec_entry)

        try:
            explanation = self.explainer.explain_recommendation(
                vehicle_state=vehicle_state,
                recommendation={
                    "station": station_id,
                    "travel_distance": rec.get("travel_distance", 0.0),
                    "waiting_time": rec.get("waiting_time", 0.0),
                    "queue_length": rec.get("queue_length", 0),
                    "charging_cost": rec.get("charging_cost", 0.0),
                    "available_ports": rec.get("available_ports", 0),
                    "grid_load": rec.get("grid_load", 0.0),
                    "ppo_reward": rec_entry.get("ppo_reward"),
                },
                station_manager=self.station_manager,
            )
            self.last_explanation = explanation
            try:
                self.explainer.export_explanation_json(explanation, Path("outputs") / "latest_explanation.json")
            except Exception:
                pass
        except Exception:
            self.last_explanation = None

    def _process_charging(self, step: int) -> None:
        # check vehicles assigned
        for vid, assign in list(self.assignments.items()):
            if not hasattr(assign, 'charging_price_per_kwh'):
                assign.charging_price_per_kwh = None
            if not hasattr(assign, 'estimated_session_cost'):
                assign.estimated_session_cost = None
            if not hasattr(assign, 'charged_energy_kwh'):
                assign.charged_energy_kwh = 0.0
            if not hasattr(assign, 'session_charging_cost'):
                assign.session_charging_cost = 0.0

            try:
                # check if vehicle at assigned station (compare road id to station lane if available)
                # map vid to SUMO id
                sumo_vid = self.vm_to_sumo.get(vid, vid)
                try:
                    current_ids = set(traci.vehicle.getIDList())
                except Exception:
                    current_ids = set()
                if sumo_vid not in current_ids:
                    continue
                vroad = traci.vehicle.getRoadID(sumo_vid)
            except Exception:
                continue

            st_state = self._get_station_snapshot(assign.station_id) if assign.station_id else None
            at_station = False
            target_edge = getattr(assign, 'target_edge_id', None)
            if target_edge and vroad:
                at_station = (
                    vroad == target_edge or
                    vroad.startswith(f'{target_edge}_') or
                    target_edge.startswith(f'{vroad}_')
                )

            if at_station and assign.status == 'enroute':
                assign.arrived_step = step
                self.charging_events.append({
                    'event_type': 'arrived_at_station',
                    'step': step,
                    'vehicle_id': vid,
                    'station_id': assign.station_id,
                    'road_id': vroad,
                })
                tracked_entry = self._tracked_lifecycle_entry(vid)
                if tracked_entry is not None and tracked_entry.get('arrival_time') is None:
                    tracked_entry['arrival_time'] = int(step)
                if not assign.port_id and hasattr(self.station_manager, 'reserve_port'):
                    assign.port_id = self.station_manager.reserve_port(assign.station_id)
                    if assign.port_id:
                        self._invalidate_station_snapshots()
                if assign.port_id:
                    assign.status = 'charging'
                    st_state = self._get_station_snapshot(assign.station_id) if assign.station_id else None
                    port = None
                    for p in (st_state.get('ports') or []):
                        if p.get('port_id') == assign.port_id:
                            port = p
                            break
                    power_kw = float(port.get('power_kw', 22.0)) if port else 22.0
                    vrec = self.vehicle_manager.get_vehicle(vid)
                    if vrec:
                        energy_needed = max(0.0, energy_needed_kwh(vrec.battery_pct, 80.0, vrec.battery_capacity_kwh))
                        seconds = max(1, int((energy_needed / max(0.1, power_kw)) * 3600))
                        assign.charging_end_step = step + seconds
                        try:
                            # Hold vehicle at station while charging; release on completion.
                            traci.vehicle.setSpeed(sumo_vid, 0.0)
                        except Exception:
                            pass
                    battery_now = round(float(vrec.battery_pct), 2) if vrec else None
                    self.charging_events.append({
                        'event_type': 'charging_started',
                        'step': step,
                        'vehicle_id': vid,
                        'station_id': assign.station_id,
                        'port_id': assign.port_id,
                        'estimated_end': assign.charging_end_step,
                        'power_kw': power_kw,
                        'charging_price_per_kwh': assign.charging_price_per_kwh,
                        'estimated_session_cost': assign.estimated_session_cost,
                        'battery_pct': battery_now,
                    })
                    tracked_entry = self._tracked_lifecycle_entry(vid)
                    if tracked_entry is not None:
                        queue_join_time = tracked_entry.get('queue_join_time')
                        tracked_entry['charging_start'] = int(step)
                        if queue_join_time is not None:
                            tracked_entry['queue_time'] = int(step) - int(queue_join_time)
                        elif tracked_entry.get('queue_time') is None:
                            tracked_entry['queue_time'] = 0
                    self._transition_tracked_lifecycle_state(
                        vid,
                        'CHARGING',
                        step=step,
                        reason='charging_started',
                        details={'station_id': assign.station_id, 'port_id': assign.port_id},
                    )
                else:
                    assign.status = 'waiting'
                    queue_position = 1 + sum(
                        1
                        for other_vid, other_assign in self.assignments.items()
                        if other_vid != vid and other_assign.station_id == assign.station_id and other_assign.status == 'waiting'
                    )
                    self.charging_events.append({
                        'event_type': 'queue_joined',
                        'step': step,
                        'vehicle_id': vid,
                        'station_id': assign.station_id,
                        'queue_position': queue_position,
                    })
                    tracked_entry = self._tracked_lifecycle_entry(vid)
                    if tracked_entry is not None and tracked_entry.get('queue_join_time') is None:
                        tracked_entry['queue_join_time'] = int(step)
                    self._transition_tracked_lifecycle_state(
                        vid,
                        'QUEUED',
                        step=step,
                        reason='queue_joined',
                        details={'station_id': assign.station_id, 'queue_position': queue_position},
                    )

            if assign.status == 'waiting':
                if hasattr(self.station_manager, 'reserve_port'):
                    assign.port_id = self.station_manager.reserve_port(assign.station_id)
                    if assign.port_id:
                        self._invalidate_station_snapshots()
                if assign.port_id:
                    assign.status = 'charging'
                    sumo_vid = self.vm_to_sumo.get(vid, vid)
                    vroad = None
                    try:
                        vroad = traci.vehicle.getRoadID(sumo_vid)
                    except Exception:
                        vroad = None
                    st_state = self._get_station_snapshot(assign.station_id) if assign.station_id else None
                    port = None
                    for p in (st_state.get('ports') or []):
                        if p.get('port_id') == assign.port_id:
                            port = p
                            break
                    power_kw = float(port.get('power_kw', 22.0)) if port else 22.0
                    vrec = self.vehicle_manager.get_vehicle(vid)
                    if vrec:
                        energy_needed = max(0.0, energy_needed_kwh(vrec.battery_pct, 80.0, vrec.battery_capacity_kwh))
                        seconds = max(1, int((energy_needed / max(0.1, power_kw)) * 3600))
                        assign.charging_end_step = step + seconds
                        try:
                            # Hold vehicle at station while charging; release on completion.
                            traci.vehicle.setSpeed(sumo_vid, 0.0)
                        except Exception:
                            pass
                    battery_now = round(float(vrec.battery_pct), 2) if vrec else None
                    self.charging_events.append({
                        'event_type': 'charging_started',
                        'step': step,
                        'vehicle_id': vid,
                        'station_id': assign.station_id,
                        'port_id': assign.port_id,
                        'estimated_end': assign.charging_end_step,
                        'power_kw': power_kw,
                        'charging_price_per_kwh': assign.charging_price_per_kwh,
                        'estimated_session_cost': assign.estimated_session_cost,
                        'battery_pct': battery_now,
                    })
                    tracked_entry = self._tracked_lifecycle_entry(vid)
                    if tracked_entry is not None:
                        queue_join_time = tracked_entry.get('queue_join_time')
                        tracked_entry['charging_start'] = int(step)
                        if queue_join_time is not None:
                            tracked_entry['queue_time'] = int(step) - int(queue_join_time)
                        elif tracked_entry.get('queue_time') is None:
                            tracked_entry['queue_time'] = 0

            if assign.status == 'charging' and assign.port_id:
                st_state = self._get_station_snapshot(assign.station_id) if assign.station_id else None
                power_kw = 22.0
                if st_state:
                    for p in (st_state.get('ports') or []):
                        if p.get('port_id') == assign.port_id:
                            power_kw = float(p.get('power_kw', 22.0))
                            break
                vrec = self.vehicle_manager.get_vehicle(vid)
                if vrec and vrec.battery_capacity_kwh > 0:
                    charged_kwh = power_kw / 3600.0
                    pct_gain = (charged_kwh / vrec.battery_capacity_kwh) * 100.0
                    if pct_gain > 0:
                        assign.charged_energy_kwh += charged_kwh
                        if assign.charging_price_per_kwh is not None:
                            assign.session_charging_cost += charged_kwh * assign.charging_price_per_kwh
                        vrec.battery_pct = min(100.0, vrec.battery_pct + pct_gain)
                        vrec.remaining_range_km = vrec.battery_capacity_kwh * 1000.0 / 200.0 * (vrec.battery_pct / 100.0)
                        vrec.charging_requirement_kwh = max(0.0, energy_needed_kwh(vrec.battery_pct, 80.0, vrec.battery_capacity_kwh))
                        if step % 20 == 0:
                            self.charging_events.append({
                                'event_type': 'charging_progress',
                                'step': step,
                                'vehicle_id': vid,
                                'station_id': assign.station_id,
                                'port_id': assign.port_id,
                                'battery_pct': round(float(vrec.battery_pct), 2),
                                'session_charging_cost': round(assign.session_charging_cost, 4),
                            })
                    self._update_tracked_battery_metrics(vid, vrec.battery_pct, step)

            should_complete = False
            if assign.status == 'charging':
                vrec = self.vehicle_manager.get_vehicle(vid)
                if vrec and vrec.battery_pct >= 80.0:
                    should_complete = True
                if assign.charging_end_step and step >= assign.charging_end_step:
                    should_complete = True

            if should_complete:
                # complete charging
                if assign.port_id:
                    try:
                        self.station_manager.release_port(assign.port_id)
                        self._invalidate_station_snapshots()
                    except Exception:
                        pass
                # update vehicle battery to at least 80% (target)
                vrec = self.vehicle_manager.get_vehicle(vid)
                if vrec:
                    vrec.battery_pct = max(vrec.battery_pct, 80.0)
                    # recompute remaining range (assume 200 Wh/km average)
                    vrec.remaining_range_km = vrec.battery_capacity_kwh * 1000.0 / 200.0 * (vrec.battery_pct / 100.0)
                    vrec.charging_requirement_kwh = max(0.0, energy_needed_kwh(vrec.battery_pct, 80.0, vrec.battery_capacity_kwh))

                self._transition_tracked_lifecycle_state(
                    vid,
                    'CHARGED',
                    step=step,
                    reason='charging_completed',
                    details={'station_id': assign.station_id, 'port_id': assign.port_id, 'battery_pct': round(float(vrec.battery_pct), 2) if vrec else None},
                )

                # route back to original destination
                try:
                    sumo_vid = self.vm_to_sumo.get(vid, vid)
                    try:
                        current_ids = set(traci.vehicle.getIDList())
                    except Exception:
                        current_ids = set()
                    if sumo_vid not in current_ids:
                        continue

                    known_edges = set()
                    try:
                        known_edges.update(traci.edge.getIDList())
                    except Exception:
                        pass
                    if assign.original_destination in known_edges:
                        try:
                            traci.vehicle.setSpeed(sumo_vid, -1.0)
                        except Exception:
                            pass
                        traci.vehicle.changeTarget(sumo_vid, assign.original_destination)
                        self.charging_events.append({
                            'event_type': 'resumed_route',
                            'step': step,
                            'vehicle_id': vid,
                            'station_id': assign.station_id,
                            'destination_edge': assign.original_destination,
                        })
                        tracked_entry = self._tracked_lifecycle_entry(vid)
                        if tracked_entry is not None:
                            tracked_entry['resume_event'] = int(step)
                        self._transition_tracked_lifecycle_state(
                            vid,
                            'RESUMING_TRIP',
                            step=step,
                            reason='resumed_route',
                            details={'station_id': assign.station_id, 'destination_edge': assign.original_destination},
                        )
                except Exception:
                    pass

                # log charging complete
                self.charging_events.append({
                    'event_type': 'charging_completed',
                    'step': step,
                    'vehicle_id': vid,
                    'station_id': assign.station_id,
                    'port_id': assign.port_id,
                    'battery_pct': round(float(vrec.battery_pct), 2) if vrec else None,
                    'session_charging_cost': round(assign.session_charging_cost, 4),
                })
                tracked_entry = self._tracked_lifecycle_entry(vid)
                if tracked_entry is not None:
                    tracked_entry['charging_end'] = int(step)
                    if vrec is not None:
                        self._update_tracked_battery_metrics(vid, vrec.battery_pct, step)
                # remove assignment
                del self.assignments[vid]

    def _refresh_dashboard(self) -> None:
        # Build state from cached bulk station snapshots to keep per-step refresh fast.
        vehicles: list[dict[str, Any]] = []
        station_details: list[dict[str, Any]] = []
        station_snapshots = self._get_station_snapshots(force_refresh=False)
        station_lookup = {str(item.get('station_id')): item for item in station_snapshots}
        stations_total = len(station_snapshots)
        stations_available = 0
        stations_occupied = 0
        stations_queue_total = 0

        try:
            active_sumo_ids = set(traci.vehicle.getIDList())
        except Exception:
            active_sumo_ids = set()

        try:
            simulation_time = float(traci.simulation.getTime())
        except Exception:
            simulation_time = float(getattr(self, 'current_step', 0))

        try:
            control_state = self.runtime_control.read()
        except Exception:
            control_state = {"status": "play", "speed": 1.0}

        speed_by_vehicle: dict[str, float] = {}
        edge_speed: dict[str, list[float]] = {}
        for sumo_vid in active_sumo_ids:
            try:
                speed = float(traci.vehicle.getSpeed(sumo_vid))
            except Exception:
                speed = 0.0
            speed_by_vehicle[sumo_vid] = speed
            try:
                edge_id = str(traci.vehicle.getRoadID(sumo_vid))
            except Exception:
                edge_id = ""
            if edge_id:
                edge_speed.setdefault(edge_id, []).append(speed)

        if not self._initial_battery_snapshot:
            for vrec in self.vehicle_manager.list_vehicles():
                self._initial_battery_snapshot[vrec.vehicle_id] = (float(vrec.battery_pct), float(vrec.battery_capacity_kwh))

        energy_consumed_kwh = 0.0
        for vrec in self.vehicle_manager.list_vehicles():
            initial = self._initial_battery_snapshot.get(vrec.vehicle_id)
            if initial is None:
                continue
            initial_pct, capacity_kwh = initial
            consumed = max(0.0, (float(initial_pct) - float(vrec.battery_pct)) / 100.0 * float(capacity_kwh))
            energy_consumed_kwh += consumed

        for snapshot in station_snapshots:
            sid = str(snapshot.get('station_id'))
            available_ports = int(snapshot.get('available_ports', 0) or 0)
            total_ports = int(snapshot.get('total_ports', 0) or 0)
            occupied_ports = int(snapshot.get('occupied_ports', 0) or 0)
            queue_len = int(snapshot.get('queue_length', max(0, total_ports - available_ports)) or 0)
            price_per_kwh = snapshot.get('price_per_kwh')
            grid_load_kw = float(snapshot.get('grid_load_kw', 0.0) or 0.0)
            lat = snapshot.get('lat')
            lon = snapshot.get('lon')
            sx: float | None = None
            sy: float | None = None
            if self.net is not None and lat is not None and lon is not None:
                try:
                    sx, sy = self.net.convertLonLat2XY(float(lon), float(lat))
                except Exception:
                    sx, sy = None, None

            station_details.append({
                'id': sid,
                'name': snapshot.get('name'),
                'x': sx,
                'y': sy,
                'lat': lat,
                'lon': lon,
                'capacity': total_ports,
                'total_ports': total_ports,
                'available_ports': available_ports,
                'queue': queue_len,
                'price': float(price_per_kwh) if price_per_kwh is not None else None,
                'waiting_min': snapshot.get('avg_wait_estimate'),
                'grid_load_kw': grid_load_kw,
                # Backward-compatible keys consumed by existing dashboard/frontends.
                'station_id': sid,
                'price_per_kwh': float(price_per_kwh) if price_per_kwh is not None else None,
                'avg_wait_min': snapshot.get('avg_wait_estimate'),
                'occupied_ports': occupied_ports,
                'charging_ports': int(snapshot.get('charging_ports', 0) or 0),
                'queue_len': queue_len,
                'occupancy_percent': float(snapshot.get('occupancy_percent', 0.0) or 0.0),
                'charging_state': snapshot.get('charging_state'),
            })

            stations_available += available_ports
            stations_occupied += occupied_ports
            stations_queue_total += queue_len

        # include full fleet with canonical per-vehicle records
        for v in self.vehicle_manager.list_vehicles():
            sumo_vid = self.vm_to_sumo.get(v.vehicle_id, v.vehicle_id)
            if sumo_vid in active_sumo_ids:
                try:
                    edge = traci.vehicle.getRoadID(sumo_vid)
                except Exception:
                    edge = None
                try:
                    pos = traci.vehicle.getPosition(sumo_vid)
                except Exception:
                    pos = (None, None)
                try:
                    route_edges = list(traci.vehicle.getRoute(sumo_vid))
                except Exception:
                    route_edges = []
            else:
                edge = None
                pos = (None, None)
                route_edges = []

            assigned = self.assignments.get(v.vehicle_id)
            rec_station = assigned.station_id if assigned else None
            charging_status = assigned.status if assigned else 'driving'

            distance = None
            travel_time = None
            charging_price_per_kwh = None
            estimated_session_cost = None
            session_charging_cost = None
            queue_len = None
            waiting_time = None
            available_ports = None
            occupied_ports = None
            grid_load = None

            if rec_station:
                st = station_lookup.get(rec_station)
                if assigned is not None:
                    travel_time = assigned.recommended_travel_time_min
                    waiting_time = assigned.recommended_waiting_time_min
                    queue_len = assigned.recommended_queue_length
                    charging_price_per_kwh = assigned.charging_price_per_kwh
                    estimated_session_cost = assigned.estimated_session_cost
                    session_charging_cost = assigned.session_charging_cost
                if st:
                    try:
                        lat = st.get('lat')
                        lon = st.get('lon')
                        if lat is not None and lon is not None and pos and pos[0] is not None and pos[1] is not None and self.net is not None:
                            sx, sy = self.net.convertLonLat2XY(float(lon), float(lat))
                            distance = round(math.hypot(float(pos[0]) - float(sx), float(pos[1]) - float(sy)) / 1000.0, 3)
                    except Exception:
                        distance = None
                if charging_price_per_kwh is None and st and st.get('price_per_kwh') is not None:
                    charging_price_per_kwh = float(st.get('price_per_kwh'))
                if queue_len is None:
                    queue_len = max(
                        0,
                        int((st or {}).get('total_ports', 0) or 0) - int((st or {}).get('available_ports', 0) or 0),
                    )
                if waiting_time is None:
                    waiting_time = (st or {}).get('avg_wait_estimate')
                available_ports = (st or {}).get('available_ports')
                occupied_ports = (st or {}).get('occupied_ports')
                grid_load = float((st or {}).get('grid_load_kw', 0.0) or 0.0)

            if estimated_session_cost is None and charging_price_per_kwh is not None:
                estimated_session_cost = max(0.0, energy_needed_kwh(v.battery_pct, 80.0, v.battery_capacity_kwh)) * charging_price_per_kwh
            if session_charging_cost is None:
                session_charging_cost = 0.0

            ppo_reward = None
            for r in reversed(self.recommendation_log):
                if r.get('vehicle_id') == v.vehicle_id:
                    ppo_reward = r.get('ppo_reward')
                    break

            speed_value = speed_by_vehicle.get(sumo_vid)
            vehicle_payload = {
                'id': v.vehicle_id,
                'battery': round(float(v.battery_pct), 2),
                'speed': speed_value,
                'destination': v.destination,
                'x': pos[0] if pos else None,
                'y': pos[1] if pos else None,
                'tracked': bool(v.tracked),
                'recommendation': {
                    'station_id': rec_station,
                    'station_name': (station_lookup.get(rec_station) or {}).get('name') if rec_station else None,
                    'queue': queue_len,
                    'price': charging_price_per_kwh,
                    'waiting_min': waiting_time,
                    'available_ports': available_ports,
                    'grid_load_kw': grid_load,
                    'reward': ppo_reward,
                },
                'route': route_edges,
                'status': charging_status,
                # Backward-compatible keys consumed by existing dashboard/frontends.
                'vehicle_id': v.vehicle_id,
                'source': v.source,
                'battery_capacity_kwh': v.battery_capacity_kwh,
                'battery_pct': float(v.battery_pct),
                'remaining_range_km': float(v.remaining_range_km),
                'charging_requirement_kwh': float(v.charging_requirement_kwh),
                'current_edge': edge,
                'current_position': {'x': pos[0], 'y': pos[1]} if pos else None,
                'recommended_station': rec_station,
                'distance': distance,
                'travel_time': travel_time,
                'charging_price_per_kwh': charging_price_per_kwh,
                'estimated_session_cost': estimated_session_cost,
                'session_charging_cost': session_charging_cost,
                'charging_cost': session_charging_cost,
                'queue_length': queue_len,
                'waiting_time': waiting_time,
                'available_ports': available_ports,
                'occupied_ports': occupied_ports,
                'grid_load': grid_load,
                'charging_status': charging_status,
                'ppo_reward': ppo_reward,
                'simulation_step': getattr(self, 'current_step', None),
            }
            vehicles.append(vehicle_payload)

        active_vehicle_count = len(active_sumo_ids)
        avg_speed = statistics.mean(speed_by_vehicle.values()) if speed_by_vehicle else 0.0
        slow_vehicle_count = sum(1 for speed in speed_by_vehicle.values() if speed <= 2.0)
        congestion_percent = (slow_vehicle_count / max(1, active_vehicle_count)) * 100.0 if active_vehicle_count else 0.0
        traffic_density = (active_vehicle_count / max(1, len(edge_speed))) if edge_speed else 0.0

        traffic: list[dict[str, Any]] = []
        max_edge_count = max((len(speeds) for speeds in edge_speed.values()), default=1)
        for edge_id, speeds in edge_speed.items():
            avg_edge_speed = statistics.mean(speeds) if speeds else 0.0
            congestion = (len(speeds) / max(1, max_edge_count))
            traffic.append({
                'edge_id': edge_id,
                'density': len(speeds),
                'congestion': round(float(congestion), 4),
                'speed': round(float(avg_edge_speed), 4),
            })
        traffic.sort(key=lambda item: item['congestion'], reverse=True)

        waits = [float(item.get('avg_wait_estimate', 0.0) or 0.0) for item in station_snapshots if item.get('avg_wait_estimate') is not None]
        avg_wait_min = statistics.mean(waits) if waits else 0.0

        tracked_ids = [v.vehicle_id for v in self.vehicle_manager.list_tracked_vehicles()]
        latest_recommendation = self.recommendation_log[-1] if self.recommendation_log else None
        ppo_summary = {
            'ev_id': latest_recommendation.get('vehicle_id') if latest_recommendation else None,
            'action': latest_recommendation.get('ppo_action') if latest_recommendation else None,
            'station_id': latest_recommendation.get('selected_station') if latest_recommendation else None,
            'reward': latest_recommendation.get('ppo_reward') if latest_recommendation else None,
            'battery': latest_recommendation.get('battery_pct') if latest_recommendation else None,
            'observation': latest_recommendation.get('ppo_observation') if latest_recommendation else None,
            'recommendation_timestamp': latest_recommendation.get('recommendation_timestamp_utc') if latest_recommendation else None,
            'reason': latest_recommendation.get('recommendation_reason') if latest_recommendation else None,
        }

        summary = {
            'charging_vehicles': sum(1 for vehicle in vehicles if str(vehicle.get('charging_status') or '').lower() == 'charging'),
            'avg_wait_min': round(avg_wait_min, 2),
            'active_assignments': len(self.assignments),
            'total_vehicles': len(vehicles),
        }

        control_status = str(control_state.get('status') or 'play').lower()
        simulation_status = 'PAUSED' if control_status == 'paused' else 'RUNNING'
        tracked_vehicle = None
        tracked_sumo_id = None
        tracked_vehicle_payload = None
        if tracked_ids:
            tracked_vehicle = self.vehicle_manager.get_vehicle(tracked_ids[0])
            if tracked_vehicle is not None:
                tracked_sumo_id = self.vm_to_sumo.get(tracked_vehicle.vehicle_id, tracked_vehicle.vehicle_id)
                if tracked_sumo_id in active_sumo_ids:
                    try:
                        tracked_vehicle_payload = {
                            'x': traci.vehicle.getPosition(tracked_sumo_id)[0],
                            'y': traci.vehicle.getPosition(tracked_sumo_id)[1],
                            'road_id': traci.vehicle.getRoadID(tracked_sumo_id),
                            'speed': speed_by_vehicle.get(tracked_sumo_id),
                        }
                    except Exception:
                        tracked_vehicle_payload = None
        latest_assignment = None
        if tracked_vehicle is not None:
            latest_assignment = self.assignments.get(tracked_vehicle.vehicle_id)
        latest_recommendation_entry = self.recommendation_log[-1] if self.recommendation_log else None
        canonical_simulation = {
            'step': int(getattr(self, 'current_step', 0)),
            'time': float(simulation_time),
            'status': control_status,
            'speed': float(control_state.get('speed', 1.0) or 1.0),
            'vehicle_count': len(vehicles),
            'ev_count': len(self.vehicle_manager.list_vehicles()),
            'connection': 'LIVE',
            'running': control_status != 'paused',
            'paused': control_status == 'paused',
            'simulation_status': simulation_status,
        }
        tracked_lifecycle_entry = self.tracked_lifecycle.get(tracked_vehicle.vehicle_id) if tracked_vehicle is not None else None
        lifecycle_state = None
        lifecycle_history = []
        if tracked_lifecycle_entry is not None:
            lifecycle_state = tracked_lifecycle_entry.get('state')
            lifecycle_history = tracked_lifecycle_entry.get('state_history') or []
        if lifecycle_state is None:
            lifecycle_state = latest_assignment.status if latest_assignment is not None else ('charging' if latest_recommendation_entry is not None else 'TRAVELING')
        canonical_tracked_ev = {
            'id': tracked_vehicle.vehicle_id if tracked_vehicle is not None else (tracked_ids[0] if tracked_ids else None),
            'x': tracked_vehicle_payload.get('x') if tracked_vehicle_payload else None,
            'y': tracked_vehicle_payload.get('y') if tracked_vehicle_payload else None,
            'road_id': tracked_vehicle_payload.get('road_id') if tracked_vehicle_payload else None,
            'speed': tracked_vehicle_payload.get('speed') if tracked_vehicle_payload else None,
            'battery': round(float(tracked_vehicle.battery_pct), 2) if tracked_vehicle is not None else None,
            'battery_percent': round(float(tracked_vehicle.battery_pct), 2) if tracked_vehicle is not None else None,
            'state': lifecycle_state,
            'lifecycle_state': lifecycle_state,
            'state_history': lifecycle_history,
            'selected_station': getattr(latest_assignment, 'station_id', None) if latest_assignment is not None else None,
            'destination': getattr(tracked_vehicle, 'destination', None) if tracked_vehicle is not None else None,
            'charging': latest_assignment is not None and getattr(latest_assignment, 'status', None) in {'charging', 'waiting', 'enroute'},
        }
        canonical_ppo_decision = {
            'ev': latest_recommendation_entry.get('vehicle_id') if latest_recommendation_entry else None,
            'station': latest_recommendation_entry.get('selected_station') if latest_recommendation_entry else None,
            'action': latest_recommendation_entry.get('ppo_action') if latest_recommendation_entry is not None else None,
            'reward': latest_recommendation_entry.get('ppo_reward') if latest_recommendation_entry else None,
            'battery': latest_recommendation_entry.get('battery_pct') if latest_recommendation_entry else None,
            'reason': latest_recommendation_entry.get('recommendation_reason') if latest_recommendation_entry else None,
            'timestamp': latest_recommendation_entry.get('step') if latest_recommendation_entry else None,
            'observation': latest_recommendation_entry.get('ppo_observation') if latest_recommendation_entry else None,
            'recommendation_timestamp': latest_recommendation_entry.get('recommendation_timestamp_utc') if latest_recommendation_entry else None,
        }
        canonical_station = {
            'id': None,
            'name': None,
            'x': None,
            'y': None,
            'ports_total': None,
            'ports_available': None,
            'queue': None,
            'price': None,
            'wait': None,
            'load': None,
            'utilization': None,
        }
        if station_details:
            first_station = station_details[0]
            canonical_station = {
                'id': first_station.get('id') or first_station.get('station_id'),
                'name': first_station.get('name'),
                'x': first_station.get('x'),
                'y': first_station.get('y'),
                'ports_total': first_station.get('total_ports'),
                'ports_available': first_station.get('available_ports'),
                'queue': first_station.get('queue'),
                'price': first_station.get('price'),
                'wait': first_station.get('waiting_min'),
                'load': first_station.get('grid_load_kw'),
                'utilization': first_station.get('occupancy_percent') if first_station.get('occupancy_percent') is not None else first_station.get('utilization'),
            }
        canonical_vehicles = [
            {
                'id': vehicle.get('id') or vehicle.get('vehicle_id'),
                'x': vehicle.get('x'),
                'y': vehicle.get('y'),
                'speed': vehicle.get('speed'),
                'battery': vehicle.get('battery'),
                'state': vehicle.get('status') or vehicle.get('charging_status') or vehicle.get('state'),
            }
            for vehicle in vehicles
        ]
        canonical_stations = [
            {
                'id': station.get('id') or station.get('station_id'),
                'x': station.get('x'),
                'y': station.get('y'),
                'ports_total': station.get('total_ports'),
                'ports_available': station.get('available_ports'),
                'queue': station.get('queue'),
                'price': station.get('price'),
                'wait': station.get('waiting_min'),
                'load': station.get('grid_load_kw'),
                'utilization': station.get('occupancy_percent') if station.get('occupancy_percent') is not None else station.get('utilization'),
            }
            for station in station_details
        ]
        canonical_history = {
            'rewards': [entry.get('ppo_reward') for entry in self.recommendation_log if entry.get('ppo_reward') is not None],
            'recommendations': [
                {'vehicle_id': entry.get('vehicle_id'), 'selected_station': entry.get('selected_station')}
                for entry in self.recommendation_log
                if entry.get('vehicle_id') is not None or entry.get('selected_station') is not None
            ],
            'timestamps': [entry.get('step') for entry in self.recommendation_log if entry.get('step') is not None],
        }
        dashboard_state = {
            'simulation': canonical_simulation,
            'telemetry': {
                'timestamp': int(time.time() * 1000),
                'step': int(getattr(self, 'current_step', 0)),
            },
            'tracked_ev': canonical_tracked_ev,
            'ppo_decision': canonical_ppo_decision,
            'station': canonical_station,
            'history': canonical_history,
            'kpis': {
                'avg_speed': round(float(avg_speed), 4),
                'traffic_density': round(float(traffic_density), 4),
                'avg_wait_min': round(float(avg_wait_min), 4),
                'energy_consumed_kwh': round(float(energy_consumed_kwh), 4),
            },
            'stations_summary': {
                'total': stations_total,
                'available': stations_available,
                'occupied': stations_occupied,
                'queue_total': stations_queue_total,
            },
            'vehicles': vehicles,
            'station_details': station_details,
            'traffic': traffic,
            'ppo': ppo_summary,
            'tracked_ids': tracked_ids,
            'tracked_lifecycle': [self.tracked_lifecycle[vehicle_id] for vehicle_id in tracked_ids if vehicle_id in self.tracked_lifecycle],
            'summary': summary,
            'metrics': self.simulation_metrics[-20:],
            'recommendations': self.recommendation_log[-20:],
            'charging_events': self.charging_events[-50:],
            'latest_recommendation': latest_recommendation,
            'network': {},
            'xai': {'latest_explanation': getattr(self, 'last_explanation', None) or {}, 'explanation_count': len(self.recommendation_log)},
        }
        self._last_dashboard_state = dashboard_state
        try:
            canonical_payload = self.dashboard._build_canonical_state(dashboard_state)
            self.dashboard.update(canonical_payload)
        except Exception:
            self.dashboard.update(dashboard_state)

    def _collect_metrics(self, step: int) -> None:
        try:
            total_vehicles = len(traci.vehicle.getIDList())
        except Exception:
            total_vehicles = len(self.vehicle_manager.list_vehicles())

        charging_count = 0
        waiting_count = 0
        for a in self.assignments.values():
            if a.status == 'charging':
                charging_count += 1
            if a.status == 'waiting':
                waiting_count += 1

        waits = []
        for station in self._get_station_snapshots(force_refresh=False):
            if station.get('avg_wait_estimate') is not None:
                waits.append(float(station.get('avg_wait_estimate', 0.0) or 0.0))

        avg_wait = statistics.mean(waits) if waits else 0.0

        metric = {
            'step': step,
            'total_vehicles': total_vehicles,
            'assigned': len(self.assignments),
            'charging': charging_count,
            'waiting': waiting_count,
            'avg_wait_min': avg_wait,
        }
        self.simulation_metrics.append(metric)

    def _emit_runtime_metrics(self, step: int, elapsed_seconds: float) -> None:
        try:
            total_vehicles = len(self.vehicle_manager.list_vehicles())
            active = total_vehicles
            charging = sum(1 for a in self.assignments.values() if a.status == 'charging')
            waiting = sum(1 for a in self.assignments.values() if a.status == 'waiting')
            low_battery = sum(1 for vehicle in self.vehicle_manager.list_vehicles() if getattr(vehicle, 'battery_pct', 0.0) <= self.charge_threshold_pct)
        except Exception:
            active = len(self.vehicle_manager.list_vehicles())
            charging = 0
            waiting = 0
            low_battery = 0

        waits = []
        available_ports_total = 0
        occupied_ports_total = 0
        queue_length_total = 0
        station_count = 0
        station_manager = getattr(self, 'station_manager', None)
        if station_manager is not None:
            for station in self._get_station_snapshots(force_refresh=False):
                if station.get('avg_wait_estimate') is not None:
                    waits.append(float(station.get('avg_wait_estimate', 0.0) or 0.0))
                available = int(station.get('available_ports', 0) or 0)
                occupied = int(station.get('occupied_ports', 0) or 0)
                available_ports_total += available
                occupied_ports_total += occupied
                queue_length_total += int(station.get('queue_length', max(0, occupied - int(station.get('charging_ports', 0) or 0))) or 0)
                station_count += 1

        avg_wait = statistics.mean(waits) if waits else 0.0
        utilization = 0.0
        if available_ports_total + occupied_ports_total:
            utilization = occupied_ports_total / float(available_ports_total + occupied_ports_total)

        monitoring_started_at = getattr(self, '_monitoring_started_at', time.time())
        update_metric('simulation_steps_total', step + 1)
        update_metric('simulation_running', 1)
        update_metric('simulation_duration_seconds', max(0.0, time.time() - monitoring_started_at))
        update_metric('simulation_speed', 1.0 / max(elapsed_seconds, 1e-6))
        update_metric('simulation_fps', 1.0 / max(elapsed_seconds, 1e-6))
        update_metric('simulation_step', step)
        update_metric('active_vehicles', active)
        update_metric('charging_vehicles', charging)
        update_metric('waiting_vehicles', waiting)
        update_metric('low_battery_vehicles', low_battery)
        update_metric('vehicle_count', active)
        update_metric('available_ports', available_ports_total)
        update_metric('occupied_ports', occupied_ports_total)
        update_metric('average_wait_time', avg_wait)
        update_metric('average_queue_length', max(0.0, queue_length_total / max(1, station_count)))
        update_metric('charging_queue_length', queue_length_total)
        update_metric('station_utilization_percent', utilization * 100.0)
        update_metric('station_utilization', utilization)
        vehicle_list = list(self.vehicle_manager.list_vehicles())
        vehicle_count = len(vehicle_list)
        update_metric('average_soc', sum(float(v.battery_pct) for v in vehicle_list) / max(1, vehicle_count))
        update_metric('distance_travelled', sum(float(v.remaining_range_km or 0.0) for v in vehicle_list) * 0.0)
        update_metric('average_speed', 0.0)
        station_count_for_density = station_count if station_count > 0 else 1
        update_metric('traffic_density', max(0.0, float(active) / max(1, station_count_for_density)))
        update_metric('finished_trips', len(self.charging_events))
        last_reward = getattr(self, '_monitoring_last_reward', 0.0)
        episode_number = getattr(self, '_monitoring_episode', 0)
        update_metric('episode_reward', last_reward)
        update_metric('episode_number', episode_number)
        update_metric('ppo_reward', last_reward)
        update_metric('ppo_reward_avg', last_reward)
        update_metric('policy_inference_latency', 0.0)
        update_metric('decision_time', 0.0)

        collectors = getattr(self, '_monitoring_collectors', [])
        for collector in collectors:
            try:
                collector.collect({
                    'running': True,
                    'steps': step + 1,
                    'duration_seconds': max(0.0, time.time() - getattr(self, '_monitoring_started_at', time.time())),
                    'speed': 1.0 / max(elapsed_seconds, 1e-6),
                    'active': active,
                    'charging': charging,
                    'waiting': waiting,
                    'low_battery': low_battery,
                    'successful': len(self.recommendation_log),
                    'failed': 0,
                    'utilization_percent': utilization * 100.0,
                    'available_ports': available_ports_total,
                    'occupied_ports': occupied_ports_total,
                    'avg_wait_time': avg_wait,
                    'avg_queue_length': max(0.0, queue_length_total / max(1, station_count)),
                    'ppo_reward': last_reward,
                    'ppo_reward_avg': last_reward,
                    'episode_reward': last_reward,
                    'episode_length': step + 1,
                    'policy_inference_latency': 0.0,
                    'decision_time': 0.0,
                })
            except Exception:
                pass

    def _save_results(self) -> None:
        out_dir = Path('outputs')
        out_dir.mkdir(parents=True, exist_ok=True)

        # recommendation_log.csv
        rec_file = out_dir / 'recommendation_log.csv'
        with rec_file.open('w', newline='', encoding='utf-8') as f:
            if self.recommendation_log:
                writer = csv.DictWriter(f, fieldnames=list(self.recommendation_log[0].keys()))
                writer.writeheader()
                for r in self.recommendation_log:
                    writer.writerow(r)

        # charging_events.csv
        ch_file = out_dir / 'charging_events.csv'
        with ch_file.open('w', newline='', encoding='utf-8') as f:
            if self.charging_events:
                writer = csv.DictWriter(f, fieldnames=list({k for e in self.charging_events for k in e.keys()}))
                writer.writeheader()
                for e in self.charging_events:
                    writer.writerow(e)

        # simulation_metrics.csv
        sim_file = out_dir / 'simulation_metrics.csv'
        with sim_file.open('w', newline='', encoding='utf-8') as f:
            if self.simulation_metrics:
                writer = csv.DictWriter(f, fieldnames=list(self.simulation_metrics[0].keys()))
                writer.writeheader()
                for m in self.simulation_metrics:
                    writer.writerow(m)

        # evaluation summary
        summary = {
            'total_recommendations': len(self.recommendation_log),
            'total_charging_events': len(self.charging_events),
            'avg_reward': statistics.mean(self.reward_curve) if self.reward_curve else None,
            'avg_wait_min': statistics.mean([m.get('avg_wait_min', 0.0) for m in self.simulation_metrics]) if self.simulation_metrics else None,
        }
        summary_file = out_dir / 'evaluation_summary.csv'
        with summary_file.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            writer.writeheader()
            writer.writerow(summary)

        # plots
        if plt:
            # reward_curve.png
            if self.reward_curve:
                plt.figure()
                plt.plot(self.reward_curve)
                plt.title('Reward Curve')
                plt.xlabel('Recommendation #')
                plt.ylabel('Reward')
                plt.savefig(out_dir / 'reward_curve.png')

            # waiting_time_comparison.png
            if self.simulation_metrics:
                plt.figure()
                steps = [m['step'] for m in self.simulation_metrics]
                waits = [m.get('avg_wait_min', 0.0) for m in self.simulation_metrics]
                plt.plot(steps, waits)
                plt.title('Average Waiting Time')
                plt.xlabel('Step')
                plt.ylabel('Avg wait (min)')
                plt.savefig(out_dir / 'waiting_time_comparison.png')

            # station_utilization.png (charging count over time)
            plt.figure()
            charging = [m.get('charging', 0) for m in self.simulation_metrics]
            plt.plot([m['step'] for m in self.simulation_metrics], charging)
            plt.title('Station Utilization (charging count)')
            plt.xlabel('Step')
            plt.ylabel('Charging vehicles')
            plt.savefig(out_dir / 'station_utilization.png')

            # grid_load_distribution.png (approx from station metrics at last step)
            grid_loads = []
            for station in self._get_station_snapshots(force_refresh=True):
                grid_loads.append(float(station.get('grid_load_kw', 0.0) or 0.0))
            if grid_loads:
                plt.figure()
                plt.hist(grid_loads, bins=20)
                plt.title('Grid Load Distribution')
                plt.xlabel('kW')
                plt.ylabel('Stations')
                plt.savefig(out_dir / 'grid_load_distribution.png')

