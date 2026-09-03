import { DigitalTwinScene } from './scene.js';
import { CameraController } from './camera-controller.js';

export class ThreeTwinRenderer {
  constructor(container, options = {}) {
    this.container = container;
    this.scene = new DigitalTwinScene(container, options);
    this.cameraController = new CameraController(this.scene.camera, this.scene.renderer.domElement, this.scene.controls);
    this.state = {
      network: { edges: [], bounds: {} },
      vehicles: [],
      stations: [],
      trafficLights: [],
      trackedIds: [],
      followVehicleId: null,
    };
    this._bindControls();
  }

  _bindControls() {
    const followButton = document.getElementById('follow-vehicle');
    if (followButton) {
      followButton.addEventListener('click', () => {
        const nextId = this.state.vehicles[0]?.id || this.state.vehicles[0]?.vehicle_id || null;
        this.state.followVehicleId = nextId;
        this.scene.setFollowVehicle(nextId);
        const targetVehicle = this.state.vehicles.find((vehicle) => (vehicle.id || vehicle.vehicle_id) === nextId);
        this.cameraController.setFollowVehicle(targetVehicle || null);
      });
    }
    const resetButton = document.getElementById('reset-view');
    if (resetButton) {
      resetButton.addEventListener('click', () => {
        this.state.followVehicleId = null;
        this.scene.setFollowVehicle(null);
        this.scene.frameNetwork(this.scene.state.network.bounds || {});
      });
    }
  }

  render(payload) {
    const data = payload.data || payload;
    if (!data || typeof data !== 'object') {
      console.error('[3D WS] malformed snapshot', payload);
      return;
    }
    if (!Array.isArray(data.vehicles) || !Array.isArray(data.stations)) {
      console.error('[3D WS] snapshot missing vehicles or stations', data);
      return;
    }
    const network = data.network && Array.isArray(data.network.edges) && data.network.edges.length
      ? data.network
      : (payload.network && Array.isArray(payload.network.edges) ? payload.network : this.state.network);
    this.state.network = network;
    this.state.vehicles = data.vehicles || [];
    this.state.stations = data.station_details || data.stations || [];
    this.state.trafficLights = data.traffic_lights || [];
    this.state.trackedIds = data.tracked_ids || [];
    const latestRecommendation = data.latest_recommendation || (Array.isArray(data.recommendations) ? data.recommendations[data.recommendations.length - 1] : null);
    this.state.selectedStationId = (
      data.ppo?.station_id ||
      data.xai?.latest_explanation?.policy_probability_visualization?.selected_station_id ||
      latestRecommendation?.selected_station ||
      null
    );
    console.info('[3D WS] snapshot', payload.simulation_time ?? payload.step ?? '-', 'vehicles=', this.state.vehicles.length, 'stations=', this.state.stations.length);
    console.info('[3D VEHICLES] ids=', this.state.vehicles.map((vehicle) => vehicle.id || vehicle.vehicle_id));
    console.info('[3D STATIONS] count=', this.state.stations.length, 'recommended=', this.state.selectedStationId || '-');
    this.scene.setRoadNetwork(this.state.network);
    this.scene.setVehicles(this.state.vehicles, this.state.trackedIds);
    this.scene.setStations(this.state.stations);
    this.scene.setTrafficLights(this.state.trafficLights);
    this.scene.setTrafficDensity(this.state.vehicles);
    this.scene.setRoutes((this.state.vehicles || [])
      .filter((vehicle) => Array.isArray(vehicle.destination_path) && vehicle.destination_path.length > 1)
      .map((vehicle) => ({
        points: vehicle.destination_path,
        color: vehicle.charging_status === 'waiting' ? 0xf59e0b : (vehicle.charging_status === 'charging' ? 0x2dd4bf : 0x38bdf8),
      })));
    const trackedLead = this.state.vehicles.find((vehicle) => this.state.trackedIds.includes(vehicle.id || vehicle.vehicle_id));
    this.scene.setSelectedVehicle((trackedLead && (trackedLead.id || trackedLead.vehicle_id)) || this.state.vehicles?.[0]?.id || this.state.vehicles?.[0]?.vehicle_id || null);
    this.scene.highlightStation(this.state.selectedStationId);
    this.cameraController.update();
  }

  highlightStation(stationId) {
    this.scene.highlightStation(stationId || null);
  }
}
