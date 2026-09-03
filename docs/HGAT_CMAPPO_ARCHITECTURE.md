# Architectural Audit: HGAT-CMAPPO Integration

## 1. Current System Architecture

The existing Smart EV Charging Station Recommendation System is built on a modular Python/SUMO architecture simulating electric vehicle (EV) charging lifecycles across Bengaluru, India:

- **Road Network & Traffic Simulation**: SUMO (`sumo/bengaluru.sumocfg`, `sumo/bengaluru.net.xml`, `sumo/bengaluru.rou.xml`) controlled via TraCI (`src/simulation/controller.py`).
- **EV Model & Lifecycle**: [`src/ev_model/ev.py`](file:///d:/Smart_EV_Station_Recommandadtion_System/src/ev_model/ev.py) tracks state of charge (SOC %), battery capacity (kWh), energy consumption rate (kWh/km), current edge, speed, and destination.
- **EV Fleet Management**: [`src/ev_management/vehicle_manager.py`](file:///d:/Smart_EV_Station_Recommandadtion_System/src/ev_management/vehicle_manager.py) manages multi-EV fleets ($10$ to $1000$ EVs).
- **Charging Station Database**: [`data/stations/stations.sqlite`](file:///d:/Smart_EV_Station_Recommandadtion_System/data/stations/stations.sqlite) containing $500+$ real Bengaluru station locations, port counts, pricing, queue lengths, power ratings, and grid loads. Managed via [`src/station_management/station_manager.py`](file:///d:/Smart_EV_Station_Recommandadtion_System/src/station_management/station_manager.py).
- **Recommendation Engine**: [`src/recommendation_engine/engine.py`](file:///d:/Smart_EV_Station_Recommandadtion_System/src/recommendation_engine/engine.py) provides live station selection using deterministic multi-objective weights or PPO inference.

---

## 2. Current PPO Flow

The current PPO implementation uses a single-agent or multi-EV wrapped policy:
- **Baseline Checkpoint**: Saved at [`runs/ppo_ckpt/ppo_ev_final.zip`](file:///d:/Smart_EV_Station_Recommandadtion_System/runs/ppo_ckpt/ppo_ev_final.zip).
- **Observation Space**: Flat $53$-dimensional vector per EV:
  - $5$ EV features: `[battery_pct, battery_kwh, distance_to_destination, speed, current_time_step]`
  - $K \times 6 = 48$ Candidate Station features for $K=8$ candidate stations: `[distance_km, travel_time_min, free_ports, price_per_kwh, avg_wait_min, queue_len]`
- **Action Space**: Discrete action $a \in \{0, \dots, K-1\}$ selecting one candidate station from top-$K$ candidate list.
- **Reward Function**: Multi-objective linear combination balancing arrival success, waiting time, charging cost, detour distance, and port availability.

---

## 3. Existing Limitations

1. **Flat Vector Representation**: The flat $53$-D observation ignores topological road network relations, spatial station proximity graphs, and inter-EV contention edges.
2. **Independent Agent Decisions**: Individual EV agents act without observing joint EV decisions, leading to potential station herd behavior and port contention at popular stations.
3. **Unconstrained Optimization**: Dual safety conditions (SOC depletion, station queue overflows, port exhaustion, detour limits, grid overload) are penalized via soft penalties rather than guaranteed via formal constraint optimization.

---

## 4. HGAT-CMAPPO Research Architecture

```
                SUMO / TraCI Environment
                           |
                           v
              Heterogeneous Graph Builder
            (EV, Station, Road/Traffic, Grid)
                           |
                           v
           Heterogeneous Graph Attention (HGAT)
                      Encoder
                           |
                           v
        Multi-Agent Embeddings & Feature Fusion
                           |
            +--------------+--------------+
            |                             |
            v                             v
     MAPPO Decentralized          Centralized Critic
        Actor Policy                 (Global State)
            |                             |
            v                             |
    Categorical Action                    |
      Probability                         |
            |                             |
            v                             |
    Constraint & Action                   |
        Masking                           |
            |                             |
            v                             v
    Selected Station             Primal-Dual Lagrangian
     Recommendation                 Constraint Update
            |                             |
            +--------------+--------------+
                           |
                           v
            SUMO Vehicle Rerouting & Lifecycle
```

### Integration Points:
- **Graph Builder** (`src/rl/hgat/graph_builder.py`): Constructs heterogeneous graph objects containing EV nodes, Station nodes, Road/Traffic nodes, and Grid nodes from live simulation state.
- **HGAT Encoder** (`src/rl/hgat/hgat_encoder.py`): Multi-head heterogeneous graph attention network mapping variable graph structures to fixed-size node embeddings.
- **MAPPO Policy & Centralized Critic** (`src/rl/mappo/actor_critic.py`): Parameter-shared actor policy with discrete categorical action heads and global centralized critic.
- **Lagrangian Constraint Layer** (`src/rl/constraints/lagrangian.py`): Tracks 5 explicit constraint signals and updates learnable dual multipliers $\lambda_1, \dots, \lambda_5$.
- **Action Masking** (`src/rl/policies/action_masking.py`): Hard masks unreachable stations, full ports, and severe queue violations before categorical sampling.
