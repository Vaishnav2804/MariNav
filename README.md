<img src="https://github.com/user-attachments/assets/9e15dfe7-102f-4990-8488-deb7ae480a14" alt="A log" width="150" height="200"> &nbsp; &nbsp; &nbsp;  <img width="150" height="200" alt="Media (1)" src="https://github.com/user-attachments/assets/93748423-f317-44e1-98ab-4b6b1a45be77" /> &nbsp; &nbsp; &nbsp; <img width="150" height="200" alt="Media (1)" src="https://github.com/user-attachments/assets/9c57c4f6-83c4-4d9c-9a3e-db9588ea1065"/>

# MariNav - Maritime Route Optimization using Reinforcement Learning!


`MariNav` is a reinforcement learning (RL) environment built to simulate and optimize tanker navigation across oceanic routes represented by H3 hexagonal grids. It integrates real-world wind data, historical route frequencies, and fuel consumption models, allowing agents to learn efficient and realistic maritime paths under environmental constraints.

**This environment is compatible with `PPO` and `MaskablePPO` from Stable-Baselines3 and `sb3-contrib`, making it suitable for training with or without action masking support.** It also supports intrinsic exploration techniques such as `RND` (Random Network Distillation) from the `rllte.xplore.reward` module.

---

Below is an image that computes shortest path from a start and goal H3 cell in a map using Dijistra's algorithm. We can use `MariNav` to develop a better optimal path using RL, given the weather, fuel and time constraints. 

  <img width="1316" height="430" alt="image" src="https://github.com/user-attachments/assets/277ccf0d-f9ad-4038-a790-3fe58ac41810" />


  ### Comparison:

<img width="1303" height="418" alt="image" src="https://github.com/user-attachments/assets/4054eb15-dc41-41ac-80e0-0d022d4ea222" />

---

## Installation

```bash
git clone https://github.com/Vaishnav2804/MariNav
cd MariNav
pip install -r requirements.txt
```

---

## Quick Start

```python
from Env.MariNav import MariNav
from sb3_contrib import MaskablePPO
import networkx as nx

# Load input data
wind_map = load_full_wind_map("august_2018_60min_windmap_v2.csv")
graph = nx.read_gexf("GULF_VISITS_CARGO_TANKER_AUGUST_2018.gexf").to_undirected()

# Define H3 region pool
h3_pool = [
    "862b160d7ffffff", "860e4da17ffffff",
    "861b9101fffffff", "862b256dfffffff",
    "862b33237ffffff"
]

# Create environment
env = MariNav(
    h3_pool=h3_pool,
    graph=graph,
    wind_map=wind_map,
    h3_resolution=6,
    wind_threshold=22,
    render_mode="human"
)

model = MaskablePPO("MlpPolicy", env, verbose=1) # Or use PPO
model.learn(total_timesteps=100_000)
```

### Using Random Network Distillation

```python
from rllte.xplore.reward import RND
from rllte.xplore.wrapper import RLeXploreWithOnPolicyRL
from stable_baselines3 import PPO

irs = RND(env, device="cpu")

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=240_000_000, callback=RLeXploreWithOnPolicyRL(irs))
```

## Example Experiment Results

<img width="1735" height="647" alt="image" src="https://github.com/user-attachments/assets/f59f597a-a823-47ef-b6ca-664af4703970" />

PPO with action masks comparatively performes better than the standard PPO with MLP policy - (For the given environment) -- Play around to figure out more!

---

## What It Does

MariNav models real-world vessel navigation by:

* Using **H3 grids** to represent navigable ocean regions
* Integrating **timestamped wind data**
* Leveraging **historical route usage** from AIS-derived graphs
* Providing **multi-objective rewards** for training robust RL agents
* We can use MariNav to compute best path for a given pair of H3 cells in a continuous observation space. 


---

## Features

### Maritime Environment

* **Hex-Based Ocean Model**: Built using Uber H3 (resolution 6)
* **Graph Navigation**: Based on real ship visits stored in a weighted `NetworkX` graph
* **Wind-Aware Dynamics**: Wind speed and direction influence vessel speed and fuel
* **Dynamic Action Space**: Multi-discrete choices over neighbor cells + discrete speeds

### Multi-Objective Reward Function

The reward function combines:

* **Progress Reward** – distance reduction toward the goal
* **Fuel Penalty** – penalizes fuel-heavy maneuvers
* **Wind Penalty** – penalizes adverse wind alignment
* **Alignment Penalty** – penalizes angular misalignment with wind
* **ETA Penalty** – encourages timely arrival
* **Frequency Reward** – rewards travel along historically common routes

### Logging and Visualization

* TensorBoard metrics for each reward component
* Route frequency analysis and pair visitation tracking
* Matplotlib + GeoPandas-based rendering (headless safe)
* CSV logging of each episode and step-level transition

---



## Training with VecEnvs and Callbacks

The main training script supports:

* Parallel training using `SubprocVecEnv`
* Reward normalization with `VecNormalize`
* Early stopping and evaluation checkpoints
* Logging with TensorBoard, CSVs, and SB3-compatible callbacks

### Launch Training

```bash
python train_marinav.py
```

### Environment Factory (example)

```python
def make_env():
    def _init():
        base_env = MariNav(
            h3_pool=H3_POOL,
            graph=G_visits,
            wind_map=full_wind_map,
            h3_resolution=6,
            wind_threshold=22,
            render_mode="human"
        )
        base_env.visited_path_counts = global_visited_path_counts
        return Monitor(base_env)
    return _init
```

---

##  Environment Specs

### Observation Space

8-dimensional vector:

* `[lat, lon]` of current position
* `speed_over_ground`
* `wind_direction`
* `[lat, lon]` of start & goal

### Action Space

Multi-discrete:

* Neighbor index (variable per cell)
* Speed index (5 levels from 8–22 knots)

### Termination Conditions

* Goal reached (success)
* Max episode length exceeded (truncated)
* Invalid move (failure)

---

## Advanced Options

### Custom Reward Tuning

Modify constants in `MariNav.py`:

```python
PROGRESS_REWARD_FACTOR = 2
FUEL_PENALTY_SCALE = -0.001
WIND_PENALTY_VALUE = -1.0
ETA_PENALTY = -2.0
```

---

## TensorBoard Logging

Run:

```bash
tensorboard --logdir ./logs/
```

Access at: [http://localhost:6006](http://localhost:6006)

### Reward Logs Include:

* `episodic_return`
* `progress_reward`
* `fuel_penalty`
* `wind_penalty`
* `alignment_penalty`
* `eta_penalty`
* `step_penalty`
* `frequency_reward`

---

## Path Tracking Analytics

MariNav tracks how often each `(start_h3, goal_h3)` pair is selected and successfully completed. These metrics help debug agent behavior and ensure balanced training.

---

## Data Requirements

You must provide:

* `WIND_MAP_PATH`: CSV with wind speed & direction per H3 cell per timestamp
* `GRAPH_PATH`: GEXF file of navigable ocean routes
* `H3_POOL`: Valid H3 indices for start/goal sampling

---

## Contributing

1. Fork the repo
2. Make a feature branch
3. Commit and push changes
4. Open a pull request

---

## Acknowledgments

* [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
* [Uber H3](https://h3geo.org/)
* [GeoPandas](https://geopandas.org/)
* [NetworkX](https://networkx.org/)
* [Perplexity](https://www.perplexity.ai/)
* [RLeXplore](https://github.com/RLE-Foundation/RLeXplore)
---

To cite the inspiration clearly and professionally in your `README.md`, you can add an **Acknowledgment** or **Citation** section like this:

---

## Citation

The weighted graph used in `MariNav` was inspired by the following research:

**Learning Spatio-Temporal Vessel Behavior using AIS Trajectory Data and Markovian Models in the Gulf of St. Lawrence**
*Google Scholar:* [https://scholar.google.ca/citations?view\_op=view\_citation\&hl=en\&user=aiL559gAAAAJ\&citation\_for\_view=aiL559gAAAAJ:9yKSN-GCB0IC](https://scholar.google.ca/citations?view_op=view_citation&hl=en&user=aiL559gAAAAJ&citation_for_view=aiL559gAAAAJ:9yKSN-GCB0IC)

This work helped guide the design of the frequency-weighted transition graph for maritime routing in the Gulf of St. Lawrence.

---


