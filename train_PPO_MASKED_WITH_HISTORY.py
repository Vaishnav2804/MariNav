# Standard library imports
import argparse
import csv
import math  # Import math for pi
import os
from datetime import datetime, timedelta
from multiprocessing import Manager
import multiprocessing as mp

# Local application/specific imports
import matplotlib

# Third-party library imports
import networkx as nx
import numpy as np
from Env.Callbacks import *
from Env.MariNav import *
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn, get_schedule_fn
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from utils import *

# Limit thread usage to prevent CPU overload
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

matplotlib.use("Agg")  # ✅ headless backend (important!)
manager = Manager()
global_visited_path_counts = manager.dict()  # shared across processes
global_pair_selection_counts = manager.dict()

# Training parameters
DEFAULT_HISTORY_LEN = 8  # Default length of historical data to consider for training.
CALLBACK_CHECK_INTERVAL = 5000  # How often (e.g., in steps or iterations) to check for callback conditions during training.
DEFAULT_PATIENCE = 1000000  # Number of iterations with no improvement after which training will stop (for early stopping).
DEFAULT_MIN_DELTA = (
    2.0  # Minimum change in the monitored quantity to qualify as an improvement.
)

PAIR_LIST = [
    ("862b12ccfffffff", "861b91adfffffff"),
    ("861b91a0fffffff", "862b12ccfffffff"),
    ("861ab28b7ffffff", "862b12ccfffffff"),
    ("860e4daafffffff", "861ab685fffffff"),
    ("862b12ccfffffff", "862b2d137ffffff"),
    ("861ab6847ffffff", "860e4daafffffff"),
]

WIND_MAP_PATH = "wind_and_graph_2024/2024_august_wind_data.csv"  # File path to a CSV containing wind map data.
GRAPH_PATH = "wind_and_graph_2024/GULF_VISITS_cargo_tanker_2024_merged.gexf"  # File path to a GEXF file, likely representing a graph or network of cargo tanker visits in the Gulf.


def make_env(no_positive_rews: bool = False):
    def _init():
        base_env = MariNav(
            pairs=PAIR_LIST,
            graph=G_visits,
            wind_map=full_wind_map,
            h3_resolution=H3_RESOLUTION,
            wind_threshold=22,
            render_mode="human",
            no_positive_rews=no_positive_rews,
        )
        base_env.visited_path_counts = global_visited_path_counts
        base_env.pair_selection_counts = global_pair_selection_counts

        env = MariNavWithHistory(base_env, history_len=DEFAULT_HISTORY_LEN)
        return Monitor(env)

    return _init


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=31, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no-positive-rews", action="store_true", help="Disable positive rewards"
    )
    args = parser.parse_args()
    seed = args.seed
    no_positive_rews = args.no_positive_rews
    mp.set_start_method("fork", force=True)
    set_seed(seed=seed)

    print(f"Loading wind map from {WIND_MAP_PATH}...")
    full_wind_map = load_full_wind_map(WIND_MAP_PATH)
    print(f"Loading graph from {GRAPH_PATH}...")
    G_visits = nx.read_gexf(GRAPH_PATH).to_undirected()
    print("Data loading complete.")

    # 1. Wrap the base environment to include observation history
    envs = 16
    vec_env = SubprocVecEnv(
        [make_env(no_positive_rews=no_positive_rews) for _ in range(envs)]
    )
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # 2. Define policy architecture and features extractor
    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64])  # ✅ Explicit networks
    )

    # Define a log directory for TensorBoard
    learning_rate_schedule = get_linear_fn(start=7e-4, end=1e-5, end_fraction=1.0)
    # 3. Instantiate PPO model
    log_prefix = (
        "PPO_Masked_no_positive" if no_positive_rews else "PPO_MASKED_WITH_HISTORY_"
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = MaskablePPO(
        policy="MlpPolicy",
        env=vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        ent_coef=0.01,
        learning_rate=learning_rate_schedule,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        tensorboard_log=f"./tensorboard_logs/{log_prefix}_{seed}_{timestamp}/",
        device="cpu",
        seed=seed,  # For reproducibility
    )

    # 4. Print the model architecture for inspection
    print("\n--- 🔧 Model Architecture ---")
    print(model.policy)
    for name, param in model.policy.named_parameters():
        print(f"{name:<40} {list(param.shape)}")
    print("----------------------------\n")

    eval_callback = EvalCallback(
        eval_env=vec_env,  # Wrap with Monitor
        best_model_save_path=f"./models/{log_prefix}_{seed}_{timestamp}",
        log_path=f"./eval_logs/{log_prefix}_{seed}",  # important!
        eval_freq=8000,
        deterministic=False,
        render=False,
    )

    # Set up Early Stopping Callback
    early_stop = EarlyStoppingCallback(
        log_path=f"./eval_logs/{log_prefix}_{seed}",
        patience=DEFAULT_PATIENCE,
        min_delta=2.0,
        check_freq=16000,
    )

    step_logger = StepRewardLoggerCallback()
    info_logging_callback = InfoLoggingCallback()
    ep_stat_callback = EpisodeStatsCallback()

    callback = CallbackList(
        [
            eval_callback,
            early_stop,
            step_logger,
            info_logging_callback,
            ep_stat_callback,
        ]
    )

    # Train the model
    print(f"Starting training for {10_000_000} timesteps...")
    model.learn(total_timesteps=10_000_000, callback=callback)
    print("Training finished.")
