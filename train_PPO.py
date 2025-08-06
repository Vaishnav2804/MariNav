# Standard library imports
import csv
import math  # Import math for pi
import multiprocessing as mp
import os
from collections import defaultdict
from datetime import datetime, timedelta
from multiprocessing import Manager

# Local application/specific imports
import matplotlib

# Third-party library imports
import networkx as nx
import numpy as np
from Env.Callbacks import *
from Env.MariNav import *
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn, get_schedule_fn
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from utils import *

matplotlib.use("Agg")  # ✅ headless backend (important!)


# Training parameters
DEFAULT_HISTORY_LEN = 12
CALLBACK_CHECK_INTERVAL = 5000
DEFAULT_PATIENCE = 1000000
DEFAULT_MIN_DELTA = 2.0

H3_POOL = [
    "862b160d7ffffff",
    "860e4da17ffffff",
    "861b9101fffffff",
    "862b256dfffffff",
    "862b33237ffffff",
]

WIND_MAP_PATH = "august_2018_60min_windmap_v2.csv"
GRAPH_PATH = "GULF_VISITS_CARGO_TANKER_AUGUST_2018.gexf"

manager = Manager()
global_visited_path_counts = manager.dict()  # shared across processes
global_pair_selection_counts = manager.dict()


def make_env():
    def _init():
        env = MariNav(
            h3_pool=H3_POOL,
            graph=G_visits,
            wind_map=full_wind_map,
            h3_resolution=H3_RESOLUTION,
            wind_threshold=22,
            render_mode="human",
        )
        env.visited_path_counts = global_visited_path_counts
        env.pair_selection_counts = global_pair_selection_counts
        return Monitor(env)

    return _init


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    print(f"Loading wind map from {WIND_MAP_PATH}...")
    full_wind_map = load_full_wind_map(WIND_MAP_PATH)
    print(f"Loading graph from {GRAPH_PATH}...")
    G_visits = nx.read_gexf(GRAPH_PATH).to_undirected()
    print("Data loading complete.")

    # 1. Wrap the base environment to include observation history
    envs = 16
    vec_env = SubprocVecEnv([make_env() for _ in range(envs)])
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # 2. Define policy architecture and features extractor
    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64])  # ✅ Explicit networks
    )

    # Define a log directory for TensorBoard
    learning_rate_schedule = get_linear_fn(start=7e-4, end=1e-5, end_fraction=1.0)
    # 3. Instantiate PPO model

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        ent_coef=0.01,
        learning_rate=learning_rate_schedule,
        n_steps=2048,
        batch_size=128,
        n_epochs=15,
        tensorboard_log="./logs/",
        device="cpu",
    )

    # 4. Print the model architecture for inspection
    print("\n--- 🔧 Model Architecture ---")
    print(model.policy)
    for name, param in model.policy.named_parameters():
        print(f"{name:<40} {list(param.shape)}")
    print("----------------------------\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    eval_callback = EvalCallback(
        eval_env=vec_env,  # Wrap with Monitor
        best_model_save_path=f"./ppo_gulf_tanker_MLP_PPO_240000000_{timestamp}",
        log_path="./eval_logs",  # important!
        eval_freq=8000,
        deterministic=False,
        render=False,
    )

    # Set up Early Stopping Callback
    early_stop = EarlyStoppingCallback(
        log_path="./eval_logs",
        patience=DEFAULT_PATIENCE,
        min_delta=2.0,
        check_freq=16000,
    )

    step_logger = StepRewardLoggerCallback()
    info_logging_callback = InfoLoggingCallback()

    callback = CallbackList(
        [eval_callback, early_stop, step_logger, info_logging_callback]
    )

    # Train the model
    print(f"Starting training for {240000000} timesteps...")
    model.learn(total_timesteps=240_000_000, callback=callback)
    print("Training finished.")
