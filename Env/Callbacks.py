import hashlib
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class EarlyStoppingCallback(BaseCallback):
    """
    A custom callback for implementing early stopping based on the performance
    of an agent during training. This callback monitors a specified metric
    (e.g., mean reward from evaluations) and stops training if the metric
    doesn't improve significantly over a certain number of checks.

    It also saves the model periodically at `check_freq` intervals.

    :param log_path: The directory where evaluation logs (e.g., 'evaluations.npz')
                     and models are saved.
    :param patience: The number of `check_freq` intervals to wait for improvement
                     before stopping training.
    :param min_delta: The minimum change in the monitored metric to qualify as an
                      improvement. Improvements smaller than this value are not
                      considered significant.
    :param check_freq: How often (in training steps/timesteps) to check for
                       early stopping conditions and save the model.
    :param verbose: Verbosity level, passed to the `BaseCallback` constructor.
                    (0 for no output, 1 for info, 2 for debug).
    """

    def __init__(
        self,
        log_path: str,
        patience: int = 10,
        min_delta: float = 1.0,
        check_freq: int = 10000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.log_path = log_path
        self.patience = patience
        self.min_delta = min_delta
        self.check_freq = check_freq
        self.counter = 0
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq == 0:
            try:
                monitor_file = os.path.join(self.log_path, "evaluations.npz")
                if os.path.exists(monitor_file):
                    data = np.load(monitor_file)
                    rewards = data["results"]
                    if rewards.shape[0] > 0:
                        latest_mean = np.mean(rewards[-1])
                        if self.verbose:
                            print(
                                f"[EarlyStopping] Step {self.num_timesteps}: mean eval reward = {latest_mean:.2f}"
                            )

                        model_path = os.path.join(
                            self.log_path, f"model_step_{self.num_timesteps}"
                        )
                        self.model.save(model_path)

                        if latest_mean > self.best_mean_reward + self.min_delta:
                            print(
                                f"✅ New best model! Mean reward improved from {self.best_mean_reward:.2f} to {latest_mean:.2f}"
                            )
                            self.best_mean_reward = latest_mean
                            self.counter = 0
                        else:
                            self.counter += 1
                            print(
                                f"[EarlyStopping] No improvement for {self.counter} evaluations"
                            )

                        if self.counter >= self.patience:
                            print("🛑 Early stopping triggered.")
                            return False
            except Exception as e:
                print(f"[EarlyStopping] Failed to read eval log: {e}")

        return True


class StepRewardLoggerCallback(BaseCallback):
    """
    A simple custom callback for logging the 'reward_per_step' metric during training.

    This callback inherits from `BaseCallback` and is designed to extract and
    record the immediate reward received at each training step. It's useful
    for monitoring the granular reward signal over time, often visualized
    using tools like TensorBoard.

    :param verbose: Verbosity level, passed to the `BaseCallback` constructor.
                    (0 for no output, 1 for info, 2 for debug).
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        for i, info in enumerate(self.locals["infos"]):
            if "reward_per_step" in info:
                self.logger.record("reward/step_reward", info["reward_per_step"])
        return True


def short_hash(s, length=8):
    return hashlib.md5(s.encode()).hexdigest()[:length]


class InfoLoggingCallback(BaseCallback):
    """
    A custom callback for logging various training information, including per-step
    rewards and penalties, as well as episode-specific metrics.

    This callback extends `BaseCallback` and is designed to extract and log
    details from the `info` dictionary provided during the training process.
    It logs different reward components, penalties, and other relevant metrics
    to a logger (e.g., for TensorBoard visualization).

    :param verbose: Verbosity level, passed to the `BaseCallback` constructor.
                    (0 for no output, 1 for info, 2 for debug).
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            # -------- Per-Step Logging --------
            self._log_if_present(info, "self_progress_reward", "step/progress_reward")
            self._log_if_present(info, "self_frequency_reward", "step/frequency_reward")
            self._log_if_present(info, "self_wind_penalty", "step/wind_penalty")
            self._log_if_present(
                info, "self_alignment_penalty", "step/alignment_penalty"
            )
            self._log_if_present(info, "self_fuel_penalty", "step/fuel_penalty")
            self._log_if_present(info, "self_eta_penalty", "step/eta_penalty")
            self._log_if_present(
                info, "self_base_step_penalty", "step/base_step_penalty"
            )
            self._log_if_present(info, "self_total_reward", "step/total_step_reward")

            self._log_if_present(info, "reward_per_step", "step/reward_per_step")
            self._log_if_present(info, "distance_to_goal", "step/distance_to_goal")
            self._log_if_present(info, "speed", "step/speed")
            if "visited_path_counts" in info:
                for path_key, count in info["visited_path_counts"].items():
                    short_key = short_hash(path_key)
                    tb_key = f"episode/visited_path/{short_key}"
                    self.logger.record(tb_key, count)

            if "pair_selection_counts" in info:
                for path_key, count in info["pair_selection_counts"].items():
                    short_key = short_hash(path_key)
                    tb_key = f"episode/pair_selected/{short_key}"
                    self.logger.record(tb_key, count)

            # -------- Per-Episode Logging --------
            if "epi" in info:
                episode_info = info["epi"]
                self._log_if_present(episode_info, "r", "episode/total_reward")
                self._log_if_present(episode_info, "l", "episode/length")
                self._log_if_present(
                    episode_info, "progress_reward", "episode/progress_reward"
                )
                self._log_if_present(
                    episode_info, "frequency_reward", "episode/frequency_reward"
                )
                self._log_if_present(
                    episode_info, "wind_penalty", "episode/wind_penalty"
                )
                self._log_if_present(
                    episode_info, "alignment_penalty", "episode/alignment_penalty"
                )
                self._log_if_present(
                    episode_info, "fuel_penalty", "episode/fuel_penalty"
                )
                self._log_if_present(episode_info, "eta_penalty", "episode/eta_penalty")
                self._log_if_present(
                    episode_info, "base_step_penalty", "episode/base_step_penalty"
                )
                self._log_if_present(episode_info, "true_etr", "episode/true_etr")

        return True

    def _log_if_present(self, info_dict, key, tb_key):
        if key in info_dict:
            self.logger.record(tb_key, info_dict[key])



class EpisodeStatsCallback(BaseCallback):
    """
    Reproduce SB3's ep_rew_mean logic, but using info["epi"].
    Aggregates over episodes finished during *this rollout* only.
    """

    def __init__(self, prefix="rollout", verbose=0):
        super().__init__(verbose)
        self.prefix = prefix
        self._epis_this_rollout = []

    def _on_rollout_start(self) -> None:
        # Reset at start of rollout
        self._epis_this_rollout = []

    def _on_step(self) -> bool:
        dones = self.locals["dones"]
        infos = self.locals["infos"]
        for done, info in zip(dones, infos):
            if done and "epi" in info and isinstance(info["epi"], dict):
                # store one dict per finished episode
                self._epis_this_rollout.append(info["epi"].copy())
        return True

    def _on_rollout_end(self) -> None:
        print(self._epis_this_rollout)
        if not self._epis_this_rollout:
            return

        # Aggregate like SB3: mean over just-this-rollout episodes
        keys = set().union(*[epi.keys() for epi in self._epis_this_rollout])
        for key in sorted(keys):
            values = [epi[key] for epi in self._epis_this_rollout if key in epi]
            # Only log numeric values
            values = [float(v) for v in values if isinstance(v, (int, float))]
            if values:
                mean_val = sum(values) / len(values)
                self.logger.record(f"{self.prefix}/{key}_mean", mean_val)

