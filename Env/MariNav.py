# Standard library imports
import csv
import math  # Import math for pi
import os
import random
from collections import defaultdict
from datetime import datetime, timedelta

import contextily as ctx

# Third-party library imports
# Local application/specific imports
import geopandas as gpd
import gymnasium as gym
import h3
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from gymnasium import spaces
from shapely.geometry import Polygon

# --- Constants and Configuration ---
# Define common constants used in the environment
KNOTS_TO_MPS = 0.514444
MIN_SPEED_KNOTS = 8.0
MAX_SPEED_KNOTS = 22.0
NUM_SPEED_LEVELS = 5
H3_RESOLUTION = 6
DEFAULT_WIND_THRESHOLD = 10.0  # m/s
DEFAULT_HORIZON_MULTIPLIER = 5

# Reward coefficients
GOAL_REWARD = 0
INVALID_MOVE_PENALTY = -1900
PROGRESS_REWARD_FACTOR = 2
FREQUENCY_REWARD_CLIP = 0.5
WIND_PENALTY_VALUE = -1.0
FUEL_PENALTY_SCALE = -0.001
ETA_PENALTY_SCALE = -0.001
SPEED_BONUS_FACTOR = 0.3  # Not used for now.
BASE_STEP_PENALTY = -1

# Fuel consumption parameters
FUEL_SPEED_FACTOR = 0.05
FUEL_WIND_FACTOR = 0.02
FUEL_DRAG_FACTOR = 0.5

# Observation space normalization
OBS_LOW_BOUND = 0
OBS_HIGH_BOUND = 1

# CSV Logging
CSV_BUFFER_SIZE = 10000
CSV_FILE_PATH = "environment_log.csv"

# Training parameters
DEFAULT_HISTORY_LEN = 8
CALLBACK_CHECK_INTERVAL = 5000
DEFAULT_PATIENCE = 200
DEFAULT_MIN_DELTA = 2.0


class MariNav(gym.Env):
    """
    A reinforcement learning environment for simulating a tanker's movement
    across H3 hexagonal grid cells, considering wind conditions and fuel consumption.
    """

    def __init__(
        self,
        graph: nx.Graph,
        wind_map: dict,
        pairs: list[tuple[str, str]],
        h3_resolution: int = H3_RESOLUTION,
        wind_threshold: float = DEFAULT_WIND_THRESHOLD,
        no_positive_rews: bool = False,
        render_mode: str = None,
    ):
        """
        Initializes the MariNav.

        Args:
            graph (nx.Graph): A NetworkX graph representing navigable H3 cells.
                              Edges should ideally have a 'weight' attribute indicating
                              visit frequency or other relevant metric.
            wind_map (dict): A dictionary where keys are H3 cell IDs and values are
                             another dictionary: {timestamp: (u_wind, v_wind)}.
                             'u_wind' is the east-west wind component, 'v_wind' is the north-south.
            start_h3_list (list[str]): List of possible starting H3 indices.
            goal_h3_list (list[str]): List of possible goal H3 indices.
            h3_resolution (int): The H3 resolution for the grid. Defaults to 6.
            wind_threshold (float): The wind speed (in m/s) above which a penalty is applied.
        """
        super().__init__()

        self.graph = graph
        self.wind_map = wind_map
        self.wind_threshold = wind_threshold
        self.render_mode = render_mode
        self.prioritize_until = 0
        self.no_positive_rews = no_positive_rews

        self.pairs = pairs

        # Collect all H3 IDs seen across pairs so we can validate them
        self.valid_hex_ids = set(self.graph.nodes)
        self.h3_pool = list({h for pair in pairs for h in pair})

        # Validate pool nodes exist in graph
        for hexagon in self.h3_pool:
            if hexagon not in self.graph:
                raise ValueError(f"H3 cell {hexagon} not in graph.")

        self.pair_selection_counts = defaultdict(int)
        self.h3_resolution = h3_resolution

        # Validate Start/Goal Nodes Are Reachable
        for hexagon in self.h3_pool:
            if hexagon not in self.graph:
                raise ValueError(f"Start H3 cell {hexagon} not in graph.")

        self.start_h3: str | None = None
        self.goal_h3: str | None = None
        self.observation_space = spaces.Box(
            low=OBS_LOW_BOUND, high=OBS_HIGH_BOUND, shape=(8,), dtype=np.float32
        )

        self.speed_range = (MIN_SPEED_KNOTS, MAX_SPEED_KNOTS)
        self.k = 5  # Maximum number of neighbors to consider for action space
        self.action_space = spaces.MultiDiscrete([self.k, NUM_SPEED_LEVELS])

        self._distance_cache = {}
        self.total_steps = 0
        # CSV logging attributes (currently commented out in step, can be reactivated)
        self.csv_buffer: list[dict] = []
        self.csv_buffer_size = CSV_BUFFER_SIZE
        self.csv_file_path = CSV_FILE_PATH
        self.csv_header_written = False
        self.visited_path_counts = defaultdict(int)  # key: (start_h3, goal_h3)
        self.max_distance_reference = self.shortest_path_length(
            self.pairs[0][0],  # src (0th in tuple)
            self.pairs[0][1],  # dst (1st in tuple)
        )
        self.reset()

        try:
            self.max_distance = nx.shortest_path_length(
                self.graph, self.start_h3, self.goal_h3
            )
        except nx.NetworkXNoPath:
            raise ValueError(
                f"No path exists from {self.start_h3} to {self.goal_h3} in water-only graph."
            )

        if self.max_distance > 100:
            self.horizon = int(self.max_distance * DEFAULT_HORIZON_MULTIPLIER)
        elif self.max_distance > 50:
            self.horizon = int(self.max_distance * 1.75)
        else:
            self.horizon = int(self.max_distance * 1.5)

    def shortest_path_length(self, src: str, dst: str) -> int:
        key = (src, dst)
        if key not in self._distance_cache:
            if src not in self.graph or dst not in self.graph:
                raise ValueError(f"[Path Error] {src} or {dst} not in graph.")
            try:
                self._distance_cache[key] = nx.shortest_path_length(
                    G=self.graph, source=src, target=dst
                )
            except nx.NetworkXNoPath:
                raise RuntimeError(f"[Path Error] No path from {src} to {dst}")
        return self._distance_cache[key]

    def get_all_neighbors(self):
        all_neighbors = list(h3.grid_ring(self.current_h3))
        neighbors = self._get_valid_neighbors()

        # Step 1: Remove prev_h3 if it's there
        if hasattr(self, "prev_h3") and self.prev_h3 in all_neighbors:
            all_neighbors.remove(self.prev_h3)

        # Step 2: Ensure len == self.k (e.g. 5) by removing ONE neighbor
        if len(all_neighbors) > self.k:
            # Prefer removing the current cell if it sneaks in
            if self.current_h3 in all_neighbors:
                all_neighbors.remove(self.current_h3)
            else:
                # Try to remove a non-valid neighbor first
                removable_nonvalid = [n for n in all_neighbors if n not in neighbors]

                if removable_nonvalid:
                    # Deterministically remove the first non-valid (or pick by some stable key)
                    # Using lexicographic order for stability:
                    to_remove = sorted(removable_nonvalid)[0]
                    all_neighbors.remove(to_remove)
                else:
                    # All are valid neighbors → remove the one with the least edge weight
                    # Edge weights default to 0 if missing
                    def edge_weight(nbr: str) -> float:
                        if self.graph.has_edge(self.current_h3, nbr):
                            return self.graph[self.current_h3][nbr].get("weight", 0.0)
                        # If there's no explicit edge (shouldn't happen for valid neighbors), use 0
                        return 0.0

                    # Choose neighbor with minimum weight; break ties deterministically by H3 index
                    to_remove = min(neighbors, key=lambda n: (edge_weight(n), n))
                    all_neighbors.remove(to_remove)

        # Save for masks
        self.all_neighbors = all_neighbors
        self.neighbors = [n for n in neighbors if n in all_neighbors]
        return all_neighbors, self.neighbors

    def action_masks(self) -> list[np.ndarray]:
        """
        Returns action masks for MultiDiscrete action space.

        For MultiDiscrete spaces, this should return a list of boolean arrays,
        one for each sub-action space.

        Returns:
            list[np.ndarray]: List containing [neighbor_mask, speed_mask]
                            where each mask is a boolean array
        """
        self.is_maskable = True
        all_neighbors, neighbors = self.get_all_neighbors()

        # Now all_neighbors is of length self.k, and contains as many valid neighbors as possible
        assert len(all_neighbors) == self.k

        # Neighbor mask
        neighbor_mask = np.zeros(
            self.k, dtype=bool
        )  # self.k = len(all_neighbors) assumed

        if neighbors:
            valid_indices = [
                all_neighbors.index(n) for n in neighbors if n in all_neighbors
            ]
            neighbor_mask[valid_indices] = True

        # Mask for speed selection (second sub-action)
        # All speeds are always valid regardless of position
        speed_mask = np.ones(NUM_SPEED_LEVELS, dtype=bool)

        return [neighbor_mask, speed_mask]

    def _set_latlon_bounds(self) -> None:
        """
        Calculates and sets the min/max latitude and longitude of all H3 cells
        in the graph. These bounds are used for normalizing observations.
        """
        lats, lons = zip(*[h3.cell_to_latlng(h) for h in self.graph.nodes])
        self.min_lat, self.max_lat = min(lats), max(lats)
        self.min_lon, self.max_lon = min(lons), max(lons)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.is_maskable = False
        # All possible (start, goal) pairs where start ≠ goal
        pairs = self.pairs
        # Trigger prioritization every 1M steps, for the next 10k steps
        if self.total_steps > (62500):
            self.prioritize_until = 6250
            self.total_steps = 0

        if self.total_steps < self.prioritize_until:
            # Prioritized sampling
            if not self.visited_path_counts:
                selected_idx = np.random.choice(len(pairs))
            else:
                counts = np.array(
                    [self.visited_path_counts.get((s, g), 0) for (s, g) in pairs]
                )
                max_count = max(counts.max(), 1)
                weights = max_count - counts + 1
                probs = weights / weights.sum()
                selected_idx = np.random.choice(len(pairs), p=probs)
        else:
            # Uniform random sampling
            selected_idx = np.random.choice(len(pairs))

        self.start_h3, self.goal_h3 = pairs[selected_idx]
        key = (self.start_h3, self.goal_h3)
        self.pair_selection_counts[key] = self.pair_selection_counts.get(key, 0) + 1
        self.current_h3 = self.start_h3

        # Reset other episode state
        self.prev_h3 = None
        self.step_count = 0
        self.episode_reward = 0.0
        self.episode_neg_reward = 0.0
        self.episode_wind_penalty = 0.0
        self.episode_fuel_penalty = 0.0
        self.episode_eta_penalty = 0.0
        self.episode_base_step_penalty = 0.0
        self.episode_progress_reward = 0.0
        self.episode_frequency_reward = 0.0
        self.trajectory = []

        random_day = np.random.randint(1, 30)
        random_hour = np.random.randint(0, 24)
        self.current_time = datetime(2022, 8, random_day, random_hour, 0)
        self.episode_start_time = self.current_time

        total_distance_m = self.shortest_path_length(self.start_h3, self.goal_h3)
        self.target_eta = total_distance_m / (
            10 * KNOTS_TO_MPS
        )  # seconds, assuming 10 knots
        self.eta_window = 1800  # ±30 minutes in seconds

        self._set_latlon_bounds()

        return self._get_observation(), {}

    def _get_valid_neighbors(self) -> list[str]:
        """
        Gets valid neighbors for the current H3 cell, excluding the previous H3 cell
        to prevent immediate U-turns.
        """
        neighbors = list(self.graph.neighbors(self.current_h3))
        if self.prev_h3 in neighbors:
            neighbors.remove(self.prev_h3)
        return neighbors

    def _calculate_rewards(
        self,
        distance_before: int,
        distance_after: int,
        speed: float,
        wind_speed: float,
        move_direction: float,
        wind_direction: float,
    ) -> float:
        """
        Calculates the various reward components for the current step.
        """
        # Progress reward: based on reduction in distance to goal
        progress_reward = PROGRESS_REWARD_FACTOR * (distance_before - distance_after)

        # Frequency reward: based on historical visit frequency of the edge
        if self.prev_h3 and self.graph.has_edge(self.prev_h3, self.current_h3):
            edge_weight = self.graph[self.prev_h3][self.current_h3].get("weight", 1)
        else:
            edge_weight = 0  # or some penalty

        frequency_reward = np.clip(np.log1p(edge_weight) / 5, 0, FREQUENCY_REWARD_CLIP)

        # Wind penalty: if wind speed exceeds threshold
        wind_penalty = WIND_PENALTY_VALUE if wind_speed > self.wind_threshold else 0.0

        # Speed bonus: positive reward for moving when progress is made
        # speed_bonus = SPEED_BONUS_FACTOR * speed if progress_reward > 0 else 0

        # Fuel penalty: based on estimated fuel consumption
        fuel_consumed = self._estimate_fuel(
            speed, wind_speed, move_direction, wind_direction
        )
        fuel_penalty = FUEL_PENALTY_SCALE * fuel_consumed

        # ETA penalty: based on travel time (encourages faster routes)
        travel_time = self._estimate_travel_time(speed)
        eta_penalty = ETA_PENALTY_SCALE * travel_time

        # Base step penalty
        base_step_penalty = BASE_STEP_PENALTY

        total_reward = (
            progress_reward
            + frequency_reward
            + wind_penalty
            + fuel_penalty
            + eta_penalty
            + base_step_penalty
        )

        return {
            "total_reward": total_reward,
            "progress_reward": progress_reward,
            "frequency_reward": frequency_reward,
            "wind_penalty": wind_penalty,
            "fuel_penalty": fuel_penalty,
            "eta_penalty": eta_penalty,
            "base_step_penalty": base_step_penalty,
        }

    def _get_current_wind_conditions(self) -> tuple[float, float, float]:
        """
        Retrieves wind speed and direction for the current H3 cell and time.
        Returns (wind_u, wind_v, wind_speed, wind_direction).
        """
        h3_wind_data = self.wind_map.get(self.current_h3, {})

        if not h3_wind_data:
            return 0.0, 0.0, 0.0, 0.0  # No wind data available

        # Find closest timestamp from h3_wind_data to self.current_time
        target_time = pd.Timestamp(self.current_time)
        closest_time = min(h3_wind_data.keys(), key=lambda t: abs(t - target_time))
        wind_u, wind_v = h3_wind_data[closest_time]

        wind_speed = np.sqrt(wind_u**2 + wind_v**2)
        wind_direction = np.arctan2(wind_v, wind_u)  # Radians

        return wind_u, wind_v, wind_speed, wind_direction
    
    def _finalize_epi(self, info: dict) -> dict:
        """
        Attach episode statistics to info when an episode ends.
        Mirrors how Monitor provides info["episode"].
        """
        info.update(
            {
                "epi": {
                    "r": (
                        self.episode_neg_reward
                        if self.no_positive_rews
                        else self.episode_reward
                    ),
                    "l": self.step_count,
                    "progress_reward": self.episode_progress_reward,
                    "frequency_reward": self.episode_frequency_reward,
                    "wind_penalty": self.episode_wind_penalty,
                    "fuel_penalty": self.episode_fuel_penalty,
                    "eta_penalty": self.episode_eta_penalty,
                    "base_step_penalty": self.episode_base_step_penalty,
                    "episode_reward": self.episode_reward,
                    "episode_neg_reward": self.episode_neg_reward,
                }
            }
        )
        return info


    def step(
        self, action: tuple[int, int]
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Takes one step in the environment given an action.

        Args:
            action (tuple): An action taken by the agent, composed of (neighbor_idx, speed_level).

        Returns:
            tuple: A tuple containing:
                - observation (np.array): The new observation of the environment.
                - reward (float): The reward received from the action.
                - terminated (bool): True if the episode has ended (goal reached).
                - truncated (bool): True if the episode has ended (horizon reached).
                - info (dict): A dictionary containing additional information about the step.
        """
        self.total_steps += 1
        self.step_count += 1
        info: dict = {}
        info["revisiting_loop"] = 0

        neighbor_idx, speed_level = action
        # neighbor_idx, speed_level = action
        speed = np.linspace(self.speed_range[0], self.speed_range[1], NUM_SPEED_LEVELS)[
            speed_level
        ]

        if self.is_maskable:
            ring = self.all_neighbors
            neighbors = self.neighbors
        else:
            ring, neighbors = self.get_all_neighbors()

        # Step 1: Check ring index range
        if neighbor_idx >= len(ring):
            raise Exception("neighbor_idx out of ring bounds")

        if not neighbors:
            self.episode_reward += INVALID_MOVE_PENALTY
            info = {"reason": "no valid neighbors"}
            info = self._finalize_epi(info)
            return (
                self._get_observation(),
                INVALID_MOVE_PENALTY,
                True,
                False,
                info,
            )

        # Step 2: Get candidate neighbor from ring
        candidate = ring[neighbor_idx]
        # Step 3: Make sure it's in valid neighbors
        if candidate not in neighbors:
            self.episode_reward += INVALID_MOVE_PENALTY
            info = {"reason": "selected ring neighbor not in valid neighbors"}
            info = self._finalize_epi(info)
            return (
                self._get_observation(),
                INVALID_MOVE_PENALTY,
                True,
                False,
                info,
            )

        # Step 4: It's valid
        selected_neighbor = candidate

        try:
            distance_before = self.shortest_path_length(self.current_h3, self.goal_h3)
            self.prev_h3 = self.current_h3
            self.current_h3 = selected_neighbor
            distance_after = self.shortest_path_length(selected_neighbor, self.goal_h3)
        except nx.NetworkXNoPath:
            # No valid path exists — strongly penalize the move
            # Invalid move — path is blocked (likely due to land)
            return (
                self._get_observation(),
                INVALID_MOVE_PENALTY,
                False,
                True,
                {
                    "reason": "no ocean path to goal",
                    "current_h3": self.current_h3,
                    "selected_neighbor": selected_neighbor,
                    "goal_h3": self.goal_h3,
                    "step_count": self.step_count,
                },
            )

        # self.buffer_selected_info(info) # Uncomment to reactivate CSV logging
        self.trajectory.append(self.current_h3)

        travel_time = self._estimate_travel_time(speed)
        self.current_time += timedelta(seconds=travel_time)

        terminated = self.current_h3 == self.goal_h3
        truncated = self.step_count >= self.horizon
        done = terminated or truncated

        # Get wind conditions
        wind_u, wind_v, wind_speed, wind_direction = self._get_current_wind_conditions()

        # Compute vessel movement direction (heading)
        lat1, lon1 = (
            h3.cell_to_latlng(self.prev_h3) if self.prev_h3 else (0, 0)
        )  # Handle initial step
        lat2, lon2 = h3.cell_to_latlng(self.current_h3)
        move_direction = np.arctan2(lat2 - lat1, lon2 - lon1) if self.prev_h3 else 0.0

        calculated_rewards = self._calculate_rewards(
            distance_before,
            distance_after,
            speed,
            wind_speed,
            move_direction,
            wind_direction,
        )

        step_wind_penalty = calculated_rewards["wind_penalty"]
        step_fuel_penalty = calculated_rewards["fuel_penalty"]
        step_eta_penalty = calculated_rewards["eta_penalty"]
        step_base_step_penalty = calculated_rewards["base_step_penalty"]
        step_progress_reward = calculated_rewards["progress_reward"]
        step_frequency_reward = calculated_rewards["frequency_reward"]
        step_total_reward = calculated_rewards["total_reward"]
        step_total_neg_reward = (
            step_total_reward - step_progress_reward - step_frequency_reward
        )

        if terminated:
            key = (self.start_h3, self.goal_h3)
            self.visited_path_counts[key] = self.visited_path_counts.get(key, 0) + 1
            step_total_reward += GOAL_REWARD

        distance_scaler = self.max_distance_reference / self.max_distance
        step_total_reward = step_total_reward * distance_scaler
        step_total_neg_reward = step_total_neg_reward * distance_scaler

        self.episode_reward += step_total_reward
        self.episode_wind_penalty += step_wind_penalty
        self.episode_fuel_penalty += step_fuel_penalty
        self.episode_eta_penalty += step_eta_penalty
        self.episode_base_step_penalty += step_base_step_penalty
        self.episode_progress_reward += step_progress_reward
        self.episode_frequency_reward += step_frequency_reward
        self.episode_neg_reward += step_total_neg_reward

        if done:
            print("Done")
            info.update(
                {
                    "epi": {
                        "r": (
                            self.episode_neg_reward
                            if self.no_positive_rews
                            else self.episode_reward
                        ),
                        "l": self.step_count,
                        "progress_reward": self.episode_progress_reward,
                        "frequency_reward": self.episode_frequency_reward,
                        "wind_penalty": self.episode_wind_penalty,
                        "fuel_penalty": self.episode_fuel_penalty,
                        "eta_penalty": self.episode_eta_penalty,
                        "base_step_penalty": self.episode_base_step_penalty,
                        "episode_reward": self.episode_reward,
                        "episode_neg_reward": self.episode_neg_reward,
                    },
                    "visited_path_counts": {
                        f"{s}->{g}": count
                        for (s, g), count in self.visited_path_counts.items()
                    },
                    "pair_selection_counts": {
                        f"{s}->{g}": count
                        for (s, g), count in self.pair_selection_counts.items()
                    },
                }
            )

        info["distance_before"] = distance_before
        info["distance_after"] = distance_after

        info.update(
            {
                "terminated": terminated,
                "truncated": truncated,
                "step_count": self.step_count,
                "current_h3": self.current_h3,
                "prev_h3": self.prev_h3,
                "distance_to_goal": distance_after,
                # Step-level reward components with self_ prefix
                "self_progress_reward": step_progress_reward,
                "self_frequency_reward": step_frequency_reward,
                "self_wind_penalty": step_wind_penalty,
                "self_fuel_penalty": step_fuel_penalty,
                "self_eta_penalty": step_eta_penalty,
                "self_base_step_penalty": step_base_step_penalty,
                "self_total_reward": step_total_reward,
                # Environment dynamics
                "speed": speed,
                "current_time": self.current_time,
                "wind_direction": wind_direction,
                "move_direction": move_direction,
            }
        )

        info["step_action_continuous"] = action
        info["step_action_discrete"] = (neighbor_idx, speed_level)
        info["reward_per_step"] = step_total_reward

        return (
            self._get_observation(speed, wind_direction),
            step_total_neg_reward if self.no_positive_rews else step_total_reward,
            terminated,
            truncated,
            info,
        )

    def _estimate_fuel(
        self,
        speed: float,
        wind_speed: float,
        move_direction: float,
        wind_direction: float,
    ) -> float:
        """
        Estimates the fuel consumed during a step based on vessel speed, wind speed,
        and the alignment between vessel movement and wind direction.

        Args:
            speed (float): Vessel speed in knots.
            wind_speed (float): Wind speed magnitude in m/s.
            move_direction (float): Ship's heading in radians.
            wind_direction (float): Wind direction in radians.

        Returns:
            float: Estimated fuel consumed.
        """
        # Calculate a factor based on the alignment of movement and wind.
        # When aligned, cos(0) = 1, angle_factor = 2. When opposite, cos(pi) = -1, angle_factor = 0.
        angle_factor = 1 + np.cos(move_direction - wind_direction)
        # Adjust drag: worse (higher) when sailing against the wind (angle_factor is low)
        adjusted_drag = 1 + FUEL_DRAG_FACTOR * (1 - angle_factor)
        # Fuel consumption formula: (speed^3 component adjusted by drag) + (wind_speed component)
        return (
            FUEL_SPEED_FACTOR * (speed**3) * adjusted_drag
            + FUEL_WIND_FACTOR * wind_speed
        )

    def _estimate_travel_time(self, speed: float) -> float:
        """
        Estimates the time (in seconds) it takes to travel between two adjacent
        H3 cells based on the given speed and the average edge length of an H3 cell.

        Args:
            speed (float): Vessel speed in knots.

        Returns:
            float: Estimated travel time in seconds.
        """
        meters_per_h3_grid_unit = h3.average_hexagon_edge_length(
            self.h3_resolution, unit="m"
        )

        grid_distance = (
            1
            if self.prev_h3 is None
            else h3.grid_distance(self.prev_h3, self.current_h3)
        )

        distance_m = grid_distance * meters_per_h3_grid_unit

        speed_mps = speed * KNOTS_TO_MPS
        return distance_m / max(speed_mps, 1e-6)  # Avoid division by zero

    def _get_observation(
        self, speed: float = MIN_SPEED_KNOTS, wind_direction: float = 0.0
    ) -> np.ndarray:
        """
        Generates the observation array for the agent.

        Args:
            speed (float): Current vessel speed. Defaults to `MIN_SPEED_KNOTS` if not provided.
            wind_direction (float): Current wind direction in radians. Defaults to 0.0 if not provided.

        Returns:
            np.array: A 8-element numpy array representing the observation:
                    [norm_lat, norm_lon, norm_speed, norm_wind_dir,
                    norm_start_lat, norm_start_lon, norm_goal_lat, norm_goal_lon]
        """

        def normalize_latlon(h3_index):
            lat, lon = h3.cell_to_latlng(h3_index)
            norm_lat = (lat - self.min_lat) / (self.max_lat - self.min_lat)
            norm_lon = (lon - self.min_lon) / (self.max_lon - self.min_lon)
            return norm_lat, norm_lon

        # Normalize current location
        norm_lat, norm_lon = normalize_latlon(self.current_h3)

        # Normalize wind direction from [-pi, pi] to [0, 1]
        norm_wind_dir = (wind_direction + math.pi) / (2 * math.pi)

        # Normalize speed using log scale
        log_min = np.log10(self.speed_range[0] + 1e-5)
        log_max = np.log10(self.speed_range[1] + 1e-5)
        norm_speed = (np.log10(speed + 1e-5) - log_min) / (log_max - log_min)

        # Normalize start and goal coordinates
        norm_start_lat, norm_start_lon = normalize_latlon(self.start_h3)
        norm_goal_lat, norm_goal_lon = normalize_latlon(self.goal_h3)

        # Final observation
        obs = np.array(
            [
                norm_lat,
                norm_lon,
                norm_speed,
                norm_wind_dir,
                norm_start_lat,
                norm_start_lon,
                norm_goal_lat,
                norm_goal_lon,
            ],
            dtype=np.float32,
        )

        obs = np.nan_to_num(obs, nan=0.0, posinf=OBS_HIGH_BOUND, neginf=OBS_LOW_BOUND)
        assert np.all(np.isfinite(obs)), f"Non-finite obs: {obs}"
        return obs

    def render(self, mode: str = "human") -> None:
        if mode != "human":
            super().render(mode=mode)
            return

        render_dir = "renders"
        os.makedirs(render_dir, exist_ok=True)

        # Create ocean GeoDataFrame once
        cell_polys = []
        for h in self.valid_hex_ids:
            boundary = h3.cell_to_boundary(h)
            poly = Polygon(boundary)
            cell_polys.append({"h3": h, "geometry": poly})
        ocean_gdf = gpd.GeoDataFrame(cell_polys, crs="EPSG:4326").to_crs(epsg=3857)

        # Trajectory GeoDataFrame
        traj_polys = []
        for h in self.trajectory:
            poly = Polygon(h3.cell_to_boundary(h))
            traj_polys.append({"h3": h, "geometry": poly})
        traj_gdf = gpd.GeoDataFrame(traj_polys, crs="EPSG:4326").to_crs(epsg=3857)

        # Current position
        current_poly = Polygon(h3.cell_to_boundary(self.current_h3))
        current_gdf = gpd.GeoDataFrame(
            [{"geometry": current_poly}], crs="EPSG:4326"
        ).to_crs(epsg=3857)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 10))
        ocean_gdf.plot(ax=ax, facecolor="lightblue", edgecolor="gray", linewidth=0.2)
        traj_gdf.plot(ax=ax, facecolor="orange", edgecolor="black", linewidth=0.5)
        current_gdf.plot(ax=ax, facecolor="red", edgecolor="black")
        ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite)
        ax.set_title(f"Step {self.step_count} | Time: {self.current_time}")

        plt.tight_layout()
        plt.savefig(f"{render_dir}/24M_step_{self.step_count:05d}.png")
        plt.close(fig)  # Prevent memory leaks

    def buffer_selected_info(self, info: dict) -> None:
        """
        Buffers selected information from the info dictionary to be written to a CSV file.
        """
        selected_info = {
            "distance_to_goal": info["distance_to_goal"],
            "progress": info["progress"],
            "override_reward": info["override_reward"],
            "wind_penalty": info["wind_penalty"],
            "fuel_penalty": info["fuel_penalty"],
            "speed": info["speed"],
            "fuel_consumed": info["fuel_consumed"],
            "raw_reward": info["raw_reward"],
            "current_time": info["current_time"],
            "wind_direction": info["wind_direction"],
            "move_direction": info["move_direction"],
            "angle_diff": info["angle_diff"],
            "eta_penalty": info["eta_penalty"],
        }

        self.csv_buffer.append(selected_info)

        if len(self.csv_buffer) >= self.csv_buffer_size:
            self.flush_buffer_to_csv()

    def flush_buffer_to_csv(self) -> None:
        """
        Writes the buffered step information to a CSV file.
        """
        if not self.csv_buffer:
            return

        write_header = not self.csv_header_written
        with open(self.csv_file_path, mode="a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.csv_buffer[0].keys())
            if write_header:
                writer.writeheader()
                self.csv_header_written = True
            writer.writerows(self.csv_buffer)
        self.csv_buffer.clear()


class MariNavWithHistory(gym.Wrapper):
    """
    A Gym wrapper to append a history of observations to the current observation.
    This is useful for models like MinGRU/RNN that process sequences.
    """

    def __init__(self, env: gym.Env, history_len: int = DEFAULT_HISTORY_LEN):
        super().__init__(env)
        self.history_len = history_len
        self.obs_buffer: list[np.ndarray] = []

        obs_space = env.observation_space

        # Calculate history parts
        history_low = np.tile(obs_space.low[:-4], history_len)
        history_high = np.tile(obs_space.high[:-4], history_len)

        # Full low and high
        full_low = np.concatenate([history_low, obs_space.low[-4:]])
        full_high = np.concatenate([history_high, obs_space.high[-4:]])

        self.observation_space = spaces.Box(
            low=full_low,
            high=full_high,
            shape=full_low.shape,
            dtype=np.float32,
        )

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        """
        Resets the environment and initializes the observation buffer with
        the first observation (excluding coords) repeated `history_len` times.
        """
        obs, info = self.env.reset(**kwargs)

        # Store only the truncated obs (exclude last 4 coords)
        self.obs_buffer = [obs[:-4]] * self.history_len

        # Concatenate history with the last 4 fixed coords
        stacked_obs = np.concatenate(
            [np.concatenate(self.obs_buffer, axis=0), obs[-4:]], axis=0
        )

        return stacked_obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Takes a step in the environment, appends the new observation to the buffer,
        and returns the concatenated history as the observation.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs_buffer.append(obs[:-4])
        self.obs_buffer = self.obs_buffer[
            -self.history_len :
        ]  # Keep only the latest `history_len` observations
        stacked_obs = np.concatenate(
            [np.concatenate(self.obs_buffer, axis=0), obs[-4:]], axis=0
        )  # Append the last 4 elements (start_lat,start_lon,goal_lat,goal_lon)
        return stacked_obs, reward, terminated, truncated, info
