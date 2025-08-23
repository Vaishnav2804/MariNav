from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn

# Local application/specific imports
from minGRU_pytorch import minGRU

DEFAULT_HISTORY_LEN = 8


class MinGRUPolicyBackbone(nn.Module):
    """
    Neural network backbone using a minGRU for sequence processing in a policy.

    Args:
        input_dim (int): Dimension of the input observation for each time step.
        gru_dim (int): Dimension of the GRU's hidden state and output.
        expansion_factor (float): Factor for expanding the GRU's internal dimension.
    """

    def __init__(
        self, input_dim: int = 4, gru_dim: int = 128, expansion_factor: float = 1.5
    ):
        super().__init__()
        self.input_dim = input_dim
        self.gru_dim = gru_dim

        # Project environment observation to GRU dimension
        self.embedding = nn.Linear(input_dim, gru_dim)
        self.min_gru = minGRU(dim=gru_dim, expansion_factor=expansion_factor)
        # Optional; add nn.Linear if you want to down-project
        self.output_layer = nn.Identity()

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """
        Processes a sequence of observations through the minGRU backbone.

        Args:
            obs_seq (torch.Tensor): Tensor of shape `[batch_size, sequence_len, input_dim]`.

        Returns:
            torch.Tensor: Final feature vector for policy/value network, shape `[batch_size, gru_dim]`.
        """
        # [Batch_size, Sequence_len, Input_dim] -> [Batch_size, Sequence_len, GRU_dim]
        x = self.embedding(obs_seq)
        # Process sequence through minGRU -> [Batch_size, Sequence_len, GRU_dim]
        x = self.min_gru(x)
        # Take the output from the last time step -> [Batch_size, GRU_dim]
        x = x[:, -1, :]
        return self.output_layer(x)


class MinGRUFeaturesExtractor(BaseFeaturesExtractor):
    """
    A custom features extractor for Stable Baselines3 that uses MinGRUPolicyBackbone
    to process sequential observations.

    It reshapes a flattened observation space from a `TankerEnvWithHistory` wrapper
    back into a sequence before feeding it to the MinGRU backbone.
    """

    def __init__(
        self, observation_space: spaces.Box, seq_len: int = DEFAULT_HISTORY_LEN
    ):
        # The `observation_space` here comes from `TankerEnvWithHistory`,
        # which flattens the observations into `[history_len * original_obs_dim]`.
        # So, `original_obs_dim` is `observation_space.shape[0] / seq_len`.

        if observation_space.shape[0] % seq_len != 0:
            raise ValueError(
                f"Observation space dimension ({observation_space.shape[0]}) "
                f"from wrapped environment must be divisible by seq_len ({seq_len})."
                f"Expected flattened shape: [batch_size, seq_len * original_obs_dim]"
            )

        feature_dim_per_step = observation_space.shape[0] // seq_len
        gru_output_dim = 128  # This should match the `gru_dim` in MinGRUPolicyBackbone

        # Initialize BaseFeaturesExtractor with the *output* dimension of this extractor
        super().__init__(observation_space, features_dim=gru_output_dim)

        self.seq_len = seq_len
        self.feature_dim_per_step = feature_dim_per_step

        # Instantiate the backbone with the correct `input_dim` and `gru_dim`
        self.backbone = MinGRUPolicyBackbone(
            input_dim=self.feature_dim_per_step, gru_dim=self.features_dim
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Processes the observations.

        Args:
            observations (torch.Tensor): Tensor of shape `[batch_size, flat_obs_dim]`
                                         where `flat_obs_dim = seq_len * feature_dim_per_step`.

        Returns:
            torch.Tensor: The extracted features from the MinGRU backbone.
        """
        batch_size = observations.shape[0]

        # Reshape the flattened observations:
        # From [batch_size, seq_len * feature_dim_per_step]
        # To   [batch_size, seq_len, feature_dim_per_step]
        reshaped_observations = observations.view(
            batch_size, self.seq_len, self.feature_dim_per_step
        )

        return self.backbone(reshaped_observations)
