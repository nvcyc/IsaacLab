# Copyright (c) 2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch


class EpisodeData:
    """Class to store episode data."""

    def __init__(self) -> None:
        """Initializes the modifier class.

        Args:
            cfg: Configuration parameters.
            data_dim: The dimensions of the data to be modified. First element is the batch size
                which usually corresponds to number of environments in the simulation.
            device: The device to run the modifier on.
        """
        self._data = dict()
        self._next_action_index = 0
        self._seed = None
        self._env_id = None
        self._success = None

    @property
    def data(self):
        """Returns the episode data."""
        return self._data

    @data.setter
    def data(self, data: dict):
        """Set the episode data."""
        self._data = data

    @property
    def seed(self):
        """Returns the random number generator seed."""
        return self._seed

    @seed.setter
    def seed(self, seed: int):
        """Set the random number generator seed."""
        self._seed = seed

    @property
    def env_id(self):
        """Returns the environment ID."""
        return self._env_id

    @env_id.setter
    def env_id(self, env_id: int):
        """Set the environment ID."""
        self._env_id = env_id

    @property
    def success(self):
        """Returns the success value."""
        return self._success

    @success.setter
    def success(self, success: bool):
        """Set the success value."""
        self._success = success

    def is_empty(self):
        """Check if the episode data is empty."""
        return not bool(self._data)

    def add(self, key: str, value: torch.Tensor | dict):
        """Add a key-value pair to the dataset.

        The key can be nested by using the "/" character.
        For example: "obs/joint_pos".

        Args:
            key: The key name.
            value: The corresponding value of tensor type or of dict type.
        """
        # check datatype
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                self.add(f"{key}/{sub_key}", sub_value)
            return

        sub_keys = key.split("/")
        current_dataset_pointer = self._data
        for sub_key_index in range(len(sub_keys)):
            if sub_key_index == len(sub_keys) - 1:
                # Add value to the final dict layer
                if sub_keys[sub_key_index] not in current_dataset_pointer:
                    current_dataset_pointer[sub_keys[sub_key_index]] = value.unsqueeze(0)
                else:
                    current_dataset_pointer[sub_keys[sub_key_index]] = torch.cat(
                        (current_dataset_pointer[sub_keys[sub_key_index]], value.unsqueeze(0))
                    )
                break
            # key index
            if sub_keys[sub_key_index] not in current_dataset_pointer:
                current_dataset_pointer[sub_keys[sub_key_index]] = dict()
            current_dataset_pointer = current_dataset_pointer[sub_keys[sub_key_index]]

    def get_initial_state(self) -> torch.Tensor | None:
        """Get the initial state from the dataset."""
        if "initial_state" not in self._data:
            return None
        return self._data["initial_state"]

    def get_next_action(self) -> torch.Tensor | None:
        """Get the next action from the dataset."""
        if "actions" not in self._data:
            return None
        if self._next_action_index >= len(self._data["actions"]):
            return None
        action = self._data["actions"][self._next_action_index]
        self._next_action_index += 1
        return action
