import json
from pathlib import Path
from typing import Optional

from loguru import logger as logging


class StateManager:
    def __init__(self, state_mapping_dir: Path):
        # State-to-index mapping for tabular methods
        # Maps state tuples (hashable representation) to unique integer indices
        self.state_to_idx_dict: dict[tuple, int] = {}
        self.n_unique_states = 0  # Counter for unique states encountered

        # Directory for saving/loading state mappings (default: hrl_models/option_critic/state_mappings/)
        self.state_mapping_dir = state_mapping_dir
        self.state_mapping_dir.mkdir(exist_ok=True)
        self.current_level: Optional[int] = None

    def get_state_mapping_file(self, level: int) -> Path:
        """Get the file path for a level's state mapping

        Args:
            level (int): The level number

        Returns:
            Path: Path to the JSON file for this level's state mapping
        """
        return self.state_mapping_dir / f"level_{level}_state_mapping.json"

    def load_state_mapping(self, level: int) -> bool:
        """Load state-to-index mapping from file if it exists

        Args:
            level (int): The level number to load mapping for

        Returns:
            bool: True if mapping was loaded successfully, False otherwise
        """
        mapping_file = self.get_state_mapping_file(level)

        if not mapping_file.exists():
            return False

        try:
            with open(mapping_file, "r") as f:
                data = json.load(f)

            # Convert list of [state_list, index] pairs back to dictionary with tuple keys
            # JSON doesn't support tuple keys, so we stored as list of pairs
            state_mapping_list = data["state_to_idx_dict"]
            self.state_to_idx_dict = {tuple(k): v for k, v in state_mapping_list}
            self.n_unique_states = data["n_unique_states"]

            logging.info(
                f"Loaded state mapping for level {level}: {self.n_unique_states} unique states"
            )
            return True

        except Exception as e:
            logging.warning(f"Failed to load state mapping for level {level}: {e}")
            return False

    def save_state_mapping(self, level: int) -> bool:
        """Save state-to-index mapping to file

        Args:
            level (int): The level number to save mapping for

        Returns:
            bool: True if mapping was saved successfully, False otherwise
        """
        mapping_file = self.get_state_mapping_file(level)
        print("mapping_file: ", mapping_file)

        try:
            # Convert tuple keys to lists for JSON serialization
            # Store as list of [state_list, index] pairs since JSON keys must be strings
            state_mapping_list = [
                [list(k), v] for k, v in self.state_to_idx_dict.items()
            ]

            data = {
                "state_to_idx_dict": state_mapping_list,
                "n_unique_states": self.n_unique_states,
                "level": level,
            }

            with open(mapping_file, "w") as f:
                json.dump(data, f, indent=2)

            logging.info(
                f"Saved state mapping for level {level}: {self.n_unique_states} unique states to {mapping_file}"
            )
            return True

        except Exception as e:
            logging.error(f"Failed to save state mapping for level {level}: {e}")
            return False
