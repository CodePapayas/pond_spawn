import copy as c
import json
import random as r
from pathlib import Path

TERRAIN_RANGE = 10

ROOT_DIR = Path(__file__).resolve().parent.parent
BIOME_PATH = ROOT_DIR / "biomes" / "biome.json"


def generate_biome_id(prefix: str = "b") -> str:
    """Return a short random biome id like 'b_1a2b3c4d'. Uses 8 hex chars."""
    return f"{prefix}_{r.getrandbits(32):08x}"


def clamp(value, low, high):
    """Clamp *value* to the inclusive range [low, high]."""
    return max(low, min(value, high))


class Biome:
    """
    Represents a biome found on a single tile in a given simulation.

    A biome defines the environmental properties of a tile including:
    - Movement speed modifier (affects how fast organisms can move)
    - Visibility (affects how far organisms can see)
    - Food units (how much food is available in this tile)

    Biomes can be randomly generated from a base template with randomized
    feature values within defined ranges.
    """

    # Class-level base biome template loaded once
    _base_biome = None

    @classmethod
    def _load_base_data(cls):
        """Load base biome template (done once)."""
        if cls._base_biome is None:
            with open(BIOME_PATH) as file:
                cls._base_biome = json.load(file)

    def __init__(self, biome_dict=None):
        """
        Initialize a Biome instance.

        Args:
            biome_dict (dict, optional): If provided, initializes the biome with
                this data. If None, the biome must be populated using generate()
                or other methods.
        """
        self._load_base_data()

        if biome_dict:
            self.id = biome_dict["id"]
            self.features = biome_dict["features"]
        else:
            self.id = None
            self.features = {}

    def generate(self):
        """
        Generate a new random biome with randomized feature values.

        This method creates a new biome by:
        1. Deep copying the base biome template
        2. Assigning a unique random ID
        3. Randomizing feature values within their defined min/max ranges
        4. Randomly selecting food_units from available options

        Returns:
            Biome: Returns self for method chaining
        """
        biome_data = c.deepcopy(self._base_biome)
        self.id = generate_biome_id()
        self.features = biome_data["features"]

        # Randomize features with min/max ranges
        for feature_name, feature_info in self.features.items():
            if isinstance(feature_info, dict) and "min" in feature_info and "max" in feature_info:
                self.features[feature_name]["value"] = r.uniform(
                    feature_info["min"], feature_info["max"]
                )
            elif isinstance(feature_info, list):
                # For list features like food_units, select a random value
                self.features[feature_name] = r.choice(feature_info)

        return self

    def to_dict(self):
        """
        Convert the biome to a dictionary representation.

        Returns:
            dict: Dictionary containing id and features
        """
        return {
            "id": self.id,
            "features": self.features,
        }

    @classmethod
    def from_dict(cls, biome_dict):
        """
        Create a Biome instance from a dictionary.

        Args:
            biome_dict (dict): Dictionary containing biome data

        Returns:
            Biome: New Biome instance with the provided data
        """
        return cls(biome_dict)

    def get_movement_speed(self):
        """
        Get the movement speed modifier for this biome.

        Returns:
            float: Movement speed value, or None if not set
        """
        if isinstance(self.features.get("movement_speed"), dict):
            return self.features["movement_speed"].get("value")
        return self.features.get("movement_speed")

    def get_fertility(self):
        """
        Get the movement speed modifier for this biome.

        Returns:
            float: Movement speed value, or None if not set
        """
        if isinstance(self.features.get("fertility"), dict):
            return self.features["fertility"].get("value")
        return self.features.get("fertility")

    def get_visibility(self):
        """
        Get the visibility value for this biome.

        Returns:
            float: Visibility value, or None if not set
        """
        if isinstance(self.features.get("visibility"), dict):
            return self.features["visibility"].get("value")
        return self.features.get("visibility")

    def get_food_units(self):
        """
        Get the food units available in this biome.

        Returns:
            int: Number of food units
        """
        key_check = "food_units"

        if key_check in self.features:
            if self.features["food_units"] == None:
                self.features["food_units"] = 0
        else:
            self.features["food_units"] = 0
            
        return self.features.get("food_units")


if __name__ == "__main__":
    pass
