import json as j
import random as r
import copy as c
from pathlib import Path

try:
    from controllers.brain import Brain
except ModuleNotFoundError:
    from brain import Brain


# Resolve paths relative to this file
ROOT_DIR = Path(__file__).resolve().parent.parent
GENOME_PATH = ROOT_DIR / "genomes" / "genome.json"
BRAIN_CONFIG_PATH = ROOT_DIR / "brains" / "brain.json"


def generate_genome_id(prefix: str = "g") -> str:
    """Return a short random genome id like 'g_1a2b3c4d'. Uses 8 hex chars."""
    return f"{prefix}_{r.getrandbits(32):08x}"


def clamp(value, low, high):
    """Clamp *value* to the inclusive range [low, high]."""
    return max(low, min(value, high))


class Genome:
    """
    Represents a genetic blueprint for an organism, including traits and neural network weights.

    This class encapsulates all genetic information for an organism including:
    - Unique identifier
    - Trait values (vision, speed, metabolism, etc.)
    - Neural network weights for the brain

    The genome can be randomly generated, mutated, and used to configure a Brain instance.
    """

    # Class-level constants loaded once
    _base_genome = None
    _brain_weight_count = None

    @classmethod
    def _load_base_data(cls):
        """Load base genome template and calculate brain weight count (done once)."""
        if cls._base_genome is None:
            with open(GENOME_PATH, "r") as file:
                cls._base_genome = j.load(file)
            brain_instance = Brain(str(BRAIN_CONFIG_PATH))
            cls._brain_weight_count = brain_instance.count_weights()

    def __init__(self, genome_dict=None):
        """
        Initialize a Genome instance.

        Args:
            genome_dict (dict, optional): If provided, initializes the genome with
                this data. If None, the genome must be populated using generate()
                or other methods.
        """
        self._load_base_data()

        if genome_dict:
            self.id = genome_dict["id"]
            self.traits = genome_dict["traits"]
            self.brain_weights = genome_dict["brain_weights"]
        else:
            self.id = None
            self.traits = {}
            self.brain_weights = []

    def generate(self):
        """
        Generate a new random genome with random traits and brain weights.

        This method creates a completely new genome by:
        1. Deep copying the base genome template
        2. Assigning a unique random ID
        3. Initializing brain weights with random values in range [-0.5, 0.5]
        4. Randomizing all trait values within their defined min/max ranges

        Returns:
            Genome: Returns self for method chaining
        """
        genome_data = c.deepcopy(self._base_genome)
        self.id = generate_genome_id()

        # Initialize brain weights with random values
        self.brain_weights = [
            r.uniform(-0.5, 0.5) for _ in range(self._brain_weight_count)
        ]

        # Initialize traits
        self.traits = genome_data["traits"]
        for trait, info in self.traits.items():
            if "min" in info and "max" in info:
                info["value"] = r.uniform(info["min"], info["max"])

        return self

    def mutate(self):
        """
        Create a mutated copy of this genome.

        This method produces a new Genome instance with:
        1. A new unique ID
        2. Traits that have a 10% chance (1 in 10) of being mutated by a factor
           between 0.85 and 1.02, clamped to their min/max ranges
        3. Brain weights that have a 10% chance (1 in 10) of being mutated by
           a factor between 0.85 and 1.02

        Returns:
            Genome: A new Genome instance with mutated values
        """
        new_genome = Genome()
        new_genome.id = generate_genome_id()
        new_genome.traits = c.deepcopy(self.traits)
        new_genome.brain_weights = c.deepcopy(self.brain_weights)

        # Mutate traits
        for trait, info in new_genome.traits.items():
            random_a = r.randint(1, 10)
            random_b = r.randint(1, 10)

            if random_a == random_b:
                mutation_factor = r.uniform(0.85, 1.02)
                mutated_value = info["value"] * mutation_factor

                if "min" in info and "max" in info:
                    mutated_value = clamp(mutated_value, info["min"], info["max"])

                info["value"] = mutated_value

        # Mutate brain weights
        for i in range(len(new_genome.brain_weights)):
            random_a = r.randint(1, 10)
            random_b = r.randint(1, 10)

            if random_a == random_b:
                new_genome.brain_weights[i] = new_genome.brain_weights[i] * r.uniform(
                    0.85, 1.02
                )

        return new_genome

    def to_dict(self):
        """
        Convert the genome to a dictionary representation.

        Returns:
            dict: Dictionary containing id, traits, and brain_weights
        """
        return {
            "id": self.id,
            "traits": self.traits,
            "brain_weights": self.brain_weights,
        }

    @classmethod
    def from_dict(cls, genome_dict):
        """
        Create a Genome instance from a dictionary.

        Args:
            genome_dict (dict): Dictionary containing genome data

        Returns:
            Genome: New Genome instance with the provided data
        """
        return cls(genome_dict)


if __name__ == "__main__":
    pass
