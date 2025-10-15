import json as j
import random as r
import copy as c
from pathlib import Path

try:
    from controllers.brain import Brain
except ModuleNotFoundError:
    from brain import Brain


def generate_genome_id(prefix: str = "g") -> str:
    """Return a short random genome id like 'g_1a2b3c4d'. Uses 8 hex chars."""
    return f"{prefix}_{r.getrandbits(32):08x}"


def clamp(value, low, high):
    """Clamp *value* to the inclusive range [low, high]."""
    return max(low, min(value, high))


# Resolve paths relative to this file
ROOT_DIR = Path(__file__).resolve().parent.parent
GENOME_PATH = ROOT_DIR / "genomes" / "genome.json"
BRAIN_CONFIG_PATH = ROOT_DIR / "brains" / "brain.json"

with open(GENOME_PATH, "r") as file:
    base_genome  = j.load(file)

# Calculate weight count once at module load
brain_instance = Brain(str(BRAIN_CONFIG_PATH))
BRAIN_WEIGHT_COUNT = brain_instance.count_weights()

def genome_generator():
    genome = c.deepcopy(base_genome)
    genome["id"] = generate_genome_id()
    
    # Initialize brain weights with random values
    genome["brain_weights"] = [r.uniform(-0.5, 0.5) for _ in range(BRAIN_WEIGHT_COUNT)]
    
    for trait, info in genome["traits"].items():
        if "min" in info and "max" in info:
            info["value"] = r.uniform(info["min"], info["max"])

    return genome

def mutate_genome(genome):
    new_genome = c.deepcopy(genome)
    new_genome["id"] = generate_genome_id()
    for trait, info in new_genome["traits"].items():
        random_a = r.randint(1, 10)
        random_b = r.randint(1, 10)

        if random_a == random_b:
            mutation_factor = r.uniform(0.85, 1.02)
            mutated_value = info["value"] * mutation_factor

            if "min" in info and "max" in info:
                mutated_value = clamp(mutated_value, info["min"], info["max"])

            info["value"] = mutated_value

    for weight in range(len(new_genome["brain_weights"])):
        random_a = r.randint(1, 10)
        random_b = r.randint(1, 10)

        if random_a == random_b:
            new_genome["brain_weights"][weight] = (new_genome["brain_weights"][weight] 
                                                   * r.uniform(0.85, 1.02))
