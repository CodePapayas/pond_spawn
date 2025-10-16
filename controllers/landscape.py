import json
import random
import copy
from pathlib import Path

TERRAIN_RANGE = 10

ROOT_DIR = Path(__file__).resolve().parent.parent
BIOME_PATH = ROOT_DIR / "biomes" / "biomes.json"

def generate_biome_id(prefix: str = "b") -> str:
    """Return a short random genome id like 'g_1a2b3c4d'. Uses 8 hex chars."""
    return f"{prefix}_{r.getrandbits(32):08x}"

def clamp(value, low, high):
    """Clamp *value* to the inclusive range [low, high]."""
    return max(low, min(value, high))

