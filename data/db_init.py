"""
db_init.py

Creates a database for storing simulation information across multiple runs.
Can be run multiple times without losing existing data; Objects all use IF NOT EXISTS.

Usage:
    python db_init.py /path/to/sim.db
    python db_init.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

-- Sim config and metadata
CREATE TABLE IF NOT EXISTS sim (
    id INTEGER PRIMARY KEY DEFAULT 1,
    grid_size INTEGER NOT NULL,
    initial_pop INTEGER NOT NULL,
    initial_ticks INTEGER NOT NULL,
    final_ticks INTEGER NOT NULL,
    food_resupply INTEGER NOT NULL
);

-- Static biome metadata; Never changes, stored at initialization
CREATE TABLE IF NOT EXISTS biomes (
    x INTEGER NOT NULL,
    y INTEGER NOT NULL,
    biome_id TEXT NOT NULL,
    movement_speed REAL NOT NULL,
    visibility REAL NOT NULL,
    fertility REAL NOT NULL,
    initial_food_units INTEGER NOT NULL,
    PRIMARY KEY (x, y)
);

-- Genome traits; Snapshot captured once at birth
CREATE TABLE IF NOT EXISTS (
    agent_id TEXT NOT NULL,
    vision REAL NOT NULL,
    speed REAL NOT NULL,
    metabolism REAL NOT NULL,
    daily_nutrition_minimum REAL NOT NULL,
    energy_capacity REAL NOT NULL,
    mutation_rate REAL NOT NULL,
    clone_energy_threshold REAL NOT NULL,
    reproduction_cost REAL NOT NULL,
    intelligence REAL NOT NULL,
    attack REAL NOT NULL,
    defense REAL NOT NULL,
    aggression REAL NOT NULL,
    FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
);

-- Brain weights, stored as blob (246 floats as binary)
CREATE TABLE IF NOT EXISTS brain_weights (
    agent_id TEXT PRIMARY KEY,
    weights BLOB NOT NULL,  -- struct.pack('246f', *weights)
    FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
);

-- Per-tick population-level statistics
CREATE TABLE IF NOT EXISTS tick_stats (
    tick INTEGER PRIMARY KEY,
    alive_agents INTEGER NOT NULL,
    total_food INTEGER NOT NULL,
    avg_energy REAL NOT NULL,
    births_this_tick INTEGER NOT NULL,
    deaths_this_tick INTEGER NOT NULL,
    deaths_starvation INTEGER NOT NULL,
    deaths_old_age INTEGER NOT NULL
);

-- Per-tick trait distribution statistics
CREATE TABLE IF NOT EXISTS tick_trait_stats (
    tick INTEGER NOT NULL,
    trait_name TEXT NOT NULL,
    mean REAL NOT NULL,
    std REAL NOT NULL,
    min REAL NOT NULL,
    max REAL NOT NULL,
    median REAL NOT NULL,
    PRIMARY KEY (tick, trait_name)
);

-- Per-tick action frequencies
CREATE TABLE IF NOT EXISTS tick_actions (
    tick INTEGER NOT NULL,
    action_move INTEGER NOT NULL DEFAULT 0,
    action_turn INTEGER NOT NULL DEFAULT 0,
    action_eat INTEGER NOT NULL DEFAULT 0,
    action_reproduce INTEGER NOT NULL DEFAULT 0,
    action_sleep INTEGER NOT NULL DEFAULT 0,
    action_nothing INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (tick)
);

-- Per-tick agent state (energy, position, age)
-- This is the large table: 30k agents × 1000 ticks = 30M rows
CREATE TABLE IF NOT EXISTS agent_ticks (
    tick INTEGER NOT NULL,
    agent_id TEXT NOT NULL,
    energy REAL NOT NULL,
    x INTEGER NOT NULL,
    y INTEGER NOT NULL,
    heading INTEGER NOT NULL,
    age INTEGER NOT NULL,
    action_taken INTEGER NOT NULL,  -- 0-5 action constant
    PRIMARY KEY (tick, agent_id),
    FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
);

-- Observer cluster assignments (populated by ML observer)
CREATE TABLE IF NOT EXISTS cluster_assignments (
    tick INTEGER NOT NULL,
    agent_id TEXT NOT NULL,
    cluster_type TEXT NOT NULL,  -- 'genome', 'brain', 'combined'
    cluster_id INTEGER NOT NULL,
    confidence REAL,  -- optional, if clustering method provides it
    PRIMARY KEY (tick, agent_id, cluster_type),
    FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
);

-- Indices for common query patterns
CREATE INDEX IF NOT EXISTS idx_agents_parent ON agents(parent_id);
CREATE INDEX IF NOT EXISTS idx_agents_birth_tick ON agents(birth_tick);
CREATE INDEX IF NOT EXISTS idx_agents_death_tick ON agents(death_tick);
CREATE INDEX IF NOT EXISTS idx_agent_ticks_agent ON agent_ticks(agent_id);
CREATE INDEX IF NOT EXISTS idx_cluster_tick ON cluster_assignments(tick);
CREATE INDEX IF NOT EXISTS idx_cluster_agent ON cluster_assignments(agent_id);
"""


def init_db(db_path: str | Path) -> None:
    """
    Initialize SQLite db schema if it doesn't already exist.

    :param db_path: Description
    :type db_path: str | Path
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")

        conn.executescript(SCHEMA_SQL)
        conn.commit()


def main(argv: list[str]) -> int:
    """
    CLI entrypoint.

    :param argv: Description
    :type argv: list[str]
    :return: Description
    :rtype: int
    """
    db_path = argv[0] if argv else "simulation.db"
    init_db(db_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
