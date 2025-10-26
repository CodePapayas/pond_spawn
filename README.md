# pond_spawn
A neural network-based evolution simulator where agents compete for food and reproduce in a 10x10 grid environment.

## Overview
This project simulates artificial life using neural networks and genetic algorithms. Agents with unique genomes make decisions using their "brains" (neural networks) to survive, eat, and reproduce. Over time, natural selection should favor beneficial traits and behaviors.

## Architecture

### Core Components

#### 1. **Brain (`controllers/brain.py`)** ✅ COMPLETE
- PyTorch neural network built from JSON configuration
- Loads weights from genome at initialization
- Winner-takes-all output: 4 neurons determine agent actions
- Network architecture: 5 inputs → 8 hidden (tanh) → 8 hidden (tanh) → 4 outputs
- 156 total trainable parameters

**Inputs (5):**
- Normalized energy level (0-1)
- Food at current position (0-1)
- Nearby agent count (0-1)
- Biome visibility (0-1)
- Biome movement speed (0-1)

**Outputs (4):**
- ACTION_MOVE: Move forward in current heading
- ACTION_TURN: Turn 90° clockwise
- ACTION_EAT: Consume food at current tile
- ACTION_REPRODUCE: Create mutated offspring (25% energy cost)

#### 2. **Genome (`controllers/genome.py`)** ✅ COMPLETE
- Genetic blueprint defining agent traits and brain weights
- Class-based implementation with generation and mutation
- 7 mutable traits: vision, speed, metabolism, nutrition, energy capacity, mutation rate, reproduction threshold
- 156 brain weights stored as flat list
- Mutation: 10% chance per trait/weight, factor 0.85-1.02

#### 3. **Biome (`controllers/landscape.py`)** ✅ COMPLETE
- Environmental tile properties
- Features: movement_speed (0.8-1.05), visibility (0.25-1.0), food_units (0-3)
- Randomly generated from base template
- Affects agent perception and movement costs

#### 4. **Agent (`controllers/agent.py`)** ✅ COMPLETE
- Individual organism with genome, brain, position, energy, age, heading
- Perception: Gathers normalized sensory data from environment
- Decision: Brain outputs winner-takes-all action selection
- Actions: Move, turn, eat, reproduce
- Metabolism: Base cost + movement cost + action costs
- Reproduction: Costs 25% energy, creates mutated offspring in adjacent tile

#### 5. **Environment (`simulation.py`)** ✅ COMPLETE
- 10x10 grid of biomes
- 100 starting agents, 150 food units (avg 1.5 per tile, max 3 per tile)
- Manages agent updates, reproduction, death
- Tracks statistics: population, food, average energy
- Helper methods: `get_biome()`, `get_agents_at()`, `count_agents_in_range()`

### Test Suite (`tests/`)  ✅ COMPLETE
- **test_brain.py**: 26 tests covering initialization, forward pass, weight counting, genome loading
- **test_genome.py**: 39 tests covering generation, mutation, serialization, integration
- **65 tests total, all passing** ✅

## Current Status

### ✅ Completed
- [x] Brain neural network with JSON configuration
- [x] Genome class with traits and brain weights
- [x] Genome mutation system
- [x] Biome environmental tiles
- [x] Agent perception, decision-making, and actions
- [x] Winner-takes-all action selection
- [x] Movement system with heading (N/E/S/W)
- [x] Energy system with metabolism
- [x] Food consumption mechanics
- [x] Reproduction with mutation
- [x] Environment grid and simulation loop
- [x] Comprehensive test coverage
- [x] Agent age tracking

### 🚧 In Progress / TODO

#### High Priority
- [ ] **Multi-agent eating priority**: Implement speed-based eating order when multiple agents on same tile
- [ ] **Food respawn system**: Add periodic food generation to sustain population
- [ ] **Energy capacity limits**: Enforce max energy from genome trait
- [ ] **Visualization**: Create visual representation of grid state (Pygame/Matplotlib)
- [ ] **Data logging**: Track evolution metrics (avg traits, population, survival time)

#### Medium Priority
- [ ] **Biome effects**: Apply movement_speed and visibility modifiers to agent actions
- [ ] **Vision range**: Implement proper vision-based perception using genome vision trait
- [ ] **Reproduction threshold**: Use genome `clone_energy_threshold` instead of fixed 25%
- [ ] **Daily nutrition**: Implement `daily_nutrition_minimum` survival requirement
- [ ] **Mutation rate**: Use genome `mutation_rate` trait in mutation logic
- [ ] **Performance optimization**: Spatial indexing for large populations

#### Low Priority / Future Enhancements
- [ ] **Save/load simulation state**: Serialize entire environment to JSON
- [ ] **Multiple species**: Color-coded lineages or species divergence
- [ ] **Advanced biomes**: Add water/land, predator zones, etc.
- [ ] **Sexual reproduction**: Crossover between two parents
- [ ] **Learning**: Online weight updates based on success
- [ ] **Visualization dashboard**: Real-time charts and heatmaps
- [ ] **Configuration UI**: Adjust simulation parameters without code changes

## Running the Simulation

### Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install torch pytest pytest-cov ruff
```

### Run Tests
```bash
pytest                          # Run all tests
pytest tests/test_brain.py -v   # Test brain only
pytest --cov=controllers        # With coverage report
```

### Run Simulation
```bash
python simulation.py
```

## File Structure
```
pond_spawn/
├── controllers/
│   ├── agent.py      # Agent behavior and decision-making
│   ├── brain.py      # Neural network implementation
│   ├── genome.py     # Genetic blueprint and mutation
│   └── landscape.py  # Biome environmental tiles
├── brains/
│   └── brain.json    # Neural network architecture config
├── biomes/
│   └── biome.json    # Biome feature definitions
├── genomes/
│   └── genome.json   # Genome trait definitions
├── tests/
│   ├── test_brain.py
│   ├── test_genome.py
│   └── README.md
├── simulation.py     # Main simulation environment
└── README.md
```

## Configuration Files

### `brains/brain.json`
Defines neural network architecture (layers, sizes, activations)

### `genomes/genome.json`
Defines inheritable traits with min/max ranges

### `biomes/biome.json`
Defines environmental features and food availability

# Development Log

## October 25th, 2025
- ✅ Completed agent.py with winner-takes-all action selection
- ✅ Implemented all 4 actions: MOVE, TURN, EAT, REPRODUCE
- ✅ Added heading-based movement system (N/E/S/W)
- ✅ Integrated agent with environment simulation loop
- ✅ Added `count_agents_in_range()` for vision-based perception
- ✅ Agent age tracking per tick
- 📊 Simulation ready for first full runs

## October 22nd, 2025
- ✅ Completed landscape.py with Biome class
- ✅ Created simulation.py with Environment and grid management
- ✅ Added food distribution and agent spawning
- ✅ Built simulation statistics tracking

## October 15th, 2025
- ✅ Added load_from_genome() to brain.py
- ✅ Refactored genome into Genome class (matching Brain pattern)
- ✅ Added comprehensive test suite (65 tests, all passing)
- ✅ Integrated ruff for code formatting
- 🎯 Brain and genome components fully tested and complete

## Next Session Goals
1. Test run simulation with logging to identify issues
2. Implement multi-agent eating priority (speed-based)
3. Add food respawn mechanics
4. Create basic visualization (print grid with colors/symbols)
5. Track and plot evolution metrics over time