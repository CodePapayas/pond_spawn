# pond_spawn

[![Coverage](https://raw.githubusercontent.com/codepapayas/pond_spawn/badges/badges/coverage.svg)](https://github.com/codepapayas/pond_spawn)
[![CI](https://github.com/codepapayas/pond_spawn/actions/workflows/main.yml/badge.svg)](https://github.com/codepapayas/pond_spawn/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Language](https://img.shields.io/badge/language-python-blue)

### With visuals
![Simulation Visualization GIF](assets/gifs/visual_sim.gif)

### Without visuals (print to console)
![Simulation GIF](assets/gifs/sim_gif_looped.gif)

*************************

Tossing out the old AI README and writing my own. This will serve double-duty as a devlog of sorts as I stumble my way through this.

## RUNNING THE SIMULATION
*************************

```bash
# Basic usage
python -m cli.cli_sim_starter

# Custom parameters
python -m cli.cli_sim_starter --population 100 --steps 500 --grid-size 20

# Fast run without visuals
python -m cli.cli_sim_starter --no-visual --steps 5000

# Get help
python -m cli.cli_sim_starter --help
```

### Pygame Visualizer
```bash
# Basic usage
python -m cli.pygame_visualizer

# Custom parameters
python -m cli.pygame_visualizer --grid-size 20 --population 200 --cell-size 30 --fps 15

# Controls:
#   Space - Pause/Resume
#   Escape - Quit (generates stats graph on exit)
```

Place your agent sprite at `assets/sprites/callumV1.png`. The visualizer will automatically load it and rotate based on heading. Falls back to colored circles if no sprite is found.

# OVERVIEW
*************************
<h2>This is my attempt to understand a: neural networks, and b: artificial life simulations.</h2>


# OBSERVATIONS
*************************
<ul>
    <s><li>The Callums demonstrate interesting behavior with the current genomic and environmental settings. A genome that favors conserving energy at the expense of procreation predictably results a population collapse; A genome that favors reproduction tends to have about 33% of their total energy level on average across the population.</li>
    <li>The Callums are still clustering at the edges of the map, so I'm wondering if I should add the ability to loop around if you hit the edges. It also may be that the Callums are just stupid and need more options to choose from.</li>
    <li>Longer sims with more steps and larger populations make my pc cry.</li>
    <li>It turns out the feedforward nn is actually fairly efficient given it's size and lack of backpropagation. The problem, it seems, is my horribly inefficient sim loop logic.</li>
    <li>Drawing is much harder than I initially thought.</li>
    <li>These Callum's don't procreate enough, and I think it's due to environmental pressures being non-existant. Maybe seasons?</li></s>
</ul>

# DEV LOG
*************************
<h1>December 23rd, 2025</h1>
Updating after a while. I debated on abandoning this project and just starting anew with lessons learned, but I think I want to see this one through.

A lot of this isn't as connected as I thought it was. The death logic, for example, was not working. It turns out I had written the algorithm for assigning death ages incorrectly and I honestly can't remember what direction I was going with that so, ya know, just fixed that to actually work. I also remembered that tests are a thing and started updating/adding some.

TODO
<ul>
    <li>The tests need to be accurate and reflect the current state of the program. Solo work is hard; testing keeps us on task</li>
    <li>Make a test file for the environment. A lot of important functions live there now and there is a need to know if they actually work.</li>
    <li>Connect intelligence to the decision making algorithm; Create an actual action tied to it. I think it makes sense for it to dictate the rate at which an entity makes decisions in reaction to stimuli, so somehow it needs to dampen or enhance that function.</li>
</ul>

*************************

*************************
<h1>November 15th, 2025</h1>
Artificial life simulations are hard. We fixed the energy add function. It was boundless, which satisfied the Callum's but didn't fit the sim. Will write more and update TODO later.

TODO
<ul>
    <s><li>Add 4 more outputs to the brains output layer. This will allow us to make the Callums decisions more nuanced</li>
    <li>Expand the genome to have two new features: attack and defense</li>
    <li>Add logging to decisions and snapshots of state per tick so I can understan why the Callums do what they do</li></s>
</ul>

*************************

<h1>November 10th, 2025</h1>
Today was an adventure in futility. I second-guessed myself and rewrote the food resupply logic. I thought that I would make only the first supply of food be random and then just copy/paste resources when they get down to zero and also with a regular cadence. My thinking was that plants generally grow in the same place and around the same time, so maybe I could do that. What happened was the environment would get flooded with food but somehow the Calllums would still starve. As it turns out, I just didn't know what I was doing and my first attempt was good. Switched it back to random regen with some tweaks and we're semi-functional again.

More helpful changes are listed in the commit messages but the one I'm happiest with is the enhanced logic around reproduction and lifespan. I need to add a tracker for dead agents to make sure they're actually dying of old age, but it's a step in the right direction. I realize I was getting ahead of myself by wanting to add more outputs before I refined the 4 I have now. Currently this simulation is a little more complex, but the population keeps collapsing around 200 steps in so gotta figure that one out. No TODO today. Will update that later.

TODO
<ul>
    <s><li>Add 4 more outputs to the brains output layer. This will allow us to make the Callums decisions more nuanced</li>
    <li>Expand the genome to have two new features: attack and defense</li>
    <li>Add logging to decisions and snapshots of state per tick so I can understan why the Callums do what they do</li></s>
</ul>


*************************

<h1>November 1st, 2025</h1>
The Callums are making decisions that seem to support a stable, if small, population. I am now satisfied with this rough simulation and it's parameters. Snapshot: <img src="assets/11-1-grid.png">
TODO
<ul>
    <li>Add 4 more outputs to the brains output layer. This will allow us to make the Callums decisions more nuanced</li>
    <li>Expand the genome to have two new features: attack and defense</li>
    <li>Add logging to decisions and <s>snapshots of state per tick</s> so I can understan why the Callums do what they do</li>
</ul>

We also added a GitHub Action requiring Ruff linting before merge.

*************************

<h1>October 28th, 2025</h1>
I noticed that in each run the agents, now known as "Callums", were congregating on the very top of the grid. Literally row 0, all the lil guys just huddled up there. I adjusted the starting heading to be randomized instead of always facing North. The result was that the Callums now congregate on the entire perimeter of the grid, not just the top. They seem to particularly favor the corners. Example below. Apologies for the color scheme, I have a terrible eye for design.
<img src="assets/10-28-grid.png">
I also made the sim take a snapshot of the initial randomized grid and the final grid for comparison.
TODO:
<ul>
    <li>Add 4 more outputs to the brains output layer. This will allow us to make the Callums decisions more nuanced</li>
    <li>Expand the genome to have two new features: attack and defense</li>
    <li><s>Adjust the decision making function. Callums should probably decide to leave if the tile is too crowded or they're seeing aggression/food competition</s></li>
</ul>

*************************

<h1>October 26th, 2025</h1>
I tossed out the old readme (it was an ai stand-in made by Copilot) and decided to make my own. The agents brain is not choosing to 'Eat' enough, resulting in starvation, depopulation, and a HUGE accumulation of food. Thoughts on why this is below:
<ul>
    <li><s>Too much food, obviously</s></li>
    <li><s>The brain is too simple. There should be threshold triggers that are checked before the brain is called</s></li>
</ul>

With the above fixed I added a GIF to the README and am going to bed.