"""
Simple pygame visualization for pond_spawn simulation.

Usage:
    python -m cli.pygame_visualizer [options]
"""

import argparse
from pathlib import Path

import pygame

from cli.cli_sim_starter import plot_simulation_stats
from simulation import Environment

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DARK_GREEN = (34, 139, 34)
LIGHT_GREEN = (144, 238, 144)
BLUE = (65, 105, 225)
RED = (220, 20, 60)
YELLOW = (255, 215, 0)
GRAY = (128, 128, 128)


def get_food_color(food_units):
    """Return color based on food amount."""
    if food_units == 0:
        return (50, 50, 50)  # Dark gray - no food
    elif food_units <= 2:
        return (139, 69, 19)  # Brown - low food
    elif food_units <= 5:
        return DARK_GREEN
    else:
        return LIGHT_GREEN


def get_agent_color(agent):
    """Return color based on agent energy."""
    energy = agent.energy
    if energy > 70:
        return BLUE
    elif energy > 30:
        return YELLOW
    else:
        return RED


def load_agent_sprites(sprite_size):
    """
    Load and prepare agent sprites for all 4 headings.

    Returns a dict mapping heading (0-3) to rotated sprite,
    or None if sprite file not found.
    """
    sprite_path = Path(__file__).parent.parent / "assets" / "sprites" / "callumV1.png"

    if not sprite_path.exists():
        return None

    try:
        original = pygame.image.load(str(sprite_path)).convert_alpha()
        scaled = pygame.transform.scale(original, (sprite_size, sprite_size))

        # Create rotated versions for each heading
        # Assuming original sprite faces North (heading 0)
        sprites = {
            0: scaled,  # North - no rotation
            1: pygame.transform.rotate(scaled, -90),  # East - 90° clockwise
            2: pygame.transform.rotate(scaled, 180),  # South - 180°
            3: pygame.transform.rotate(scaled, 90),  # West - 90° counter-clockwise
        }
        return sprites
    except pygame.error:
        return None


def draw_agent_sprite(screen, sprites, agent, x, y, cell_size, sprite_size):
    """Draw an agent using sprite, centered in the cell."""
    sprite = sprites.get(agent.heading, sprites[0])
    # Center the sprite in the cell
    offset = (cell_size - sprite.get_width()) // 2
    screen.blit(sprite, (x + offset, y + offset))


def draw_heading_indicator(screen, x, y, heading, cell_size):
    """Draw a small triangle indicating agent heading."""
    center_x = x + cell_size // 2
    center_y = y + cell_size // 2
    size = cell_size // 4

    if heading == 0:  # North
        points = [
            (center_x, center_y - size),
            (center_x - size // 2, center_y),
            (center_x + size // 2, center_y),
        ]
    elif heading == 1:  # East
        points = [
            (center_x + size, center_y),
            (center_x, center_y - size // 2),
            (center_x, center_y + size // 2),
        ]
    elif heading == 2:  # South
        points = [
            (center_x, center_y + size),
            (center_x - size // 2, center_y),
            (center_x + size // 2, center_y),
        ]
    else:  # West
        points = [
            (center_x - size, center_y),
            (center_x, center_y - size // 2),
            (center_x, center_y + size // 2),
        ]

    pygame.draw.polygon(screen, WHITE, points)


def run_visualization(grid_size=12, population=100, cell_size=40, fps=10):
    """Run the pygame visualization."""
    pygame.init()

    # Calculate window size
    window_width = grid_size * cell_size
    window_height = grid_size * cell_size + 60  # Extra space for stats

    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Pond Spawn Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    # Load sprites (80% of cell size)
    sprite_size = int(cell_size * 0.8)
    agent_sprites = load_agent_sprites(sprite_size)
    if agent_sprites:
        print("Loaded agent sprites from assets/sprites/callumV1.png")
    else:
        print("No sprite found, using circle fallback")

    # Create environment
    env = Environment(grid_size=grid_size, num_agents=population)
    logged_stats = {}

    running = True
    paused = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_ESCAPE:
                    running = False

        if not paused:
            env.step()

        # Clear screen
        screen.fill(BLACK)

        # Draw grid
        for y in range(grid_size):
            for x in range(grid_size):
                biome = env.get_biome(x, y)
                food = biome.get_food_units() if biome else 0

                # Draw tile background based on food
                rect = pygame.Rect(x * cell_size, y * cell_size, cell_size - 1, cell_size - 1)
                pygame.draw.rect(screen, get_food_color(food), rect)

        # Draw agents
        for agent in env.agents:
            if agent.is_alive():
                ax, ay = agent.position
                pixel_x = ax * cell_size
                pixel_y = ay * cell_size

                if agent_sprites:
                    # Draw sprite
                    draw_agent_sprite(
                        screen, agent_sprites, agent, pixel_x, pixel_y, cell_size, sprite_size
                    )
                else:
                    # Fallback to circle
                    agent_color = get_agent_color(agent)
                    center_x = pixel_x + cell_size // 2
                    center_y = pixel_y + cell_size // 2
                    radius = cell_size // 3
                    pygame.draw.circle(screen, agent_color, (center_x, center_y), radius)
                    draw_heading_indicator(screen, pixel_x, pixel_y, agent.heading, cell_size)

        # Draw grid lines
        for i in range(grid_size + 1):
            pygame.draw.line(
                screen, GRAY, (i * cell_size, 0), (i * cell_size, grid_size * cell_size)
            )
            pygame.draw.line(
                screen, GRAY, (0, i * cell_size), (grid_size * cell_size, i * cell_size)
            )

        # Draw stats
        stats = env.get_stats()
        logged_stats = env.log_stats(stats, logged_stats)
        stats_y = grid_size * cell_size + 5

        step_text = font.render(f"Step: {stats['step']}", True, WHITE)
        pop_text = font.render(f"Population: {stats['alive_agents']}", True, WHITE)
        food_text = font.render(f"Food: {stats['total_food']}", True, WHITE)
        energy_text = font.render(f"Avg Energy: {stats['avg_energy']:.1f}", True, WHITE)

        screen.blit(step_text, (10, stats_y))
        screen.blit(pop_text, (120, stats_y))
        screen.blit(food_text, (280, stats_y))
        screen.blit(energy_text, (380, stats_y))

        paused_text = font.render(
            "PAUSED (Space to resume)" if paused else "Space=Pause  Esc=Quit", True, GRAY
        )
        screen.blit(paused_text, (10, stats_y + 25))

        pygame.display.flip()

        # Check for extinction
        if stats["alive_agents"] == 0:
            print(f"Extinction at step {stats['step']}")
            # Keep window open for viewing
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                    ):
                        waiting = False
                        running = False

        clock.tick(fps)

    pygame.quit()

    # Generate final graph
    if logged_stats:
        avg_traits = env.get_average_genome_traits()
        plot_simulation_stats(logged_stats, population, avg_traits)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Pygame visualization for pond_spawn")
    parser.add_argument("--grid-size", type=int, default=12, help="Grid size")
    parser.add_argument("--population", type=int, default=100, help="Initial population")
    parser.add_argument("--cell-size", type=int, default=40, help="Pixel size of each cell")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second (simulation speed)")
    return parser.parse_args()


def main():
    args = parse_args()
    run_visualization(
        grid_size=args.grid_size,
        population=args.population,
        cell_size=args.cell_size,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
