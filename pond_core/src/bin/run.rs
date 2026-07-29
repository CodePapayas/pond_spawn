/// Headless native runner — benchmarks and observes the sim without a renderer.
///
/// Usage:
///   cargo run -p pond_core --bin run --features native -- [grid] [pop] [steps] [seed] [--dump-stats PATH]
///
/// Defaults: 12×12 grid, 100 agents, 500 steps, seed 42.
///
/// `--dump-stats PATH` writes the sampled time-series as CSV — the same series
/// the browser graph panel plots. Diffing two runs' CSV is how determinism is
/// checked between builds and renderers.
use std::time::Instant;
use pond_core::World;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dump_path = args.iter().position(|a| a == "--dump-stats")
        .and_then(|i| args.get(i + 1).cloned());
    // Positional args are read from the run before any flag, so `--dump-stats`
    // and its value never get parsed as grid/pop/steps/seed.
    let positional: Vec<&String> = args[1..].iter()
        .take_while(|a| !a.starts_with("--"))
        .collect();
    let grid_size: usize = positional.first().and_then(|s| s.parse().ok()).unwrap_or(12);
    let population: usize = positional.get(1).and_then(|s| s.parse().ok()).unwrap_or(100);
    let steps: u32      = positional.get(2).and_then(|s| s.parse().ok()).unwrap_or(500);
    let seed: u64       = positional.get(3).and_then(|s| s.parse().ok()).unwrap_or(42);

    println!("pond_core — headless runner");
    println!("grid={}×{}  pop={}  steps={}  seed={}", grid_size, grid_size, population, steps, seed);
    println!("{:<8} {:<8} {:<12} {:<12} {:<10}", "step", "agents", "avg_energy", "total_food", "ms/step");
    println!("{}", "-".repeat(56));

    let mut world = World::new(grid_size, population, seed);
    let print_every = (steps / 20).max(1);
    let total_start = Instant::now();

    for s in 1..=steps {
        let t0 = Instant::now();
        world.step();
        let step_ms = t0.elapsed().as_secs_f64() * 1000.0;

        if s % print_every == 0 || s == steps {
            let stats = world.get_stats();
            println!(
                "{:<8} {:<8} {:<12.2} {:<12} {:<10.3}",
                s,
                stats.alive_agents,
                stats.avg_energy,
                stats.total_food,
                step_ms,
            );
        }

        if world.agent_count() == 0 {
            println!("extinction at step {}", s);
            break;
        }
    }

    let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
    let stats = world.get_stats();
    println!("{}", "-".repeat(56));
    println!(
        "done  total={:.1}ms  avg_step={:.3}ms  final_agents={}",
        total_ms,
        total_ms / steps as f64,
        stats.alive_agents,
    );

    // Print death tallies
    if !stats.deaths.is_empty() {
        println!("\ndeath causes:");
        let mut deaths: Vec<_> = stats.deaths.iter().collect();
        deaths.sort_by_key(|(k, _)| k.as_str());
        for (cause, count) in deaths {
            println!("  {}: {}", cause, count);
        }
    }

    if let Some(path) = dump_path {
        match std::fs::write(&path, world.stats_history.to_csv()) {
            Ok(()) => println!(
                "\nwrote {} samples to {}",
                world.stats_history.len(), path
            ),
            Err(e) => eprintln!("\nfailed to write {}: {}", path, e),
        }
    }

    // Print cluster distribution at final state
    let gc = &world.cluster.genome_cluster_ids;
    if !gc.is_empty() {
        let mut gcounts = [0u32; 8];
        let mut bcounts = [0u32; 32];
        for &id in gc { gcounts[id as usize] += 1; }
        for &id in &world.cluster.brain_cluster_ids { bcounts[id as usize] += 1; }
        println!("\ngenome clusters (k=6): {:?}", &gcounts[..6]);
        println!("brain clusters  (k=24): {:?}", &bcounts[..24]);
    }
}
