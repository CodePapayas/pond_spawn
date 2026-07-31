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
    if args.iter().any(|a| a == "--bench-cluster") {
        bench_cluster();
        return;
    }
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

    // Calibration levers. The pond's survival is a product of several of these
    // at once, and a sweep that cannot switch one off cannot say which one it
    // was — so the runner exposes them even though the browser fixes them at
    // construction.
    let no_predators = args.iter().any(|a| a == "--no-predators");
    let regen: Option<f64> = args.iter().position(|a| a == "--regen")
        .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok());

    println!("pond_core — headless runner");
    println!("grid={}×{}  pop={}  steps={}  seed={}", grid_size, grid_size, population, steps, seed);
    println!("{:<8} {:<8} {:<12} {:<12} {:<10}", "step", "agents", "avg_energy", "total_food", "ms/step");
    println!("{}", "-".repeat(56));

    let mut world = World::new(grid_size, population, seed);
    if no_predators { world.set_automatic_predators(false); }
    if let Some(r) = regen { world.set_food_regen_scale(r); }
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

        // Speciation events as they happen. Watching promotions scroll past in a
        // headless run is how the thresholds get judged; wiring the renderer
        // first would only make bad thresholds prettier.
        for ev in world.species.drain_events() {
            println!("  [species] {}", ev);
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
    // Population trait means. Balance work turns on these — "predators select
    // for defense" is a claim about this line, not about the death tally.
    let means = world.trait_means();
    const TRAIT_NAMES: [&str; 11] = [
        "vision", "speed", "metabolism", "energy_cap", "mutation",
        "repro_cost", "attack", "defense", "aggression", "intelligence", "immunity",
    ];
    println!("\ntrait means:");
    for (name, m) in TRAIT_NAMES.iter().zip(means.iter()) {
        println!("  {:<12} {:.3}", name, m);
    }
    if !world.diseases.is_empty() {
        println!("\ndiseases:");
        for d in &world.diseases {
            let carriers = world.infection.iter().filter(|&&v| v == d.id).count();
            println!(
                "  {:<28} from species {:<3} emerged {:<6} severity {:.3} contagion {:.3}{}  carriers {}",
                d.name, d.origin_species, d.emerged_step, d.severity, d.contagion,
                if d.jumped { " JUMPED" } else { "       " }, carriers,
            );
        }
    }

    if !world.predators.is_empty() {
        println!("\npredators:");
        for p in &world.predators {
            println!(
                "  tier {} · image {:?} · learned attack {:.3} · kills {}",
                p.tier, p.search_image, p.attack, p.kills,
            );
        }
    }

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

    // Species roster: live first, then the fossil record.
    if !world.species.all().is_empty() {
        println!(
            "\nspecies (live {} / total {}, {} on probation):",
            world.species.live_count(), world.species.all().len(), world.species.probation_count(),
        );
        for s in world.species.all() {
            match s.extinct_at {
                None => println!(
                    "  {:<26} #{:<3} alive    founded {:<6} age {:<6} members {:<4} peak {}",
                    s.name.full(), s.id, s.founded_step,
                    s.age(world.get_stats().step), s.members, s.peak_members,
                ),
                Some(end) => println!(
                    "  {:<26} #{:<3} extinct  founded {:<6} lived {:<6} peak {}",
                    s.name.full(), s.id, s.founded_step, s.age(end), s.peak_members,
                ),
            }
        }
    }

    // Print cluster distribution at final state
    let gc = &world.cluster.genome_cluster_ids;
    if !gc.is_empty() {
        // Sized from the live tunable, not a literal: k is a dial now.
        let k = world.tunables().cluster_k;
        let mut gcounts = vec![0u32; k];
        let mut bcounts = [0u32; 32];
        for &id in gc { gcounts[id as usize] += 1; }
        for &id in &world.brain_clusters.labels { bcounts[id as usize] += 1; }
        println!("\ngenome clusters (k={}): {:?}", k, gcounts);
        println!("brain clusters  (k=24): {:?}", &bcounts[..24]);
    }
}

/// Cost of the clustering work, by population.
///
/// Brain clustering was once ~99.5% of a 14–164 ms spike landing every 50 steps,
/// which is a visible stutter against a 16.7 ms frame budget. It is now warm
/// started and spread one iteration per step. This exists so a regression shows
/// up as a number here rather than as a stutter someone notices months later.
fn bench_cluster() {
    use pond_core::brain_cluster::{BrainClusters, COLD_ITERS, WARM_ITERS};
    use pond_core::{ClusterState, Genome};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    println!("cluster cost by population (worst single step, ms)\n");
    println!("{:<8} {:<10} {:<14} {:<14} {:<10}", "pop", "genome", "brain cold", "brain warm", "worst");
    println!("{}", "-".repeat(60));

    for n in [100usize, 300, 600, 1200] {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let genomes: Vec<Genome> = (0..n).map(|_| Genome::generate(&mut rng)).collect();

        let t = Instant::now();
        let _ = ClusterState::run(&genomes, 6, 100, None);
        let genome_ms = t.elapsed().as_secs_f64() * 1000.0;

        let mut bc = BrainClusters::new();
        bc.set_enabled(true);

        // Cold pass: worst single step across its COLD_ITERS steps.
        bc.begin(&genomes, 24, 50);
        let mut cold_worst = 0f64;
        while bc.in_progress() {
            let t = Instant::now();
            bc.advance(&genomes);
            cold_worst = cold_worst.max(t.elapsed().as_secs_f64() * 1000.0);
        }

        // Warm pass: the steady state, which is what actually runs.
        bc.begin(&genomes, 24, 100);
        let mut warm_worst = 0f64;
        while bc.in_progress() {
            let t = Instant::now();
            bc.advance(&genomes);
            warm_worst = warm_worst.max(t.elapsed().as_secs_f64() * 1000.0);
        }

        // The tick that begins a pass also runs the genome pass.
        let worst = genome_ms + warm_worst;
        println!(
            "{:<8} {:<10.3} {:<14.3} {:<14.3} {:<10.3}",
            n, genome_ms, cold_worst, warm_worst, worst,
        );
    }
    println!(
        "\ncold = {} steps, warm = {} steps; one iteration per step.",
        COLD_ITERS, WARM_ITERS,
    );
    println!("60 fps frame budget is 16.7 ms.");
}
