use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use game_of_life::chunk::{self, Chunk, NeighborData};
use game_of_life::infinite_flat_world_2::{Coord, InfiniteFlatWorld};
use rand::{SeedableRng, rngs::StdRng};

use game_of_life::toroidal_world::ToroidalWorld;

/// Generations per Criterion iteration. Set with `GOL_GENS=...`.
fn gens_per_iter() -> u64 {
    std::env::var("GOL_GENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100)
}

/// Total generations executed for the current benchmark, warmup included.
static GENS_RUN: AtomicU64 = AtomicU64::new(0);

fn clock(c: &mut Criterion) {
    let gens = gens_per_iter();
    let mut group = c.benchmark_group("clock");

    for side in [2, 4, 10, 15, 20, 25, 30] {
        let mut world = ToroidalWorld::random(side, side, &mut StdRng::seed_from_u64(42));
        let cells = world.num_cells() as u64;

        // One iteration is `gens` generations, so throughput is per iteration.
        group.throughput(Throughput::Elements(cells * gens));
        GENS_RUN.store(0, Ordering::Relaxed);

        group.bench_function(BenchmarkId::from_parameter(side), |b| {
            b.iter_custom(|iters| {
                let n = iters * gens;
                let start = Instant::now();
                for _ in 0..n {
                    world.clock();
                    black_box(&mut world);
                }
                let elapsed = start.elapsed();
                GENS_RUN.fetch_add(n, Ordering::Relaxed);
                elapsed
            })
        });

        eprintln!(
            "side {side}: {} generations executed, {} cells, alive at end {}",
            GENS_RUN.load(Ordering::Relaxed),
            cells,
            world.num_alive(),
        );
    }

    group.finish();
}

fn single_chunk(c: &mut Criterion) {
    let mut group = c.benchmark_group("apply_rules_single_chunk");
    group.throughput(Throughput::Elements((chunk::SIZE * chunk::SIZE) as u64));

    let mut dst = Chunk::empty();
    let live = Chunk::random(&mut StdRng::seed_from_u64(1));
    let zero = Chunk::empty();

    group.bench_function("known_empty", |b| {
        let nbhd = NeighborData::from_fn(|_| black_box(&live));
        b.iter(|| black_box(Chunk::check_activity(nbhd, black_box(&mut dst))))
    });

    // Opaque baselines: same data, but the compiler cannot see it.
    group.bench_function("opaque_empty", |b| {
        b.iter(|| {
            let nbhd = NeighborData::from_fn(|_| black_box(&zero));
            black_box(black_box(&live).apply_rules(nbhd, black_box(&mut dst)))
        })
    });

    group.finish();
}

fn infinite_clock(c: &mut Criterion) {
    let gens = gens_per_iter();
    let mut group = c.benchmark_group("infinite_clock");

    for side in [2, 4, 10, 15, 20] {
        let mut initial_world = InfiniteFlatWorld::empty();
        let mut rng = StdRng::seed_from_u64(42);

        // Seed a dense block of chunks
        for y in 0..side {
            for x in 0..side {
                initial_world.insert_chunk(Coord { x, y }, Chunk::random(&mut rng));
            }
        }

        // 1. Run a dummy simulation to calculate exact chunk evaluations
        let mut dummy_world = initial_world.clone();
        let mut total_active_evals = 0;
        let mut total_empty_evals = 0;

        for _ in 0..gens {
            // Your updated engine returns the exact telemetry tuple
            let (active, empty) = dummy_world.clock();
            total_active_evals += active;
            total_empty_evals += empty;
        }

        let cells_per_chunk = (chunk::SIZE * chunk::SIZE) as u64;
        let total_chunks = (total_active_evals + total_empty_evals) as u64;
        let exact_cells_per_run = total_chunks * cells_per_chunk;

        // 2. Feed the combined cell count to Criterion for accurate GCups
        group.throughput(Throughput::Elements(exact_cells_per_run));
        GENS_RUN.store(0, Ordering::Relaxed);

        group.bench_function(BenchmarkId::from_parameter(side), |b| {
            b.iter_custom(|iters| {
                let mut total_time = std::time::Duration::ZERO;
                let n = gens;

                for _ in 0..iters {
                    // Clone state outside the timer loop
                    let mut world = black_box(initial_world.clone());

                    let start = Instant::now();
                    for _ in 0..n {
                        world.clock();
                        black_box(&mut world);
                    }
                    total_time += start.elapsed();
                }

                GENS_RUN.fetch_add(iters * n, Ordering::Relaxed);
                total_time
            })
        });

        // 3. Print the router telemetry breakdown directly to the console
        eprintln!(
            "infinite side {side}: {} gens. Per tick avg -> Active: {:.1}, Empty: {:.2} (Total cells: {:.0})",
            GENS_RUN.load(Ordering::Relaxed),
            total_active_evals as f64 / gens as f64,
            total_empty_evals as f64 / gens as f64,
            exact_cells_per_run as f64 / gens as f64
        );
    }

    group.finish();
}

fn infinite_clock_sparse(c: &mut Criterion) {
    let gens = gens_per_iter();
    let mut group = c.benchmark_group("infinite_clock_sparse");

    for side in [2, 4, 10, 15, 20] {
        let mut initial_world = InfiniteFlatWorld::empty();
        let mut rng = StdRng::seed_from_u64(42);

        // Seed the exact same number of chunks (side * side),
        // but scatter them widely to destroy spatial locality.
        for y in 0..side {
            for x in 0..side {
                initial_world.insert_chunk(
                    Coord {
                        x: x * 10,
                        y: y * 10,
                    },
                    Chunk::random(&mut rng),
                );
            }
        }

        let mut dummy_world = initial_world.clone();
        let mut total_active_evals = 0;
        let mut total_empty_evals = 0;

        for _ in 0..gens {
            let (active, empty) = dummy_world.clock();
            total_active_evals += active;
            total_empty_evals += empty;
        }

        let cells_per_chunk = (chunk::SIZE * chunk::SIZE) as u64;
        let exact_cells_per_run = (total_active_evals + total_empty_evals) as u64 * cells_per_chunk;

        group.throughput(Throughput::Elements(exact_cells_per_run));
        GENS_RUN.store(0, Ordering::Relaxed);

        group.bench_function(BenchmarkId::from_parameter(side), |b| {
            b.iter_custom(|iters| {
                let mut total_time = std::time::Duration::ZERO;
                let n = gens;

                for _ in 0..iters {
                    let mut world = black_box(initial_world.clone());

                    let start = Instant::now();
                    for _ in 0..n {
                        world.clock();
                        black_box(&mut world);
                    }
                    total_time += start.elapsed();
                }

                GENS_RUN.fetch_add(iters * n, Ordering::Relaxed);
                total_time
            })
        });

        eprintln!(
            "infinite side {side}: {} gens. Per tick avg -> Active: {:.1}, Empty: {:.2} (Total cells: {:.0})",
            GENS_RUN.load(Ordering::Relaxed),
            total_active_evals as f64 / gens as f64,
            total_empty_evals as f64 / gens as f64,
            exact_cells_per_run as f64 / gens as f64
        );
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(20);
    targets = single_chunk, clock, infinite_clock, infinite_clock_sparse
}
criterion_main!(benches);
