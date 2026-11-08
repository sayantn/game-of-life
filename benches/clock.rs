use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use game_of_life::chunk::{self, Chunk, NeighborData};
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

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(20);
    targets = clock, single_chunk
}
criterion_main!(benches);
