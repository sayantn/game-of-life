use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::{SeedableRng, rngs::StdRng};

use game_of_life::toroidalworld::ToroidalWorld;

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

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(20);
    targets = clock
}
criterion_main!(benches);
