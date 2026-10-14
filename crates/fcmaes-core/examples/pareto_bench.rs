//! Insertion throughput of `ParetoArchive2D` on three synthetic streams.
//!
//! `cargo run --release -p fcmaes-core --example pareto_bench [points]`
use fcmaes_core::ParetoArchive2D;
use std::time::Instant;

struct SplitMix(u64);
impl SplitMix {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n: usize = std::env::args()
        .nth(1)
        .map_or(Ok(1_000_000), |s| s.parse())?;
    let mut rng = SplitMix(42);
    let uniform: Vec<[f64; 2]> = (0..n).map(|_| [rng.unit(), rng.unit()]).collect();
    // Noise around the front x + y = 1 shrinks over the run, like an optimizer converging.
    let converging: Vec<[f64; 2]> = (0..n)
        .map(|i| {
            let x = rng.unit();
            let noise = 0.5 * (1.0 - i as f64 / n as f64) * rng.unit();
            [x, 1.0 - x + noise]
        })
        .collect();
    // Every point is nondominated; random order exercises interior inserts.
    let antichain: Vec<[f64; 2]> = (0..n)
        .map(|_| {
            let x = rng.unit();
            [x, 1.0 - x]
        })
        .collect();
    let ids: Vec<u64> = (0..n as u64).collect();
    println!("stream,points,ns_per_point,archive,hypervolume");
    for (name, points) in [
        ("uniform", &uniform),
        ("converging", &converging),
        ("antichain", &antichain),
    ] {
        let mut archive = ParetoArchive2D::new([1.5, 1.5])?;
        let start = Instant::now();
        archive.insert_batch(points, &ids)?;
        let ns = start.elapsed().as_secs_f64() * 1e9 / n as f64;
        println!(
            "{name},{n},{ns:.1},{},{:.15}",
            archive.len(),
            archive.hypervolume()
        );
    }
    Ok(())
}
