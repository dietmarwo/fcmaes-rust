//! Timing smoke: little-endian u64 count followed by objective f64 pairs.
use fcmaes_core::{ParetoArchive2D, ParetoBounds2D, ParetoScore};
use std::{fs, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("expected binary input path")?;
    let data = fs::read(path)?;
    if data.len() < 8 {
        return Err("missing input header".into());
    }
    let n = u64::from_le_bytes(data[..8].try_into()?) as usize;
    if data.len() != 8 + 16 * n {
        return Err("invalid input length".into());
    }
    let points: Vec<[f64; 2]> = data[8..]
        .chunks_exact(16)
        .map(|bytes| {
            [
                f64::from_le_bytes(bytes[..8].try_into().unwrap()),
                f64::from_le_bytes(bytes[8..].try_into().unwrap()),
            ]
        })
        .collect();
    let payloads = vec![vec![0.0; 16]; n];
    for rep in 0..8 {
        let mut archive =
            ParetoArchive2D::new(ParetoBounds2D::nonnegative([1.0; 2])?, ParetoScore::Linear);
        let start = Instant::now();
        let mask = archive.add_batch(&points, &payloads)?;
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        println!(
            "{rep},{ms:.9},{},{:.17e},{}",
            archive.len(),
            archive.hypervolume(),
            mask.iter().filter(|&&x| x).count()
        );
    }
    Ok(())
}
