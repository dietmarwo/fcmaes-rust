//! L7: ask the archive for its exact native grid layout.

use fcmaes_core::{Archive, Rng};

pub fn run(_workers: i32) -> Result<String, String> {
    let mut archive = Archive::new(2, &[0.0, 0.0], &[1.0, 1.0], 120, 0, &mut Rng::new(42));
    for row in 0..10 {
        for column in 0..12 {
            let descriptor = [(column as f64 + 0.5) / 12.0, (row as f64 + 0.5) / 10.0];
            let niche = archive.index_of_niche(&descriptor);
            let decision = [descriptor[0], descriptor[1]];
            archive.set(
                niche,
                1.0 + row as f64 + column as f64,
                &descriptor,
                &decision,
            );
        }
    }
    let layout = archive
        .grid_layout()
        .ok_or_else(|| "expected regular grid".to_owned())?;
    Ok(format!(
        "L7 native archive | shape={}x{} cells={} capacity={} occupied={}\n",
        layout.max_columns(),
        layout.rows,
        layout.cells(),
        archive.capacity(),
        archive.occupied()
    ))
}
