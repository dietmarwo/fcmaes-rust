//! Cross-language fixture replay. CSV on stdin; prefix metrics and final fronts on stdout.
use fcmaes_core::{ParetoArchive2D, ParetoBounds2D, ParetoScore};
use std::io::{self, BufRead};

fn emit(case: &str, archive: &ParetoArchive2D<usize>) {
    for entry in archive.entries() {
        let [a, b] = entry.point();
        println!("F,{case},{a:.17e},{b:.17e},{}", entry.payload());
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut archive = ParetoArchive2D::new(
        ParetoBounds2D::new([-2.0; 2], [2.0; 2])?,
        ParetoScore::Linear,
    );
    let mut case = String::new();
    for line in io::stdin().lock().lines().skip(1) {
        let line = line?;
        let row: Vec<_> = line.split(',').collect();
        if row.len() != 4 {
            return Err("expected case,a,b,id".into());
        }
        if row[0] != case {
            emit(&case, &archive);
            archive.clear();
            case = row[0].to_owned();
        }
        let point = [row[1].parse()?, row[2].parse()?];
        let id: usize = row[3].parse()?;
        let accepted = archive.add(point, id)?.accepted();
        println!(
            "U,{case},{id},{},{:.17e},{}",
            usize::from(accepted),
            archive.hypervolume(),
            archive.len()
        );
    }
    emit(&case, &archive);
    Ok(())
}
