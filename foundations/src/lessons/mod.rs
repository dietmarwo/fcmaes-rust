//! Seven deliberately compact lessons.

pub mod l1_first_run;
pub mod l2_compare;
pub mod l3_seeds;
pub mod l4_constraints;
pub mod l5_multiobjective;
pub mod l6_mixed;
pub mod l7_archive;

/// Run one lesson or the complete ladder and return its stable text table.
pub fn run(selection: &str, workers: i32) -> Result<String, String> {
    let lessons: [fn(i32) -> Result<String, String>; 7] = [
        l1_first_run::run,
        l2_compare::run,
        l3_seeds::run,
        l4_constraints::run,
        l5_multiobjective::run,
        l6_mixed::run,
        l7_archive::run,
    ];
    if selection == "all" {
        return lessons
            .iter()
            .map(|lesson| lesson(workers))
            .collect::<Result<Vec<_>, _>>()
            .map(|parts| parts.join(""));
    }
    let index: usize = selection
        .parse()
        .map_err(|_| "lesson must be 1..7 or all".to_owned())?;
    lessons
        .get(index.wrapping_sub(1))
        .ok_or_else(|| "lesson must be 1..7 or all".to_owned())?(workers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;
    use std::time::Instant;

    #[test]
    fn lesson_files_stay_small() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lessons");
        for entry in fs::read_dir(root).unwrap() {
            let path = entry.unwrap().path();
            if path.file_name().and_then(|name| name.to_str()) == Some("mod.rs") {
                continue;
            }
            let lines = fs::read_to_string(&path).unwrap().lines().count();
            assert!(lines <= 120, "{} has {lines} lines", path.display());
        }

        let readme =
            fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("README.md")).unwrap();
        let mut current = None;
        let mut counts = [0usize; 7];
        for line in readme.lines() {
            if let Some(label) = line.strip_prefix("### L") {
                current = label
                    .split_whitespace()
                    .next()
                    .and_then(|value| value.parse::<usize>().ok())
                    .filter(|value| (1..=7).contains(value));
            } else if line.starts_with("## ") {
                current = None;
            }
            if let Some(lesson) = current {
                counts[lesson - 1] += 1;
            }
        }
        for (index, lines) in counts.into_iter().enumerate() {
            assert!(
                lines > 0 && lines <= 90,
                "L{} prose has {lines} lines",
                index + 1
            );
        }
    }

    #[test]
    fn ladder_is_deterministic_and_fast() {
        for lesson in 1..=7 {
            let started = Instant::now();
            run(&lesson.to_string(), 2).unwrap();
            assert!(
                started.elapsed().as_secs_f64() < 5.0,
                "lesson {lesson} exceeded its five-second budget"
            );
        }
        let started = Instant::now();
        let first = run("all", 2).unwrap();
        let second = run("all", 2).unwrap();
        assert_eq!(first, second);
        assert_eq!(
            first,
            include_str!("../../results/expected/ladder.txt"),
            "lesson output changed; review the teaching contract before updating the fixture"
        );
        assert!(started.elapsed().as_secs_f64() < 35.0);
    }
}
