//! Replays the shared archive fixture (also used by fast-cma-es's Python tests)
//! and checks every prefix against a brute-force oracle.
use fcmaes_core::ParetoArchive2D;
use std::collections::BTreeMap;

fn cases() -> BTreeMap<String, Vec<([f64; 2], u64)>> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/pareto_archive_2d.csv"
    );
    let text = std::fs::read_to_string(path).expect("fixture");
    let mut cases: BTreeMap<String, Vec<_>> = BTreeMap::new();
    for line in text.lines().skip(1) {
        let row: Vec<_> = line.split(',').collect();
        assert_eq!(row.len(), 4, "expected case,a,b,id");
        let point = [row[1].parse().unwrap(), row[2].parse().unwrap()];
        cases
            .entry(row[0].to_owned())
            .or_default()
            .push((point, row[3].parse().unwrap()));
    }
    cases
}

/// Area dominated by `front` (sorted by the first objective) inside `reference`.
fn area(front: &[([f64; 2], u64)], reference: [f64; 2]) -> f64 {
    let mut hv = 0.0;
    for (i, (p, _)) in front.iter().enumerate() {
        let next_x = front.get(i + 1).map_or(reference[0], |q| q.0[0]);
        hv += (next_x - p[0]) * (reference[1] - p[1]);
    }
    hv
}

#[test]
fn every_fixture_prefix_matches_the_oracle() {
    let reference = [2.0, 2.0];
    let cases = cases();
    assert_eq!(cases.len(), 6);
    let mut archive = ParetoArchive2D::new(reference).unwrap();
    for (case, rows) in &cases {
        archive.clear();
        let mut front: Vec<([f64; 2], u64)> = Vec::new();
        for &(point, id) in rows {
            let p = point.map(|x| if x == 0.0 { 0.0 } else { x });
            let expected = !front.iter().any(|(q, _)| q[0] <= p[0] && q[1] <= p[1]);
            if expected {
                front.retain(|(q, _)| !(p[0] <= q[0] && p[1] <= q[1]));
                front.push((p, id));
                front.sort_by(|a, b| a.0[0].total_cmp(&b.0[0]));
            }
            let update = archive.insert(point, id).unwrap();
            assert_eq!(update.accepted(), expected, "{case} row {id}: acceptance");
            assert_eq!(archive.len(), front.len(), "{case} row {id}: size");
            let hv = area(&front, reference);
            assert!(
                (archive.hypervolume() - hv).abs() <= 1e-11,
                "{case} row {id}: HV {} != {hv}",
                archive.hypervolume()
            );
        }
        assert_eq!(archive.front(), front, "{case}: complete front");
    }
}
