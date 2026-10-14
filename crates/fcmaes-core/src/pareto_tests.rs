use super::*;

#[test]
fn batch_payload_deferral_partial_error_snapshot_and_reset() {
    use std::cell::Cell;
    use std::rc::Rc;
    #[derive(Debug)]
    struct Payload(Rc<Cell<usize>>, usize);
    impl Clone for Payload {
        fn clone(&self) -> Self {
            self.0.set(self.0.get() + 1);
            Self(self.0.clone(), self.1)
        }
    }
    let clones = Rc::new(Cell::new(0));
    let payloads: Vec<_> = (0..4).map(|id| Payload(clones.clone(), id)).collect();
    let mut archive = ParetoArchive2D::new(Bounds::nonnegative([1.0; 2]).unwrap(), Score::Linear);
    let points = [[0.5, 0.5], [0.6, 0.6], [0.4, 0.6], [f64::NAN, 0.1]];
    let error = archive.add_batch(&points, &payloads).unwrap_err();
    assert_eq!(error.row, Some(3));
    assert_eq!(error.accepted, [true, false, true]);
    assert_eq!(error.source, Error::InvalidPoint);
    assert_eq!(clones.get(), 2);
    assert_eq!(archive.size(), 2);
    let error = archive.add_batch(&points, &payloads[..1]).unwrap_err();
    assert_eq!(error.row, None);
    assert!(error.accepted.is_empty());
    assert_eq!(clones.get(), 2);
    let snapshot = archive.front();
    archive.clear();
    assert!(archive.is_empty());
    assert_eq!(archive.hypervolume(), 0.0);
    assert_eq!(snapshot[0].0, [0.4, 0.6]);
    assert_eq!(snapshot[0].1.1, 2);
    assert_eq!(archive.add_batch(&[], &[]).unwrap(), Vec::<bool>::new());
    archive.add([0.1, 0.1], Payload(clones, 9)).unwrap();
    assert_eq!(archive.len(), 1);
}

#[test]
fn mode_tracking_preserves_population_and_all_feasible_evaluations() {
    use crate::{Fitness, Mode, ModeParams};
    let run = |track| {
        let mut mode = Mode::try_new(
            Fitness::bounded(2, 3, &[0.0; 2], &[1.0; 2]),
            2,
            1,
            None,
            &ModeParams {
                popsize: 32,
                seed: 42,
                ..Default::default()
            },
        )
        .unwrap();
        let mut archive =
            ParetoArchive2D::new(Bounds::nonnegative([2.0; 2]).unwrap(), Score::Linear);
        let mut rows = Vec::new();
        let mut trace = Vec::new();
        for _ in 0..20 {
            let xs = mode.try_ask().unwrap();
            let ys: Vec<Vec<f64>> = xs
                .iter()
                .map(|x| {
                    vec![
                        x.iter().map(|v| v * v).sum(),
                        x.iter().map(|v| (1.0 - v).powi(2)).sum(),
                        x.iter().sum::<f64>() - 1.5,
                    ]
                })
                .collect();
            for (x, y) in xs.iter().zip(&ys) {
                let id = trace.len() as u64;
                trace.push((x.clone(), y.clone()));
                if y.iter().all(|v| v.is_finite()) && y[2] <= 0.0 {
                    rows.push(([y[0], y[1]], id));
                    if track {
                        archive.add([y[0], y[1]], id).unwrap();
                    }
                }
            }
            mode.try_tell(&ys).unwrap();
        }
        (mode.result(), trace, rows, archive)
    };
    let (plain, trace0, _, _) = run(false);
    let (tracked, trace1, rows, archive) = run(true);
    assert_eq!(trace0.len(), 640);
    assert_eq!(trace0, trace1);
    assert_eq!(plain.x, tracked.x);
    assert_eq!(plain.y, tracked.y);
    assert_eq!(plain.iterations, tracked.iterations);
    let actual: Vec<_> = archive
        .entries()
        .map(|e| (e.point(), *e.payload()))
        .collect();
    assert_eq!(actual, oracle(&rows));
    close(archive.hypervolume(), union_hv(&rows, [2.0; 2]));
}

fn oracle(rows: &[([f64; 2], u64)]) -> Vec<([f64; 2], u64)> {
    let mut result: Vec<_> = rows
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(i, (p, id))| {
            let dominated = rows
                .iter()
                .any(|(q, _)| q[0] <= p[0] && q[1] <= p[1] && *q != p);
            let duplicate = rows[..i].iter().any(|(q, _)| *q == p);
            (!dominated && !duplicate).then_some((p, id))
        })
        .collect();
    result.sort_unstable_by(|a, b| a.0[0].total_cmp(&b.0[0]).then(a.0[1].total_cmp(&b.0[1])));
    result
}
// Independent union over ALL rows, not the produced front or its local slabs.
fn union_hv(rows: &[([f64; 2], u64)], reference: [f64; 2]) -> f64 {
    if rows.is_empty() {
        return 0.0;
    }
    let mut points: Vec<_> = rows.iter().map(|r| r.0).collect();
    points.sort_unstable_by(|a, b| a[0].total_cmp(&b[0]));
    let mut last_x = points[0][0];
    let mut best_y = reference[1];
    let mut hv = 0.0;
    for p in points {
        hv += (p[0] - last_x) * (reference[1] - best_y);
        last_x = p[0];
        best_y = best_y.min(p[1]);
    }
    hv + (reference[0] - last_x) * (reference[1] - best_y)
}
fn close(a: f64, b: f64) {
    assert!(a.is_finite() && b.is_finite());
    assert!(
        (a - b).abs() <= 1e-11 * a.abs().max(b.abs()).max(1.0),
        "{a} != {b}"
    );
}
fn check_stream(points: &[[f64; 2]], bounds: Bounds) {
    for score in [Score::Linear, Score::SquaredEuclidean] {
        let mut a = ScoredArchive2d::new(bounds, score);
        let mut b = ParetoArchive2D::new(bounds, score);
        let mut rows = Vec::new();
        for (id, &p) in points.iter().enumerate() {
            let aa = a.insert(p, id as u64).unwrap();
            let bb = b.insert(p, id as u64).unwrap();
            assert_eq!(aa, bb);
            rows.push((p.map(|x| if x == 0.0 { 0.0 } else { x }), id as u64));
            let expected = oracle(&rows);
            let actual: Vec<_> = b.entries().map(|e| (e.point(), *e.payload())).collect();
            assert_eq!(actual, expected);
            let mut scalar: Vec<_> = a
                .entries()
                .iter()
                .map(|e| (e.point(), *e.payload()))
                .collect();
            scalar
                .sort_unstable_by(|x, y| x.0[0].total_cmp(&y.0[0]).then(x.0[1].total_cmp(&y.0[1])));
            assert_eq!(scalar, expected);
            close(a.hypervolume(), union_hv(&rows, bounds.reference()));
            close(b.hypervolume(), union_hv(&rows, bounds.reference()));
            assert_eq!(
                a.preferred().map(|e| *e.payload()),
                b.preferred().map(|e| *e.payload())
            );
            assert_eq!(
                a.entries().iter().map(|e| *e.payload()).collect::<Vec<_>>(),
                b.ranked().iter().map(|e| *e.payload()).collect::<Vec<_>>()
            );
            let coords: Vec<_> = b.entries().map(|e| e.point()).collect();
            assert!(
                coords
                    .windows(2)
                    .all(|w| w[0][0] < w[1][0] && w[0][1] > w[1][1])
            );
        }
    }
}
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[test]
fn signed_and_unsigned_streams_600() {
    let mut rng = Rng(20261007);
    for family in 0..3 {
        for _ in 0..200 {
            let (points, bounds) = match family {
                0 => (
                    (0..100)
                        .map(|_| {
                            [
                                (rng.next() % 17) as f64 - 8.0,
                                (rng.next() % 17) as f64 - 8.0,
                            ]
                        })
                        .collect::<Vec<_>>(),
                    Bounds::new([-8.0; 2], [8.0; 2]).unwrap(),
                ),
                1 => (
                    (0..100)
                        .map(|_| [2.0 * rng.unit() - 1.0, 2.0 * rng.unit() - 1.0])
                        .collect(),
                    Bounds::new([-1.0; 2], [1.0; 2]).unwrap(),
                ),
                _ => (
                    (0..100)
                        .map(|_| {
                            let x = (rng.next() % 65) as f64 / 64.0;
                            [x, 1.0 - x]
                        })
                        .collect(),
                    Bounds::nonnegative([1.0; 2]).unwrap(),
                ),
            };
            check_stream(&points, bounds);
        }
    }
    println!("600 streams; 60,000 updates per backend/score; 240,000 archive updates checked");
}

#[test]
fn original_nan_failure_and_extreme_finite_coordinates() {
    let bounds = Bounds::nonnegative([1e308, 1.0]).unwrap();
    check_stream(
        &[
            [0.0, 1.0],
            [1e308, 0.0],
            [0.25, 0.75],
            [0.5, 0.5],
            [0.75, 0.25],
            [2.0, 0.0],
        ],
        bounds,
    );
    let mut a = ParetoArchive2D::new(bounds, Score::Linear);
    for p in [
        [0.0, 1.0],
        [1e308, 0.0],
        [0.25, 0.75],
        [0.5, 0.5],
        [0.75, 0.25],
        [2.0, 0.0],
    ] {
        a.insert(p, ()).unwrap();
    }
    assert_eq!(a.len(), 5);
    assert!(!a.entries().any(|e| e.point() == [1e308, 0.0]));
    let tiny = f64::from_bits(1);
    check_stream(
        &[
            [-tiny, tiny],
            [0.0, 0.0],
            [tiny, -tiny],
            [-tiny, 0.0],
            [-tiny, -tiny],
        ],
        Bounds::new([-tiny; 2], [tiny; 2]).unwrap(),
    );
    // Score preparation alone must stay finite on opposite signed extremes.
    let opposite = Bounds::new([-f64::MAX; 2], [f64::MAX; 2]).unwrap();
    for p in [[-f64::MAX; 2], [0.0; 2], [f64::MAX; 2]] {
        for s in [Score::Linear, Score::SquaredEuclidean] {
            assert!((0.0..=1.0).contains(&opposite.score(p, s).unwrap()));
        }
    }
}

#[test]
fn fixed_hv_values_boundaries_and_preference() {
    for score in [Score::Linear, Score::SquaredEuclidean] {
        for (reference, expected) in [([4.0, 4.0], 5.0), ([5.0, 5.0], 12.0)] {
            let bounds = Bounds::nonnegative(reference).unwrap();
            let mut a = ScoredArchive2d::new(bounds, score);
            let mut b = ParetoArchive2D::new(bounds, score);
            for p in [[1.0, 3.0], [3.0, 1.0]] {
                a.insert(p, ()).unwrap();
                b.insert(p, ()).unwrap();
            }
            assert_eq!(a.hypervolume(), expected);
            assert_eq!(b.hypervolume(), expected);
        }
        let mut a = ParetoArchive2D::new(Bounds::nonnegative([10.0; 2]).unwrap(), score);
        a.insert([1.0, 9.0], 1).unwrap();
        a.insert([6.0, 6.0], 2).unwrap();
        assert_eq!(
            *a.preferred().unwrap().payload(),
            if score == Score::Linear { 1 } else { 2 }
        );
    }
    check_stream(
        &[
            [1.0, 3.0],
            [1.0, 2.0],
            [2.0, 2.0],
            [0.0, 4.0],
            [3.0, 1.0],
            [1.0, 2.0],
        ],
        Bounds::nonnegative([4.0; 2]).unwrap(),
    );
    check_stream(&[[0.0, 0.0]], Bounds::nonnegative([0.0; 2]).unwrap());
    check_stream(
        &[[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        Bounds::nonnegative([1.0; 2]).unwrap(),
    );
    check_stream(
        &[[-0.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
        Bounds::nonnegative([1.0; 2]).unwrap(),
    );
}

#[test]
fn invalid_inputs_and_numeric_overflow_leave_state_unchanged() {
    for (lo, hi) in [
        ([f64::NAN, 0.0], [1.0; 2]),
        ([0.0; 2], [f64::INFINITY, 1.0]),
        ([2.0, 0.0], [1.0; 2]),
    ] {
        assert_eq!(Bounds::new(lo, hi), Err(Error::InvalidBounds));
    }
    let bounds = Bounds::new([-f64::MAX, 0.0], [f64::MAX, 1.0]).unwrap();
    let mut a = ScoredArchive2d::new(bounds, Score::Linear);
    let mut b = ParetoArchive2D::new(bounds, Score::Linear);
    for p in [[-f64::MAX, 1.0], [0.0, 0.0]] {
        a.insert(p, 7).unwrap();
        b.insert(p, 7).unwrap();
    }
    let state_a: Vec<_> = a
        .entries()
        .iter()
        .map(|e| (e.point(), *e.payload()))
        .collect();
    let state_b: Vec<_> = b.entries().map(|e| (e.point(), *e.payload())).collect();
    assert_eq!(a.hypervolume(), f64::MAX);
    assert_eq!(b.hypervolume(), f64::MAX);
    assert_eq!(a.insert([-f64::MAX, 0.0], 9), Err(Error::NonFiniteArea));
    assert_eq!(b.insert([-f64::MAX, 0.0], 9), Err(Error::NonFiniteArea));
    for p in [[f64::NAN, 0.0], [f64::INFINITY, 0.0], [0.0, -1.0]] {
        assert_eq!(a.insert(p, 9), Err(Error::InvalidPoint));
        assert_eq!(b.insert(p, 9), Err(Error::InvalidPoint));
    }
    assert_eq!(
        state_a,
        a.entries()
            .iter()
            .map(|e| (e.point(), *e.payload()))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        state_b,
        b.entries()
            .map(|e| (e.point(), *e.payload()))
            .collect::<Vec<_>>()
    );
    assert_eq!(a.hypervolume(), f64::MAX);
    assert_eq!(b.hypervolume(), f64::MAX);
    assert_eq!(*b.preferred().unwrap().payload(), 7);
}

#[test]
fn payload_duplicates_and_mass_deletion() {
    let bounds = Bounds::nonnegative([1.0; 2]).unwrap();
    let mut a = ScoredArchive2d::new(bounds, Score::Linear);
    let mut b = ParetoArchive2D::new(bounds, Score::Linear);
    for i in 0..256 {
        let p = [i as f64 / 256.0, 1.0 - i as f64 / 256.0];
        a.insert(p, vec![i as f64, 42.0]).unwrap();
        b.insert(p, vec![i as f64, 42.0]).unwrap();
    }
    assert_eq!(a.insert([0.0, 1.0], vec![999.0]), Ok(Update::Rejected));
    assert_eq!(b.insert([0.0, 1.0], vec![999.0]), Ok(Update::Rejected));
    assert_eq!(b.entries().next().unwrap().payload(), &vec![0.0, 42.0]);
    assert_eq!(
        a.insert([0.0, 0.0], vec![5.0]),
        Ok(Update::Accepted { removed: 256 })
    );
    assert_eq!(
        b.insert([0.0, 0.0], vec![5.0]),
        Ok(Update::Accepted { removed: 256 })
    );
    assert_eq!(a.hypervolume(), 1.0);
    assert_eq!(b.hypervolume(), 1.0);
    assert_eq!(a.preferred().unwrap().payload(), &vec![5.0]);
    assert_eq!(b.preferred().unwrap().payload(), &vec![5.0]);
}

#[test]
fn lazy_payload_is_not_built_for_rejection_or_errors() {
    use std::cell::Cell;
    let calls = Cell::new(0);
    let bounds = Bounds::nonnegative([1.0; 2]).unwrap();
    let mut a = ScoredArchive2d::new(bounds, Score::Linear);
    let mut b = ParetoArchive2D::new(bounds, Score::Linear);
    let make = || {
        calls.set(calls.get() + 1);
        vec![12.0, 34.0]
    };
    a.insert_with([0.25, 0.25], make).unwrap();
    b.insert_with([0.25, 0.25], make).unwrap();
    for p in [[0.25, 0.25], [0.5, 0.5]] {
        assert_eq!(a.insert_with(p, make), Ok(Update::Rejected));
        assert_eq!(b.insert_with(p, make), Ok(Update::Rejected));
    }
    assert_eq!(
        a.insert_with([f64::NAN, 0.0], make),
        Err(Error::InvalidPoint)
    );
    assert_eq!(
        b.insert_with([f64::NAN, 0.0], make),
        Err(Error::InvalidPoint)
    );
    assert_eq!(calls.get(), 2);
    let large = Bounds::new([-f64::MAX, 0.0], [f64::MAX, 1.0]).unwrap();
    let mut a = ScoredArchive2d::new(large, Score::Linear);
    let mut b = ParetoArchive2D::new(large, Score::Linear);
    assert_eq!(
        a.insert_with([-f64::MAX, 0.0], make),
        Err(Error::NonFiniteArea)
    );
    assert_eq!(
        b.insert_with([-f64::MAX, 0.0], make),
        Err(Error::NonFiniteArea)
    );
    assert_eq!(calls.get(), 2);
}

#[test]
fn integer_cell_union_and_long_accepted_stream() {
    let mut rng = Rng(7654);
    for _ in 0..200 {
        let mut a = ParetoArchive2D::new(
            Bounds::new([-8.0; 2], [8.0; 2]).unwrap(),
            Score::SquaredEuclidean,
        );
        let points: Vec<_> = (0..40)
            .map(|_| {
                [
                    (rng.next() % 17) as f64 - 8.0,
                    (rng.next() % 17) as f64 - 8.0,
                ]
            })
            .collect();
        for &p in &points {
            a.insert(p, ()).unwrap();
        }
        let mut cells = 0;
        for x in -8..8 {
            for y in -8..8 {
                cells += usize::from(points.iter().any(|p| p[0] <= x as f64 && p[1] <= y as f64));
            }
        }
        assert_eq!(a.hypervolume(), cells as f64);
    }
    let mut a = ParetoArchive2D::new(Bounds::nonnegative([1.0; 2]).unwrap(), Score::Linear);
    for i in 1..=50_000 {
        let x = i as f64 / 50_001.0;
        a.insert([x, 1.0 - x], i).unwrap();
    }
    assert_eq!(a.len(), 50_000);
    let rows: Vec<_> = a.entries().map(|e| (e.point(), 0)).collect();
    close(a.hypervolume(), union_hv(&rows, [1.0; 2]));
    assert_eq!(
        a.insert([0.0, 0.0], 0),
        Ok(Update::Accepted { removed: 50_000 })
    );
    assert_eq!(a.hypervolume(), 1.0);
}

#[test]
fn cached_map_reference_reset_and_payload_panic() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    for score in [Score::Linear, Score::SquaredEuclidean] {
        let mut a = ParetoArchive2D::new(Bounds::nonnegative([1.0; 2]).unwrap(), score);
        assert!(a.add([1.0; 2], 1).unwrap().accepted());
        assert_eq!(a.hypervolume(), 0.0);
        assert_eq!(a.add([1.0; 2], 2).unwrap(), Update::Rejected);
        assert_eq!(*a.preferred().unwrap().payload(), 1);
        assert!(a.add([0.5; 2], 3).unwrap().accepted());
        let before = a.front();
        let hv = a.hypervolume();
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                let _ = a.insert_with([0.25; 2], || panic!("payload factory"));
            }))
            .is_err()
        );
        assert_eq!(a.front(), before);
        assert_eq!(a.hypervolume(), hv);
        assert_eq!(*a.preferred().unwrap().payload(), 3);
        assert_eq!(a.add([2.0; 2], 4), Err(Error::InvalidPoint));
        assert!(a.add([0.25; 2], 5).unwrap().accepted());
        a.clear();
        assert!(a.preferred().is_none());
        assert!(a.add([1.0; 2], 6).unwrap().accepted());
        assert_eq!(a.front(), vec![([1.0; 2], 6)]);
        assert_eq!(a.hypervolume(), 0.0);
    }
}

#[test]
fn cached_acceptance_clone_and_panic_rollback() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    for score in [Score::Linear, Score::SquaredEuclidean] {
        let mut original = ParetoArchive2D::new(Bounds::nonnegative([10.0; 2]).unwrap(), score);
        for p in [[1.0, 9.0], [9.0, 1.0], [3.0, 7.0]] {
            original.insert(p, vec![42, 43]).unwrap();
        }
        let clone = original.clone();
        original.clear();
        drop(original);
        // Append, cached interior gap, same-key replacement, and mass deletion.
        for p in [[9.5, 0.5], [5.0, 5.0], [3.0, 6.0], [0.0, 0.0]] {
            let mut archive = clone.clone();
            let front = archive.front();
            let hv = archive.hypervolume();
            let preferred = archive.preferred().unwrap().point();
            assert!(
                catch_unwind(AssertUnwindSafe(|| {
                    let _ = archive.insert_with(p, || -> Vec<i32> { panic!("payload factory") });
                }))
                .is_err()
            );
            assert_eq!(archive.front(), front);
            assert_eq!(archive.hypervolume(), hv);
            assert_eq!(archive.preferred().unwrap().point(), preferred);
            assert!(archive.insert(p, vec![7]).unwrap().accepted());
            let rows: Vec<_> = archive.entries().map(|e| (e.point(), 0)).collect();
            close(archive.hypervolume(), union_hv(&rows, [10.0; 2]));
            assert_eq!(clone.front(), front);
            // A Rust move transfers ownership; the moved value remains usable.
            let mut moved = archive;
            moved.clear();
            assert!(moved.insert([10.0; 2], vec![8]).unwrap().accepted());
            assert_eq!(moved.hypervolume(), 0.0);
        }
    }
}
