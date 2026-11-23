use fcmaes_core::{
    ParetoArchive2D, ParetoUpdate, Rng,
    archivede::{ArchiveDe, ArchiveDeParams, ParetoArchive},
    gd, igd, pareto_indices,
    rankindex::{BordaIndex, GridIndex},
};

#[test]
fn boxes_match_closed_scan_including_ties_and_sparse_keys() {
    let mut rng = Rng::new(991);
    for d in [1, 3, 5, 40] {
        let n = if d == 40 { 40 } else { 778 };
        let p: Vec<_> = (0..n * d)
            .map(|_| (32. * rng.uniform01()).floor() / 32.)
            .collect();
        let b = BordaIndex::try_new(&p, d).unwrap();
        let g = GridIndex::try_new(&p, d, if d == 40 { 1. } else { 0.005 }).unwrap();
        for i in 0..65 {
            let mut low: Vec<_> = (0..d).map(|_| rng.uniform01() * 0.6).collect();
            let mut high: Vec<_> = low.iter().map(|l| l + 0.4).collect();
            if i <= 2 {
                low.fill(f64::NEG_INFINITY);
                high.fill(f64::INFINITY);
            }
            if i == 1 {
                high[d - 1] = 0.5
            }
            if i == 2 {
                low[0] = f64::NAN
            }
            if i == 3 {
                low = p[..d].to_vec();
                high = low.clone();
            }
            if i == 4 {
                low[0] = 2.;
                high[0] = -2.;
            }
            let want: Vec<_> = p
                .chunks_exact(d)
                .enumerate()
                .filter(|(_, r)| (0..d).all(|j| low[j] <= r[j] && r[j] <= high[j]))
                .map(|(id, _)| id as u32)
                .collect();
            let a = b.query(&low, &high);
            let q = g.query(&low, &high);
            assert!(
                a.score_candidates >= a.ids.len() as u64 && a.score_candidates <= a.axis_candidates
            );
            assert!(
                q.full_cells <= q.nonempty_cells && q.nonempty_cells as u128 <= q.touched_cells
            );
            for mut got in [a.ids, q.ids] {
                got.sort_unstable();
                assert_eq!(got, want);
            }
        }
    }
    assert!(BordaIndex::try_new(&[f64::NAN], 1).is_err());
    assert!(GridIndex::try_new(&[0.; 65], 65, 1.).is_err());
    assert_eq!(GridIndex::new(&[], 2, 1.).query(&[0.; 2], &[1.; 2]).ids, []);
    for n in [65_536, 65_537] {
        let p: Vec<_> = (0..n)
            .flat_map(|i| [i as f64, (n - 1 - i) as f64])
            .collect();
        assert_eq!(
            BordaIndex::new(&p, 2)
                .query(&[(n - 2) as f64, 1.], &[(n - 2) as f64, 1.])
                .ids,
            [n as u32 - 2]
        );
    }
}

fn check_rows<const N: usize>() {
    let mut rng = Rng::new(141);
    let mut archive = ParetoArchive::<[f64; N], usize>::new(N);
    let mut expected: Vec<([f64; N], usize)> = vec![];
    let dom = |a: &[f64; N], b: &[f64; N]| a.iter().zip(b).all(|(a, b)| a <= b) && a != b;
    for i in 0..1200 {
        let y = if i % 7 == 0 {
            [3.; N]
        } else {
            std::array::from_fn(|_| (rng.uniform01() * 12.).floor())
        };
        let accept = !expected.iter().any(|(z, _)| dom(z, &y));
        assert_eq!(
            archive.insert_with(y, || {
                assert!(accept);
                i
            }),
            accept
        );
        if accept {
            expected.retain(|(z, _)| !dom(&y, z));
            expected.push((y, i));
        }
        if i % 31 == 0 {
            expected.sort_by(|(a, i), (b, j)| {
                a.iter()
                    .sum::<f64>()
                    .total_cmp(&b.iter().sum())
                    .then(i.cmp(j))
            });
            let got: Vec<_> = archive
                .ordered_entries()
                .into_iter()
                .map(|(y, i)| (*y, *i))
                .collect();
            assert_eq!(got, expected);
        }
    }
    archive.clear();
    assert!(archive.is_empty());
}
#[test]
fn strict_archive_preserves_arrival_payloads_and_rounded_sum_ties() {
    check_rows::<1>();
    check_rows::<2>();
    check_rows::<3>();
    check_rows::<5>();
    let m = 2f64.powi(54);
    for n in [2, 3, 6] {
        let mut a = ParetoArchive::new(n);
        let mut before = vec![0.; n];
        before[0] = m;
        before[1] = 1.;
        let mut better = before.clone();
        better[1] = 0.;
        assert!(a.add(before.clone()) && a.add(better.clone()) && !a.add(before));
        assert_eq!(a.len(), 1);
        assert_eq!(a.ordered_entries()[0].0, &better);
    }
}

#[test]
fn large_three_objective_archive_preserves_equal_sums_and_decisions_across_removals() {
    let mut archive = ParetoArchive::<[f64; 3], usize>::new(3);
    let mut expected: Vec<([f64; 3], usize)> = vec![];
    let mut stream = Vec::new();
    for i in 0..1024 {
        let index = (i * 271) % 1024;
        let x = (index % 32) as f64;
        let y = (index / 32) as f64;
        let point = [x, y, 100. - x - y];
        stream.push(point);
        if i % 5 == 0 {
            stream.push(point);
        }
    }
    stream.extend([[4., 4., 25.], [3., 3., 23.], [0.; 3], [0.; 3]]);
    let dominates = |a: &[f64; 3], b: &[f64; 3]| a != b && a.iter().zip(b).all(|(x, y)| x <= y);
    for (id, y) in stream.into_iter().enumerate() {
        let admit = !expected.iter().any(|(p, _)| dominates(p, &y));
        assert_eq!(archive.insert_with(y, || id), admit);
        if admit {
            expected.retain(|(p, _)| !dominates(&y, p));
            expected.push((y, id));
        }
        if id % 53 == 0 || y == [0.; 3] {
            expected.sort_by(|(a, i), (b, j)| {
                a.iter()
                    .sum::<f64>()
                    .total_cmp(&b.iter().sum())
                    .then(i.cmp(j))
            });
            let snapshot = archive
                .ordered_entries()
                .into_iter()
                .map(|(a, i)| (*a, *i))
                .collect::<Vec<_>>();
            assert_eq!(snapshot, expected);
            assert_eq!(archive.len(), expected.len());
        }
    }
    assert_eq!(archive.len(), 2);
}
#[test]
fn existing_batch_filter_keeps_signed_zero_tie_order_and_trailing_columns() {
    let y = vec![
        vec![0., 1., f64::NAN],
        vec![-0., 1., 3.],
        vec![1., 0.],
        vec![-0., 1.],
        vec![1., 1.],
    ];
    assert_eq!(pareto_indices(&y, 2).unwrap(), [1, 3, 0, 2]);
    assert!(pareto_indices(&y, 3).is_err());
    assert!(pareto_indices(&[], 2).unwrap().is_empty());
}

#[test]
fn ask_tell_is_retryable_and_budgets_hold_for_fixed_and_dynamic_rows() {
    for budget in [0, 2, 8, 9, 31] {
        let p = ArchiveDeParams {
            population: 8,
            max_evaluations: budget,
            seed: 11,
            ..Default::default()
        };
        let mut s =
            ArchiveDe::<[f64; 3], [f64; 2]>::new([-2., 3., 0.], [4., 3., 1.], 2, p).unwrap();
        let mut dyns =
            ArchiveDe::<Vec<f64>, Vec<f64>>::new(vec![-2., 3., 0.], vec![4., 3., 1.], 2, p)
                .unwrap();
        while !s.ask().is_empty() {
            let xs = s.ask().to_vec();
            assert_eq!(xs.as_slice(), s.ask());
            assert_eq!(
                xs.iter().map(|x| x.to_vec()).collect::<Vec<_>>(),
                dyns.ask()
            );
            assert!(s.tell(&[]).is_err());
            let invalid = vec![[f64::NAN, 0.]; xs.len()];
            assert!(s.tell(&invalid).is_err());
            assert_eq!(s.ask(), xs);
            let y: Vec<_> = xs
                .iter()
                .map(|x| {
                    assert_eq!(x[1], 3.);
                    [x[0], -x[0]]
                })
                .collect();
            dyns.tell(&y.iter().map(|r| r.to_vec()).collect::<Vec<_>>())
                .unwrap();
            s.tell(&y).unwrap();
        }
        assert_eq!(s.evaluations(), budget);
        assert_eq!(s.archive().len(), budget as usize);
        assert_eq!(s.population().len(), 8.min(budget as usize));
        assert!(s.tell(&[]).is_err());
    }
    let mut s = ArchiveDe::<[f64; 1], [f64; 3]>::new(
        [0.],
        [1.],
        3,
        ArchiveDeParams {
            population: 4,
            max_evaluations: 31,
            ..Default::default()
        },
    )
    .unwrap();
    s.minimize(|x| [x[0]; 3]).unwrap();
    assert_eq!(s.population().len(), 4);
}

#[test]
fn paged_front_bulk_removal_keeps_payloads_preference_and_numeric_rollback() {
    let mut a = ParetoArchive2D::new([5000.; 2]).unwrap();
    for i in (0..450).rev().chain(450..1000) {
        let z = i as f64 * 2.;
        a.insert([z, 2000. - z], i).unwrap();
    }
    assert_eq!(
        a.insert([201., 499.], 1001).unwrap(),
        ParetoUpdate::Accepted { removed: 650 }
    );
    assert_eq!(a.preferred().unwrap().payload(), &1001);
    let before = a.front();
    let area = a.hypervolume();
    assert!(a.insert([-f64::MAX, -f64::MAX], 1002).is_err());
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || a.insert_with([201., 498.], || panic!("factory"))
        ))
        .is_err()
    );
    assert_eq!(before, a.front());
    assert_eq!(a.hypervolume(), area);
    assert_eq!(a.entries().next_back().unwrap().point(), [1998., 2.]);
}
#[test]
fn indicators_match_full_search_on_raw_and_extreme_coordinates() {
    let mut rng = Rng::new(201);
    for (d, n) in [
        (2, 39),
        (2, 40),
        (2, 140),
        (3, 127),
        (3, 128),
        (5, 256),
        (9, 156),
    ] {
        let front: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..d).map(|_| rng.uniform01()).collect())
            .collect();
        let r: Vec<Vec<f64>> = (0..51)
            .map(|_| (0..d).map(|_| rng.uniform01()).collect())
            .collect();
        let direct = r
            .iter()
            .map(|p| {
                front
                    .iter()
                    .map(|q| {
                        p.iter()
                            .zip(q)
                            .map(|(x, y)| (x - y) * (x - y))
                            .sum::<f64>()
                            .sqrt()
                    })
                    .fold(f64::INFINITY, f64::min)
            })
            .sum::<f64>()
            / r.len() as f64;
        assert_eq!(igd(&front, &r).unwrap(), direct);
        assert_eq!(gd(&r, &front).unwrap(), direct);
    }
    let m = f64::MAX;
    assert_eq!(igd(&[vec![m]], &[vec![m], vec![-m]]).unwrap(), m);
    assert_eq!(igd(&[vec![0.]], &[vec![1e-250]]).unwrap(), 1e-250);
    assert_eq!(
        igd(&[vec![0.; 3]], &[vec![1e-250, 0., 0.]]).unwrap(),
        1e-250
    );
    assert!(igd(&[], &[vec![1.]]).is_err());
}
