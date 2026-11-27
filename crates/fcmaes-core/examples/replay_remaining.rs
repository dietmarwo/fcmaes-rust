//! Seeded runs of the optimizers the remaining-optimizer patch touches; one
//! digest line per run. Built unchanged against the pinned and the patched
//! tree, the lines are diffed: `trace` covers every point and value the
//! objective saw, `result` the returned point.
use fcmaes_core::Objective;
use fcmaes_core::{
    AdvancedRetryConfig, Archive, ArchiveDe, ArchiveDeParams, BiteOpt, BiteParams, Crfmnes,
    CrfmnesParams, DaParams, DiversifierParams, Fitness, Jso, JsoParams, MoRetryConfig, Pgpe,
    PgpeParams, RetryBounds, RetryConfig, RetryRunResult, Rng, advanced_retry, diversify, moretry,
    optimize_da, retry,
};
use std::sync::Mutex;

struct Hash(u64);
impl Hash {
    fn new() -> Self {
        Hash(0xcbf29ce484222325)
    }
    fn u(&mut self, v: u64) {
        for b in v.to_le_bytes() {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x100000001b3);
        }
    }
    fn s(&mut self, v: &[f64]) {
        self.u(v.len() as u64);
        v.iter().for_each(|x| self.u(x.to_bits()));
    }
}
fn digest(v: &[f64]) -> u64 {
    let mut h = Hash::new();
    h.s(v);
    h.0
}

const OFF: [f64; 8] = [-2.71, 1.13, 0.49, -1.77, 3.01, -0.97, 2.21, -3.25];
fn shifted(x: &[f64]) -> f64 {
    x.iter()
        .enumerate()
        .map(|(i, v)| (v - OFF[i % 8]).powi(2))
        .sum()
}
fn rastrigin(x: &[f64]) -> f64 {
    10.0 * x.len() as f64
        + x.iter()
            .map(|v| v * v - 10.0 * (2.0 * std::f64::consts::PI * v).cos())
            .sum::<f64>()
}
fn plateau(x: &[f64]) -> f64 {
    x.iter().map(|v| (v.abs() * 2.0).floor()).sum()
}
fn rosen(x: &[f64]) -> f64 {
    x.windows(2)
        .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
        .sum()
}
fn holes(x: &[f64]) -> f64 {
    if x[0] > 3.0 { f64::NAN } else { shifted(x) }
}
type F = fn(&[f64]) -> f64;
const OBJ: [(&str, F); 5] = [
    ("shifted", shifted),
    ("rastrigin", rastrigin),
    ("plateau", plateau),
    ("rosen", rosen),
    ("holes", holes),
];

type Trace = Mutex<(Hash, u64, f64)>;
fn fresh() -> Trace {
    Mutex::new((Hash::new(), 0, f64::INFINITY))
}
fn see(trace: &Trace, x: &[f64], y: f64) {
    let mut t = trace.lock().unwrap();
    t.0.s(x);
    t.0.u(y.to_bits());
    t.1 += 1;
    if y.is_finite() && y < t.2 {
        t.2 = y;
    }
}
fn line(kind: &str, case: String, trace: &Trace, x: &[f64], y: f64, n: u64) {
    let t = trace.lock().unwrap();
    println!(
        "{kind}|{case}|trace={:016x}|calls={}|seen={:e}|result={:016x}|y={y:e}|n={n}",
        t.0.0,
        t.1,
        t.2,
        digest(x)
    );
}

fn scalar() {
    for (name, f) in OBJ {
        for d in [3usize, 8, 32] {
            let (lo, hi) = (vec![-5.0; d], vec![5.0; d]);
            for seed in 1..=3u64 {
                for budget in [257u64, 2000, 6000] {
                    let case = format!("{name}/d{d}/s{seed}/e{budget}");
                    for pop in [0i32, 31, 64] {
                        let t = fresh();
                        let obj = |x: &[f64]| {
                            let y = f(x);
                            see(&t, x, y);
                            y
                        };
                        let fit = Fitness::bounded(d, 1, &lo, &hi);
                        let p = JsoParams {
                            popsize: pop,
                            max_evaluations: budget,
                            seed,
                            ..Default::default()
                        };
                        let r = Jso::new(fit, &p).optimize(&obj);
                        line(
                            "jso",
                            format!("{case}/p{pop}"),
                            &t,
                            &r.x,
                            r.y,
                            r.evaluations,
                        );
                    }
                    for pop in [0i32, 16, 31] {
                        let t = fresh();
                        let batch = |xs: &[Vec<f64>]| {
                            xs.iter()
                                .map(|x| {
                                    let y = f(x);
                                    see(&t, x, y);
                                    y
                                })
                                .collect::<Vec<_>>()
                        };
                        let fit = Fitness::bounded(d, 1, &lo, &hi);
                        let p = CrfmnesParams {
                            popsize: pop,
                            max_evaluations: budget,
                            seed,
                            ..Default::default()
                        };
                        let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            Crfmnes::new(fit, &vec![0.7; d], 0.3, &p).optimize_batch(batch)
                        }));
                        match run {
                            Ok(r) => line(
                                "crfmnes",
                                format!("{case}/p{pop}"),
                                &t,
                                &r.x,
                                r.y,
                                r.evaluations,
                            ),
                            Err(_) => println!("crfmnes|{case}/p{pop}|panicked"),
                        }
                        let t = fresh();
                        let batch = |xs: &[Vec<f64>]| {
                            xs.iter()
                                .map(|x| {
                                    let y = f(x);
                                    see(&t, x, y);
                                    y
                                })
                                .collect::<Vec<_>>()
                        };
                        let fit = Fitness::bounded(d, 1, &lo, &hi);
                        let p = PgpeParams {
                            popsize: pop,
                            max_evaluations: budget,
                            seed,
                            ..Default::default()
                        };
                        let r = Pgpe::new(fit, &vec![0.7; d], &[0.3], &p).optimize_batch(batch);
                        line(
                            "pgpe",
                            format!("{case}/p{pop}"),
                            &t,
                            &r.x,
                            r.y,
                            r.evaluations,
                        );
                    }
                    for local in [false, true] {
                        let t = fresh();
                        let obj = |x: &[f64]| {
                            let y = f(x);
                            see(&t, x, y);
                            y
                        };
                        let p = DaParams {
                            max_evaluations: budget,
                            use_local_search: local,
                            seed,
                            ..Default::default()
                        };
                        let r = optimize_da(&obj, &vec![0.7; d], lo.clone(), hi.clone(), &p);
                        line(
                            "da",
                            format!("{case}/l{}", local as u8),
                            &t,
                            &r.x,
                            r.y,
                            r.evaluations,
                        );
                    }
                }
            }
        }
    }
}

fn mo(x: &[f64], nobj: usize) -> Vec<f64> {
    let g = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / (x.len() - 1) as f64;
    let mut v = vec![x[0], g * (1.0 - (x[0] / g).sqrt())];
    for k in 2..nobj {
        v.push(g * (k as f64) - x[0] * x[k % x.len()]);
    }
    v.truncate(nobj);
    v
}

fn archive_de() {
    for nobj in [2usize, 3, 5] {
        for d in [6usize, 16] {
            for (gate, tiered) in [(true, true), (false, true), (true, false)] {
                for pop in [16usize, 64] {
                    for seed in 1..=3u32 {
                        let t = fresh();
                        let params = ArchiveDeParams {
                            population: pop,
                            max_evaluations: 3000,
                            seed,
                            gate,
                            tiered,
                        };
                        let mut de: ArchiveDe<Vec<f64>, Vec<f64>> =
                            ArchiveDe::new(vec![0.0; d], vec![1.0; d], nobj, params).unwrap();
                        de.minimize(|x: &Vec<f64>| {
                            let y = mo(x, nobj);
                            let mut g = t.lock().unwrap();
                            g.0.s(x);
                            g.0.s(&y);
                            g.1 += 1;
                            y
                        })
                        .unwrap();
                        let mut h = Hash::new();
                        for (x, y) in de.population() {
                            h.s(&x);
                            h.s(&y);
                        }
                        let g = t.lock().unwrap();
                        println!(
                            "archivede|m{nobj}/d{d}/g{}t{}/p{pop}/s{seed}|trace={:016x}|calls={}|population={:016x}|archive={}|n={}",
                            gate as u8,
                            tiered as u8,
                            g.0.0,
                            g.1,
                            h.0,
                            de.archive().len(),
                            de.evaluations()
                        );
                    }
                }
            }
        }
    }
}

fn inner(
    f: &(dyn Fn(&[f64]) -> f64 + Sync),
    lo: &[f64],
    hi: &[f64],
    seed: u64,
    budget: u64,
) -> RetryRunResult {
    let p = BiteParams {
        max_evaluations: budget,
        seed,
        ..Default::default()
    };
    let r = BiteOpt::new(lo, hi, None, &p).optimize(&|x: &[f64]| f(x));
    RetryRunResult {
        x: r.x,
        y: r.y,
        evaluations: r.evaluations,
    }
}

fn retries() {
    for (name, f) in OBJ {
        for d in [4usize, 12] {
            let bounds = RetryBounds::new(vec![-5.0; d], vec![5.0; d]).unwrap();
            for seed in 1..=3u64 {
                for (runs, stats) in [(8usize, 0usize), (24, 0), (24, 16)] {
                    let config = RetryConfig {
                        num_retries: runs,
                        workers: 1,
                        max_evaluations: 600,
                        seed,
                        statistic_num: stats,
                        ..Default::default()
                    };
                    let case = format!("{name}/d{d}/s{seed}/r{runs}/st{stats}");
                    let t = fresh();
                    let obj = |x: &[f64]| {
                        let y = f(x);
                        see(&t, x, y);
                        y
                    };
                    let r = retry(&obj, &bounds, &config, |o, c| {
                        inner(o, c.bounds.lower(), c.bounds.upper(), c.run_seed, 600)
                    });
                    line("retry", case.clone(), &t, &r.x, r.y, r.evaluations);
                    let t = fresh();
                    let obj = |x: &[f64]| {
                        let y = f(x);
                        see(&t, x, y);
                        y
                    };
                    let adv = AdvancedRetryConfig {
                        retry: config.clone(),
                        ..Default::default()
                    };
                    let r = advanced_retry(&obj, &bounds, &adv, |o, c| {
                        inner(o, c.bounds.lower(), c.bounds.upper(), c.run_seed, 600)
                    });
                    line("advretry", case, &t, &r.x, r.y, r.evaluations);
                }
            }
        }
    }
}

fn mo_retries() {
    for nobj in [2usize, 3] {
        for d in [5usize, 10] {
            let bounds = RetryBounds::new(vec![0.0; d], vec![1.0; d]).unwrap();
            for exp in [2.0f64, 1.0, 3.0] {
                for seed in 1..=4u64 {
                    let mut p = MoRetryConfig::new(vec![0.1; nobj], vec![1.0; nobj]);
                    p.value_exp = exp;
                    p.retry.num_retries = 16;
                    p.retry.workers = 1;
                    p.retry.seed = seed;
                    let t = fresh();
                    let obj = |x: &[f64]| {
                        let y = mo(x, nobj);
                        let mut g = t.lock().unwrap();
                        g.0.s(x);
                        g.0.s(&y);
                        g.1 += 1;
                        y
                    };
                    let r = moretry(&obj, &bounds, &p, |o, c| {
                        inner(
                            &|x: &[f64]| o.eval_scalar(x),
                            c.bounds.lower(),
                            c.bounds.upper(),
                            c.run_seed,
                            500,
                        )
                    })
                    .unwrap();
                    let mut h = Hash::new();
                    for e in &r.entries {
                        h.s(&e.x);
                        h.s(&e.y);
                    }
                    let g = t.lock().unwrap();
                    println!(
                        "moretry|m{nobj}/d{d}/p{exp}/s{seed}|trace={:016x}|calls={}|entries={:016x}|kept={}|n={}",
                        g.0.0,
                        g.1,
                        h.0,
                        r.entries.len(),
                        r.evaluations
                    );
                }
            }
        }
    }
}

fn diversifier() {
    let d = 9;
    for (qd, k, samples) in [
        (2usize, 64usize, 0usize),
        (2, 256, 2),
        (2, 1024, 2),
        (3, 128, 2),
    ] {
        for pop in [16i32, 32] {
            for budget in [1000u64, 4096, 8000] {
                for seed in 1..=3u64 {
                    let mut rng = Rng::new(seed);
                    let (lo, hi) = (vec![0.0; d], vec![1.0; d]);
                    let mut a =
                        Archive::new(d, &vec![0.0; qd], &vec![1.0; qd], k, samples, &mut rng);
                    a.seed_uniform(&lo, &hi, &mut rng);
                    let t = fresh();
                    let mut fit = |x: &[f64]| {
                        let q = 1.0 + x.iter().map(|v| (v - 0.3).powi(2)).sum::<f64>();
                        see(&t, x, q);
                        (q, (0..qd).map(|i| x[i]).collect::<Vec<f64>>())
                    };
                    let p = DiversifierParams {
                        max_evaluations: budget,
                        popsize: pop,
                        stall_criterion: 8,
                    };
                    let (x, y) = diversify(&mut a, &mut fit, &lo, &hi, &p, &mut rng);
                    let mut h = Hash::new();
                    h.s(a.ys());
                    for row in a.xs() {
                        h.s(row);
                    }
                    let g = t.lock().unwrap();
                    println!(
                        "diversifier|q{qd}/k{k}/c{samples}/p{pop}/e{budget}/s{seed}|trace={:016x}|calls={}|archive={:016x}|result={:016x}|y={y:e}",
                        g.0.0,
                        g.1,
                        h.0,
                        digest(&x)
                    );
                }
            }
        }
    }
}

fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    let which: Vec<String> = std::env::args().skip(1).collect();
    let on = |name: &str| which.is_empty() || which.iter().any(|w| w == name);
    if on("scalar") {
        scalar();
    }
    if on("archivede") {
        archive_de();
    }
    if on("retry") {
        retries();
    }
    if on("moretry") {
        mo_retries();
    }
    if on("diversifier") {
        diversifier();
    }
}
