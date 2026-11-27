//! Seeded runs of every optimizer the two patches touch; one digest line per run.
//! Built unchanged against the pinned and the patched tree, the lines are diffed.
use fcmaes_core::{
    Archive, BiteOpt, BiteParams, Cmaes, CmaesParams, De, DeParams, DeepBiteOpt, Fitness,
    MapElitesParams, Mode, ModeParams, Rng, map_elites,
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
    fn f(&mut self, v: f64) {
        self.u(v.to_bits())
    }
    fn s(&mut self, v: &[f64]) {
        self.u(v.len() as u64);
        v.iter().for_each(|x| self.f(*x));
    }
    fn m(&mut self, v: &[Vec<f64>]) {
        self.u(v.len() as u64);
        v.iter().for_each(|x| self.s(x));
    }
}

const OFF: [f64; 8] = [-2.71, 1.13, 0.49, -1.77, 3.01, -0.97, 2.21, -3.25];
fn shifted(x: &[f64]) -> f64 {
    x.iter()
        .enumerate()
        .map(|(i, v)| (v - OFF[i % 8]).powi(2))
        .sum()
}
fn rastrigin(x: &[f64]) -> f64 {
    x.iter()
        .enumerate()
        .map(|(i, v)| {
            let z = v - 0.3 * OFF[i % 8];
            z * z - 10.0 * (2.0 * std::f64::consts::PI * z).cos() + 10.0
        })
        .sum()
}
fn plateau(x: &[f64]) -> f64 {
    shifted(x).floor()
}
fn holes(x: &[f64]) -> f64 {
    if x[0] > 1.0 && x[0] < 2.0 {
        f64::NAN
    } else if x[1] < -4.0 {
        f64::INFINITY
    } else {
        shifted(x)
    }
}
fn rosen(x: &[f64]) -> f64 {
    x.windows(2)
        .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
        .sum()
}
fn ellipsoid(x: &[f64]) -> f64 {
    // Householder reflection, then condition 1e6.
    let n = x.len() as f64;
    let s: f64 = x
        .iter()
        .enumerate()
        .map(|(i, v)| v * (1.0 + i as f64))
        .sum();
    let norm: f64 = (1..=x.len()).map(|i| (i * i) as f64).sum();
    x.iter()
        .enumerate()
        .map(|(i, v)| {
            let r = v - 2.0 * s * (1.0 + i as f64) / norm;
            10f64.powf(6.0 * i as f64 / (n - 1.0).max(1.0)) * r * r
        })
        .sum()
}
type F = fn(&[f64]) -> f64;
const OBJ: [(&str, F); 4] = [
    ("shifted", shifted),
    ("rastrigin", rastrigin),
    ("plateau", plateau),
    ("holes", holes),
];

fn traced(f: F, trace: &Mutex<(Hash, u64, f64)>) -> impl Fn(&[f64]) -> f64 + Sync + '_ {
    move |x: &[f64]| {
        let y = f(x);
        let mut t = trace.lock().unwrap();
        t.0.s(x);
        t.0.f(y);
        t.1 += 1;
        if y.is_finite() && y < t.2 {
            t.2 = y;
        }
        y
    }
}
fn line(kind: &str, case: String, result: u64, trace: &Mutex<(Hash, u64, f64)>, y: f64, n: u64) {
    let t = trace.lock().unwrap();
    println!(
        "{kind}|{case}|trace={:016x}|calls={}|seen={:e}|result={result:016x}|y={y:e}|n={n}",
        t.0.0, t.1, t.2
    );
}
fn fresh() -> Mutex<(Hash, u64, f64)> {
    Mutex::new((Hash::new(), 0, f64::INFINITY))
}

fn bite() {
    for (name, f) in OBJ {
        for d in [3usize, 8, 32] {
            for pop in [0, 16] {
                for seed in 1..=4u64 {
                    let (lo, hi) = (vec![-5.0; d], vec![5.0; d]);
                    let bp = BiteParams {
                        popsize: pop,
                        seed,
                        max_evaluations: 3000 + 7 * seed,
                        ..Default::default()
                    };
                    for arm in ["direct", "deep3", "batch1", "batch7", "batch64"] {
                        let t = fresh();
                        let obj = traced(f, &t);
                        let r = match arm {
                            "direct" => BiteOpt::new(&lo, &hi, None, &bp).optimize(&obj),
                            "deep3" => DeepBiteOpt::new(&lo, &hi, None, &bp, 3).optimize(&obj),
                            _ => {
                                let w: usize = arm[5..].parse().unwrap();
                                BiteOpt::new(&lo, &hi, None, &bp)
                                    .optimize_batch(w, |xs| xs.iter().map(|x| obj(x)).collect())
                            }
                        };
                        let mut h = Hash::new();
                        h.s(&r.x);
                        h.f(r.y);
                        h.u(r.evaluations);
                        drop(obj);
                        line(
                            "bite",
                            format!("{name},d={d},pop={pop},seed={seed},{arm}"),
                            h.0,
                            &t,
                            r.y,
                            r.evaluations,
                        );
                    }
                }
            }
        }
    }
}

fn de() {
    for (name, f) in OBJ {
        for d in [4usize, 8, 32] {
            for variant in ["plain", "ints", "normal"] {
                for seed in 1..=4u64 {
                    let (lo, hi) = (vec![-5.0; d], vec![5.0; d]);
                    let dp = DeParams {
                        seed,
                        popsize: if d == 4 { 5 } else { 32 },
                        max_evaluations: 2500,
                        ..Default::default()
                    };
                    let ints = (variant == "ints")
                        .then(|| (0..d).map(|i| i % 3 == 0).collect::<Vec<bool>>());
                    let (guess, sigma) = if variant == "normal" {
                        (vec![0.5; d], vec![1.5; d])
                    } else {
                        (vec![], vec![])
                    };
                    for arm in ["direct", "ask", "window1", "window8"] {
                        let t = fresh();
                        let obj = traced(f, &t);
                        let mut de = De::new(
                            Fitness::bounded(d, 1, &lo, &hi),
                            &guess,
                            &sigma,
                            ints.clone(),
                            &dp,
                        );
                        let r = match arm {
                            "direct" => de.optimize(&obj),
                            "window1" => de.optimize_delayed(&obj, 1),
                            "window8" => de.optimize_delayed(&obj, 8),
                            _ => {
                                for _ in 0..60 {
                                    let xs = de.ask();
                                    de.tell(&xs.iter().map(|x| obj(x)).collect::<Vec<_>>());
                                }
                                de.result()
                            }
                        };
                        let mut h = Hash::new();
                        h.s(&r.x);
                        h.f(r.y);
                        h.u(r.evaluations);
                        h.u(r.stop as u64);
                        drop(obj);
                        line(
                            "de",
                            format!("{name},d={d},{variant},seed={seed},{arm}"),
                            h.0,
                            &t,
                            r.y,
                            r.evaluations,
                        );
                    }
                }
            }
        }
    }
}

fn mo(x: &[f64], nobj: usize, ncon: usize, quant: bool) -> Vec<f64> {
    let d = x.len();
    let g = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / (d - 1) as f64;
    let mut y = Vec::with_capacity(nobj + ncon);
    for k in 0..nobj {
        let v = match k {
            0 => x[0],
            1 => g * (1.0 - (x[0] / g).sqrt()),
            _ => g * (1.0 + (k as f64 * x[k % d]).sin()) - x[(k + 1) % d],
        };
        y.push(if quant {
            (v * 8.0).round() / 8.0 * if k == 0 && x[1] < 0.2 { -0.0 } else { 1.0 }
        } else {
            v
        });
    }
    for c in 0..ncon {
        y.push(x[c % d] + x[(c + 2) % d] - 0.9 - 0.1 * c as f64);
    }
    if quant && x[2] > 0.97 {
        y[0] = f64::NAN;
    }
    y
}

fn mode() {
    for nobj in [1usize, 2, 3, 5] {
        for ncon in [0usize, 2] {
            for pop in [8i32, 31, 64, 65] {
                for (nsga, pareto) in [(true, 0.0), (false, 0.0), (false, 0.3)] {
                    for quant in [false, true] {
                        for ints in [false, true] {
                            if ints && (nobj == 5 || pop == 64) {
                                continue;
                            }
                            for seed in 1..=2u64 {
                                let d = 7;
                                let mask =
                                    ints.then(|| (0..d).map(|i| i % 2 == 1).collect::<Vec<bool>>());
                                let mut m = Mode::new(
                                    Fitness::bounded(d, nobj + ncon, &vec![0.0; d], &vec![1.0; d]),
                                    nobj,
                                    ncon,
                                    mask,
                                    &ModeParams {
                                        seed,
                                        popsize: pop,
                                        nsga_update: nsga,
                                        pareto_update: pareto,
                                        ..Default::default()
                                    },
                                );
                                let mut h = Hash::new();
                                for _ in 0..40 {
                                    let xs = m.ask();
                                    h.m(&xs);
                                    let ys: Vec<_> =
                                        xs.iter().map(|x| mo(x, nobj, ncon, quant)).collect();
                                    m.tell(&ys);
                                    h.m(&m.population());
                                }
                                let r = m.result();
                                h.m(&r.x);
                                h.m(&r.y);
                                println!(
                                    "mode|nobj={nobj},ncon={ncon},pop={pop},nsga={nsga},pareto={pareto},quant={quant},ints={ints},seed={seed}|result={:016x}|front={}",
                                    h.0,
                                    r.y.len()
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

fn map() {
    let d = 9;
    for (qd, k, samples) in [
        (2usize, 60usize, 0usize),
        (2, 256, 0),
        (2, 1025, 0),
        (2, 4096, 0),
        (2, 1, 0),
        (2, 64, 2),
        (2, 255, 2),
        (2, 256, 2),
        (2, 300, 3),
        (2, 1024, 2),
        (3, 128, 2),
        (4, 256, 2),
        (8, 128, 2),
    ] {
        for (sbx, chunk, cma) in [
            (true, 32usize, 0usize),
            (false, 32, 0),
            (true, 1, 0),
            (true, 33, 0),
            (true, 16, 3),
        ] {
            if (k == 1 || qd > 2) && chunk != 32 {
                continue;
            }
            for seed in 1..=2u64 {
                let mut rng = Rng::new(seed);
                let (lo, hi) = (vec![0.0; d], vec![1.0; d]);
                let mut a = Archive::new(d, &vec![0.0; qd], &vec![1.0; qd], k, samples, &mut rng);
                a.seed_uniform(&lo, &hi, &mut rng);
                let template = a.clone();
                let f = move |x: &[f64]| {
                    let q = 1.0 + x.iter().map(|v| (v - 0.3).powi(2)).sum::<f64>();
                    let q = if x[3] < 0.1 {
                        (q * 4.0).round() / 4.0
                    } else {
                        q
                    };
                    (
                        q,
                        (0..qd)
                            .map(|i| if i == 0 { x[0] * 1.2 - 0.1 } else { x[i] })
                            .collect::<Vec<f64>>(),
                    )
                };
                map_elites(
                    &mut a,
                    &mut { f },
                    &lo,
                    &hi,
                    &MapElitesParams {
                        generations: if chunk == 1 { 400 } else { 60 },
                        chunk_size: chunk,
                        use_sbx: sbx,
                        cma_generations: cma,
                        ..Default::default()
                    },
                    &mut rng,
                );
                let mut h = Hash::new();
                h.s(a.ys());
                h.m(a.xs());
                h.m(a.descriptors());
                a.counts().iter().for_each(|c| h.u(*c));
                h.f(a.qd_score());
                h.f(a.best_y());
                h.u(a.occupied() as u64);
                let mut q = Rng::new(77 + seed);
                let mut hq = Hash::new();
                for i in 0..4000 {
                    let p: Vec<f64> = (0..qd)
                        .map(|j| match (i + j) % 11 {
                            0 => 0.0,
                            1 => 1.0,
                            2 => -0.25,
                            3 => 1.5,
                            4 => (q.uniform01() * 8.0).round() / 8.0,
                            _ => q.uniform01(),
                        })
                        .collect();
                    hq.u(a.index_of_niche(&p) as u64);
                    hq.u(template.index_of_niche(&p) as u64);
                }
                a.argsort();
                let mut r2 = Rng::new(5);
                h.m(&a.random_xs(k.min(40), 16, &mut r2));
                // The template must be untouched by the run on its clone.
                let mut ht = Hash::new();
                ht.s(template.ys());
                template.counts().iter().for_each(|c| ht.u(*c));
                println!(
                    "map|qd={qd},k={k},samples={samples},sbx={sbx},chunk={chunk},cma={cma},seed={seed}|result={:016x}|lookup={:016x}|template={:016x}|occupied={}|score={:e}",
                    h.0,
                    hq.0,
                    ht.0,
                    a.occupied(),
                    a.qd_score()
                );
            }
        }
    }
}

fn cma() {
    for (name, f) in [
        ("shifted", shifted as F),
        ("rosen", rosen),
        ("ellipsoid", ellipsoid),
        ("holes", holes),
    ] {
        for d in [5usize, 10, 40, 63, 64, 100] {
            for (pop, gap, bounded, repair) in [
                (0, -1, false, false),
                (31, -1, false, false),
                (0, 0, false, false),
                (0, 3, false, false),
                (0, -1, true, false),
                (0, -1, true, true),
                (0, 0, true, true),
            ] {
                for seed in 1..=2u64 {
                    let p = CmaesParams {
                        popsize: pop,
                        update_gap: gap,
                        seed,
                        repair_z: repair,
                        max_evaluations: if d >= 63 { 2500 } else { 4000 },
                        ..Default::default()
                    };
                    let fit = if bounded {
                        Fitness::bounded(d, 1, &vec![-3.0; d], &vec![2.0; d])
                    } else {
                        Fitness::new(d, 1, vec![], vec![])
                    };
                    for arm in ["direct", "ask", "tellx"] {
                        let t = fresh();
                        let obj = traced(f, &t);
                        let mut es = Cmaes::new(fit.clone(), &vec![0.4; d], &[0.3], &p);
                        let r = match arm {
                            "direct" => es.optimize(&obj, 1),
                            "ask" => {
                                for _ in 0..(p.max_evaluations as usize / es.popsize()) {
                                    let xs = es.ask();
                                    if es.tell(&xs.iter().map(|x| obj(x)).collect::<Vec<_>>()) != 0
                                    {
                                        break;
                                    }
                                }
                                es.result()
                            }
                            _ => {
                                for g in 0..(p.max_evaluations as usize / es.popsize() / 2) {
                                    let mut xs = es.ask();
                                    if g % 2 == 0 {
                                        for x in xs.iter_mut() {
                                            x[0] = (x[0] * 16.0).round() / 16.0;
                                        }
                                    }
                                    let ys: Vec<_> = xs.iter().map(|x| obj(x)).collect();
                                    if es.tell_x(&ys, &xs) != 0 {
                                        break;
                                    }
                                }
                                es.result()
                            }
                        };
                        let mut h = Hash::new();
                        h.s(&r.x);
                        h.f(r.y);
                        h.u(r.evaluations);
                        h.u(r.stop as u64);
                        drop(obj);
                        line(
                            "cma",
                            format!(
                                "{name},d={d},pop={pop},gap={gap},bounded={bounded},repair={repair},seed={seed},{arm}"
                            ),
                            h.0,
                            &t,
                            r.y,
                            r.evaluations,
                        );
                    }
                }
            }
        }
    }
}

fn main() {
    let which = std::env::args().nth(1).unwrap_or_default();
    match which.as_str() {
        "bite" => bite(),
        "de" => de(),
        "mode" => mode(),
        "map" => map(),
        "cma" => cma(),
        _ => panic!("bite|de|mode|map|cma"),
    }
}
