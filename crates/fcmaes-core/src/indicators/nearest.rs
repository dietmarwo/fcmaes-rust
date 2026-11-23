//! Exact Euclidean nearest-neighbor means for GD and IGD, after shape validation.
const LEAF: usize = 20;
type Point<const D: usize> = [f64; D];
fn square<const D: usize>(p: &Point<D>, r: &[f64]) -> f64 {
    p.iter().zip(r).map(|(&a, &b)| (a - b) * (a - b)).sum()
}
fn build<const D: usize>(rows: &mut [Point<D>], axis: usize) {
    if rows.len() <= LEAF {
        return;
    }
    let (left, _, right) =
        rows.select_nth_unstable_by(rows.len() / 2, |a, b| a[axis].total_cmp(&b[axis]));
    build(left, (axis + 1) % D);
    build(right, (axis + 1) % D);
}
fn nearest<const D: usize>(rows: &[Point<D>], axis: usize, r: &[f64], best: &mut f64) {
    if rows.len() <= LEAF {
        for p in rows {
            *best = best.min(square(p, r));
        }
        return;
    }
    let mid = rows.len() / 2;
    let p = &rows[mid];
    *best = best.min(square(p, r));
    let dx = r[axis] - p[axis];
    let left = &rows[..mid];
    let right = &rows[mid + 1..];
    let (near, far) = if dx < 0. {
        (left, right)
    } else {
        (right, left)
    };
    let axis = (axis + 1) % D;
    nearest(near, axis, r, best);
    if dx * dx < *best {
        nearest(far, axis, r, best);
    }
}
fn tree<const D: usize>(sources: &[Vec<f64>], targets: &[Vec<f64>]) -> f64 {
    let mut rows: Vec<Point<D>> = targets
        .iter()
        .map(|p| std::array::from_fn(|i| p[i]))
        .collect();
    build(&mut rows, 0);
    sources
        .iter()
        .map(|r| {
            let mut best = f64::INFINITY;
            nearest(&rows, 0, r, &mut best);
            best.sqrt()
        })
        .sum::<f64>()
        / sources.len() as f64
}
fn sorted_2d(sources: &[Vec<f64>], targets: &[Vec<f64>]) -> f64 {
    let mut rows: Vec<_> = targets.iter().map(|p| [p[0], p[1]]).collect();
    rows.sort_unstable_by(|a, b| a[0].total_cmp(&b[0]));
    let monotone = rows.windows(2).all(|w| w[0][1] >= w[1][1]);
    sources
        .iter()
        .map(|r| {
            let pivot = rows.partition_point(|p| p[0] < r[0]);
            let mut best = f64::INFINITY;
            let mut visit = |p: &[f64; 2], left: bool| {
                let x = r[0] - p[0];
                let dx = x * x;
                if dx >= best {
                    return false;
                }
                let y = r[1] - p[1];
                let distance = dx + y * y;
                if monotone && distance >= best && ((left && y <= 0.) || (!left && y >= 0.)) {
                    return false;
                }
                best = best.min(distance);
                true
            };
            for p in rows[..pivot].iter().rev() {
                if !visit(p, true) {
                    break;
                }
            }
            for p in &rows[pivot..] {
                if !visit(p, false) {
                    break;
                }
            }
            best.sqrt()
        })
        .sum::<f64>()
        / sources.len() as f64
}

fn extreme_mean(sources: &[Vec<f64>], targets: &[Vec<f64>], scale: f64) -> f64 {
    let mut mean = 0.;
    for (i, r) in sources.iter().enumerate() {
        let best = targets
            .iter()
            .map(|p| {
                p.iter().zip(r).fold(0f64, |n, (&a, &b)| {
                    let dx = a - b;
                    n.hypot(if dx.is_finite() {
                        dx * scale
                    } else {
                        a * scale - b * scale
                    })
                })
            })
            .fold(f64::INFINITY, f64::min);
        if !best.is_finite() {
            return best;
        }
        mean += (best - mean) / (i + 1) as f64;
    }
    mean / scale
}
pub(super) fn mean(sources: &[Vec<f64>], targets: &[Vec<f64>], dim: usize) -> f64 {
    // For very few queries a second scan of a large target set costs more than
    // indexing could save. Retain the direct path at ordinary distance scales.
    if let [source] = sources {
        // Validation has already established the row width. A single 3D query
        // needs no tree and pays for one square root over the whole target set.
        let mean = if let [x, y, z] = source[..] {
            targets
                .iter()
                .map(|p| {
                    let (dx, dy, dz) = (x - p[0], y - p[1], z - p[2]);
                    dx * dx + dy * dy + dz * dz
                })
                .fold(f64::INFINITY, f64::min)
                .sqrt()
        } else {
            super::mean_min_distance(sources, targets, super::euclidean)
        };
        if (1e-100..1e100).contains(&mean) {
            return mean;
        }
    } else if sources.len() <= 3 {
        let mean = super::mean_min_distance(sources, targets, super::euclidean);
        if (1e-100..1e100).contains(&mean) {
            return mean;
        }
    }
    // Squaring can lose a tiny distance or overflow a large one. Use hypot for
    // such inputs. A distance above MAX may still contribute to a finite mean.
    if sources.iter().chain(targets).flatten().any(|x| {
        let a = x.abs();
        a > 1e120 || (a != 0. && a < 1e-120)
    }) {
        let mean = extreme_mean(sources, targets, 1.);
        return if mean.is_finite() || sources.len() == 1 {
            mean
        } else {
            extreme_mean(sources, targets, 2f64.powi(-34))
        };
    }
    if sources.len() >= 8 {
        match dim {
            2 if targets.len() >= 40 => return sorted_2d(sources, targets),
            3 if targets.len() >= 128 => return tree::<3>(sources, targets),
            5 if targets.len() >= 256 => return tree::<5>(sources, targets),
            _ => {}
        }
    }
    sources
        .iter()
        .map(|r| {
            targets
                .iter()
                .map(|p| {
                    p.iter()
                        .zip(r)
                        .map(|(&a, &b)| (a - b) * (a - b))
                        .sum::<f64>()
                })
                .fold(f64::INFINITY, f64::min)
                .sqrt()
        })
        .sum::<f64>()
        / sources.len() as f64
}
