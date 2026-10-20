//! Small dense kernels for the exact GP.
//!
//! Symmetric and triangular matrices use packed row-major lower-triangular
//! storage: row `i` holds columns `0..=i` at offset `i(i+1)/2`, so every inner
//! loop runs over contiguous memory and appending a row is a `push`. All sums
//! use a fixed accumulation order, so results do not depend on the Rayon pool.

/// Offset of row `i` in packed lower-triangular storage.
#[inline]
pub fn row(i: usize) -> usize {
    i * (i + 1) / 2
}

/// Dot product with eight independent accumulators reduced in a fixed order.
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = [0.0; 8];
    let mut ac = a.chunks_exact(8);
    let mut bc = b.chunks_exact(8);
    for (x, y) in (&mut ac).zip(&mut bc) {
        for k in 0..8 {
            s[k] += x[k] * y[k];
        }
    }
    let mut tail = 0.0;
    for (x, y) in ac.remainder().iter().zip(bc.remainder()) {
        tail += x * y;
    }
    ((s[0] + s[4]) + (s[1] + s[5])) + ((s[2] + s[6]) + (s[3] + s[7])) + tail
}

/// `y += a x`.
#[inline]
pub fn axpy(a: f64, x: &[f64], y: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len());
    for (y, x) in y.iter_mut().zip(x) {
        *y += a * x;
    }
}

/// Lower Cholesky factor `L` of a symmetric positive definite matrix.
#[derive(Clone, Debug)]
pub struct Cholesky {
    n: usize,
    lower: Vec<f64>,
}

impl Cholesky {
    /// Factors the packed lower triangle `a` of an `n x n` matrix after adding
    /// `shift` to its diagonal. `None` if a pivot is not positive and finite.
    pub fn factor(a: &[f64], n: usize, shift: f64) -> Option<Self> {
        debug_assert_eq!(a.len(), row(n));
        let mut lower = a.to_vec();
        for i in 0..n {
            let (head, tail) = lower.split_at_mut(row(i));
            let ri = &mut tail[..=i];
            for j in 0..i {
                let rj = &head[row(j)..row(j) + j + 1];
                ri[j] = (ri[j] - dot(&ri[..j], &rj[..j])) / rj[j];
            }
            let pivot = ri[i] + shift - dot(&ri[..i], &ri[..i]);
            if !(pivot > 0.0 && pivot.is_finite()) {
                return None;
            }
            ri[i] = libm::sqrt(pivot);
        }
        Some(Self { n, lower })
    }

    pub fn dimension(&self) -> usize {
        self.n
    }

    #[inline]
    pub fn row(&self, i: usize) -> &[f64] {
        &self.lower[row(i)..row(i) + i + 1]
    }

    pub fn diagonal(&self, i: usize) -> f64 {
        self.lower[row(i) + i]
    }

    /// `log det(L L^T)`.
    pub fn log_determinant(&self) -> f64 {
        (0..self.n).map(|i| 2.0 * libm::log(self.diagonal(i))).sum()
    }

    /// Solves `L v = b`.
    pub fn forward(&self, b: &[f64]) -> Vec<f64> {
        debug_assert_eq!(b.len(), self.n);
        let mut v = vec![0.0; self.n];
        for i in 0..self.n {
            let r = self.row(i);
            v[i] = (b[i] - dot(&r[..i], &v[..i])) / r[i];
        }
        v
    }

    /// Solves `L^T x = v` in place, by row-wise elimination.
    pub fn backward(&self, x: &mut [f64]) {
        debug_assert_eq!(x.len(), self.n);
        for i in (0..self.n).rev() {
            let r = self.row(i);
            x[i] /= r[i];
            let xi = x[i];
            axpy(-xi, &r[..i], &mut x[..i]);
        }
    }

    /// Solves `L L^T x = b`.
    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        let mut x = self.forward(b);
        self.backward(&mut x);
        x
    }

    /// Appends the row `[l, diagonal]`, an O(n) update once `l = L^{-1} k`
    /// is known.
    pub fn append(&mut self, l: &[f64], diagonal: f64) {
        debug_assert_eq!(l.len(), self.n);
        self.lower.extend_from_slice(l);
        self.lower.push(diagonal);
        self.n += 1;
    }

    /// Packed `L^{-1}`, by row-wise forward elimination.
    pub fn inverse_lower(&self) -> Vec<f64> {
        let n = self.n;
        let mut m = vec![0.0; row(n)];
        let mut acc = vec![0.0; n];
        for i in 0..n {
            let r = self.row(i);
            let acc = &mut acc[..i];
            acc.fill(0.0);
            for (k, &lik) in r[..i].iter().enumerate() {
                axpy(lik, &m[row(k)..row(k) + k + 1], &mut acc[..=k]);
            }
            let inverse = 1.0 / r[i];
            let mi = &mut m[row(i)..row(i) + i + 1];
            for (out, a) in mi[..i].iter_mut().zip(acc.iter()) {
                *out = -a * inverse;
            }
            mi[i] = inverse;
        }
        m
    }
}

/// Row `a` of `(L L^T)^{-1} = M^T M` for packed `M = L^{-1}`, columns `0..=a`.
/// Rows are independent, so callers may compute them in parallel.
pub fn inverse_row(m: &[f64], n: usize, a: usize, out: &mut [f64]) {
    debug_assert_eq!(out.len(), a + 1);
    out.fill(0.0);
    for k in a..n {
        let mk = &m[row(k)..row(k) + k + 1];
        axpy(mk[a], &mk[..=a], out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spd(n: usize) -> Vec<f64> {
        // Packed lower triangle of a well-conditioned SPD matrix.
        let mut a = vec![0.0; row(n)];
        for i in 0..n {
            for j in 0..=i {
                let x = ((i * 7 + j * 3) % 11) as f64 / 11.0;
                a[row(i) + j] = if i == j { n as f64 + 1.0 } else { x - 0.5 };
            }
        }
        a
    }
    fn get(a: &[f64], i: usize, j: usize) -> f64 {
        if j <= i { a[row(i) + j] } else { a[row(j) + i] }
    }

    #[test]
    fn factor_solve_inverse_and_append_agree() {
        for n in [1, 2, 7, 8, 9, 33] {
            let a = spd(n);
            let c = Cholesky::factor(&a, n, 0.0).unwrap();
            let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
            let x = c.solve(&b);
            for (i, bi) in b.iter().enumerate() {
                let ax: f64 = (0..n).map(|j| get(&a, i, j) * x[j]).sum();
                assert!((ax - bi).abs() < 1e-12, "n={n}");
            }
            let m = c.inverse_lower();
            let mut inverse = vec![0.0; row(n)];
            for i in 0..n {
                inverse_row(&m, n, i, &mut inverse[row(i)..row(i) + i + 1]);
            }
            for i in 0..n {
                for j in 0..n {
                    let product: f64 = (0..n).map(|k| get(&a, i, k) * get(&inverse, k, j)).sum();
                    assert!((product - f64::from(i == j)).abs() < 1e-12, "n={n}");
                }
            }
            if n > 1 {
                let head = Cholesky::factor(&a[..row(n - 1)], n - 1, 0.0).unwrap();
                let k: Vec<f64> = (0..n - 1).map(|j| a[row(n - 1) + j]).collect();
                let l = head.forward(&k);
                let mut appended = head.clone();
                appended.append(&l, libm::sqrt(a[row(n) - 1] - dot(&l, &l)));
                for (p, q) in appended.lower.iter().zip(&c.lower) {
                    assert!((p - q).abs() < 1e-12);
                }
            }
        }
        assert!(Cholesky::factor(&[1.0, 2.0, 1.0], 2, 0.0).is_none());
    }

    #[test]
    fn dot_handles_every_remainder() {
        for n in 0..20 {
            let a: Vec<f64> = (0..n).map(|i| i as f64 + 0.5).collect();
            let b: Vec<f64> = (0..n).map(|i| 1.0 - i as f64).collect();
            let expected: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            assert!((dot(&a, &b) - expected).abs() < 1e-9);
        }
    }
}
