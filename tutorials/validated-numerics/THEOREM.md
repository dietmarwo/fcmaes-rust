# Mathematical contract and trusted components

This document describes the two fixed verifiers. They reproduce two known
reference validations; they establish no new mathematics. The implementation
follows the
[pinned RadiiPolynomial theorem](https://github.com/OlivierHnt/RadiiPolynomial.jl/blob/704009d7aa2e621c0900c9c158e853cb84c0c253/docs/src/radii_polynomial_approach.md)
and its worked scalar and Cahn–Hilliard examples.

## Scalar equation

For `F(u) = u^3 - 2`, the candidate supplies a binary64 centre `u_bar` and
scalar approximate inverse `A`. The verifier checks `A != 0`, so a fixed point
of `T(u) = u - A*F(u)` is a root of `F`.

Interval arithmetic bounds `Y >= abs(A*F(u_bar))` and the first derivative
defect `Z1 >= sup(abs(1 - A*DF(u)), u in B(u_bar,R))`. The chosen binary64
radius is `R = 2*sup(Y)`. It is an exactly specified radius, not a claim that
this floating multiplication itself is an outward-rounded estimate.

The verifier independently checks

$$
Y+(Z_1-1)R\le0,\qquad Z_1<1,\qquad 0<R\le10^{-10}.
$$

It also checks that the whole ball is positive. First-order `Z1` bounds the
derivative on the entire ball, unlike the second-order centre defect below.
Existence and uniqueness radii are both `R`; no maximal uniqueness claim is made.

## Neumann steady state

The exact target is

$$
u''+\beta(u-u^3-c)=0,\quad x\in[0,1],\qquad
u'(0)=u'(1)=0,\qquad c=0,\quad\beta=1285/32.
$$

Use real, even, 2-periodic functions, represented by Fourier sequences with
frequency exactly pi and `a[-k] = a[k]`:

$$
u(x)=a_0+2\sum_{k\ge1}a_k\cos(k\pi x),\qquad
\|a\|_X=|a_0|+2\sum_{k\ge1}|a_k|.
$$

This is an l1 Banach algebra, and `sup_x abs(u(x)) <= norm(a,X)`. The symmetry
multiplicity is included by RadiiPolynomial's `Ell1()` on `evensym(Fourier(...))`;
the regression test checks the factor two. Thus a radius of at most `1e-10`
in this norm implies the same physical sup-norm error. The positive first-mode
check `a_bar[1] - R/2 > 0` proves the solution is nonconstant and fixes its sign
orientation. It does not identify a global continuation branch.

### Full approximate inverse and tails

The saved candidate has `K=32` coefficients through mode K, plus a finite
approximate inverse matrix. Extend the inverse by the diagonal inverse
Laplacian on modes above K. The zero mode remains in the finite block.

The verifier recomputes:

- `Y = norm(A*F(u_bar),X)` with interval arithmetic, including all modes of
  the cubic residual;
- the finite derivative defect on input modes through `3K+1`, including its
  full output support;
- the bound on all remaining input modes,

$$
Z_{1,\mathrm{tail}}=
\frac{\beta(1+3\|\bar u\|_X^2)}{(\pi(K+2))^2};
$$

- `Z1 = max(Z1_finite,Z1_tail)` and

$$
Z_2=3\beta\|A\|(2\|\bar u\|_X+R),\qquad
\|A\|\le\max(\|A_\mathrm{finite}\|,1/\pi^2).
$$

For the tail bound, multiplication by the degree-2K polynomial `u_bar^2`
shifts frequencies by at most 2K. An input mode above `3K+1` therefore has its
lowest affected frequency at least `K+2`, outside the finite inverse block.
The inverse-Laplacian denominator and the l1 algebra bound control all these
unresolved modes. The finite and tail estimates are both essential.

`R = 10*sup(Y)` is again a chosen exact binary64 radius. The code rechecks,
with guaranteed intervals at that explicit radius,

$$
Y+(Z_1-1)R+\tfrac12 Z_2R^2\le0,\qquad Z_1+Z_2R<1,
\qquad 0<R\le10^{-10}.
$$

It requires `Z1_finite < 1`. Restricting its defect bound to the finite square
block makes `A_finite * DF_finite` invertible by the Neumann-series argument;
therefore the square `A_finite` is injective. The diagonal tail inverse has no
zero entries. This establishes the fixed-point/zero equivalence instead of
assuming a floating matrix inversion was exact.

Although the Laplacian alone is unbounded on l1, the fixed-point map has a
bounded extension there: its tail identity and inverse-Laplacian terms cancel
the unbounded linear part. The remaining nonlinear term is a bounded
polynomial map in the Banach algebra, plus a finite-dimensional linear term.
A fixed point satisfies the full coefficient equation. The tail equation
then gives summable second-derivative coefficients because `u-u^3-c` is in
l1. This supplies a classical C2 solution; the polynomial ODE bootstraps its
regularity. Even periodic symmetry enforces the endpoint Neumann conditions.
The spatially constant chemical potential yields the corresponding stationary
Cahn–Hilliard equation. No dynamical stability conclusion follows.

The two radii fields are both this explicitly checked R. The code does not
assume that the upper endpoint of a library-returned interval satisfies a
strict inequality, and does not enlarge the uniqueness region after checking.

## Arithmetic and replay

The guarantee flag, decorations, boundedness, nonemptiness and inequalities
must all pass. The library success Boolean is an additional check. Candidate
coefficients and inverse entries are explicit binary64 points; they need not
be enclosures of a true solution or inverse. Exact pi and rational equation
constants are enclosed independently. Floating approximation success is never
used as proof evidence.

The verifier runs only in one recorded arithmetic configuration:
IntervalArithmetic's correct rounding, `Float64` bounds, set-based flavor and
fast power, its generic interval matrix product, and RadiiPolynomial's loop
convolution. A startup file or a caller could otherwise switch to
round-to-nearest, and every result would still report itself as guaranteed;
`cap.jl` therefore also requires `--startup-file=no`. The generic matrix product
does not use OpenBLAS, so enclosures do not depend on the CPU's BLAS kernels.
The loaded RadiiPolynomial and IntervalArithmetic must be the source trees
whose hashes the manifest pins, not merely packages of the same version.

Replay reparses the exact problem and candidate, reconstructs the finite/tail
operator and every bound, and compares the complete proof record. Deleting a
tail estimate, changing coefficients, setting a success flag or editing a
bound cannot create an accepted replay. Replay requires identical recorded
environments and exact record equality; cross-platform enclosure variation
can cause conservative rejection.

Trusted components include this theorem-specific adapter, the stated analytic
bounds and upstream theorem implementation, pinned Julia and interval/numerical
dependencies, and their arithmetic/platform assumptions. Package/source hashes
record identity; a writable depot or compromised machine is outside this
trusted local research model. This is an independently rerun numerical
certificate, not a proof checked by a formal proof assistant or by a second
independent interval implementation. `exact_check.py` adds an independent
check of the arithmetic only: it recomputes the retained certificates' bounds
from the same formulas in exact rational arithmetic, with pi enclosed to
`1e-60`, and requires every recorded interval to enclose the exact value.
