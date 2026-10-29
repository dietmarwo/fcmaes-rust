"""Thomson energy: native global search and native SQP with pair-work accounting."""

import sys
import numpy as np
from sources import require, sibling
import fcmaes_rust as fc
from sqp_craft import solve, SqpOptions, Options, check_derivatives


def reported_candidate(case, native, start):
    """Keep the raw audit, but report energy only at a checked sphere point."""
    raw = (
        case.check(native.candidate, native.multipliers)
        if native.candidate is not None
        else None
    )
    point = np.array(
        native.candidate if native.candidate is not None else start
    ).reshape(-1, 3)
    projected = raw is None or not raw["feasible"]
    if projected:
        norms = np.linalg.norm(point, axis=1)
        if np.any(norms < 1e-12) or not np.all(np.isfinite(norms)):
            point = np.array(start).reshape(-1, 3)
            norms = np.linalg.norm(point, axis=1)
        point = point / norms[:, None]
    accepted = case.check(point.ravel()) if projected else raw
    require(accepted["feasible"])
    return raw, point, accepted, projected


def run(preset):
    sys.path.insert(0, str(sibling("SQP-craft", "tutorials", "_support")))
    from numerics import Case, plain

    rng = np.random.default_rng(42)
    N = 60 if preset == "study" else 12
    n = 3 * N
    pair_count = N * (N - 1) // 2
    counter = {"traversals": 0, "invalid": 0}

    def kernel(x):
        x = np.asarray(x).reshape(N, 3)
        g = np.zeros_like(x)
        energy = 0.0
        for i in range(N):
            for j in range(i):
                counter["traversals"] += 1
                d = x[i] - x[j]
                r = np.sqrt(d @ d)
                if r < 1e-12:
                    raise ValueError("charge collision")
                energy += 1 / r
                delta = -d / r**3
                g[i] += delta
                g[j] -= delta
        return energy, g.ravel()

    def fun(x):
        return kernel(x)[0], np.sum(x.reshape(N, 3) ** 2, axis=1)

    def derivative(x):
        J = np.zeros((N, n))
        for i in range(N):
            J[i, 3 * i : 3 * i + 3] = 2 * x[3 * i : 3 * i + 3]
        return kernel(x)[1], J

    x0 = rng.normal(size=(N, 3))
    x0 /= np.linalg.norm(x0, axis=1)[:, None]
    case = Case(
        np.full(n, -1.1),
        np.full(n, 1.1),
        np.ones(N),
        np.ones(N),
        fun,
        derivative,
        x0.ravel(),
    )
    require(check_derivatives(case.spec, case, x0.ravel()).passed)
    call_cap = 3600 if preset == "study" else 480
    arms = []

    def local(start, cap):
        before = counter["traversals"]
        r = solve(
            case.spec,
            case,
            start,
            method="sqp",
            options=Options(threads=1),
            method_options=SqpOptions(
                qp_solver="active-set",
                max_iterations=400,
                max_callback_calls=cap,
                time_limit_seconds=120,
            ),
        )
        work = counter["traversals"] - before
        audit, point, accepted, projected = reported_candidate(case, r, start)
        # A converged attempt that fails the independent first-order check is
        # retained and flagged, not allowed to stop the other arms.
        if r.status == "converged" and not (audit and audit.get("first_order")):
            audit = dict(audit or {}, converged_without_independent_first_order=True)
        return r, audit, work, point, accepted, projected

    local_runs = []
    for k in range(3):
        start = x0.copy() if k == 0 else rng.normal(size=(N, 3))
        start /= np.linalg.norm(start, axis=1)[:, None]
        local_runs.append(local(start.ravel(), call_cap // 3))
    r, audit, _, point, accepted, projected = min(
        local_runs, key=lambda z: z[4]["objective"]
    )
    arms.append(
        dict(
            arm="SQP multistart",
            native=plain(r),
            native_check=audit,
            check=accepted,
            points=point,
            projected_after_infeasible_attempt=projected,
            starts=[
                dict(
                    native=plain(rr),
                    native_check=aa,
                    work=ww,
                    points=xx,
                    check=cc,
                    projected_after_infeasible_attempt=pp,
                )
                for rr, aa, ww, xx, cc, pp in local_runs
            ],
            search_pair_traversals=sum(v[2] for v in local_runs),
        )
    )

    def black(x):
        p = np.asarray(x).reshape(N, 3)
        norm = np.linalg.norm(p, axis=1)
        if np.any(norm < 1e-12):
            return 1e20
        try:
            return kernel((p / norm[:, None]).ravel())[0]
        except ValueError:
            counter["invalid"] += 1
            return 1e20

    for arm, budget in [("global", call_cap), ("hybrid", call_cap // 2)]:
        before = counter["traversals"]
        res = fc.optimize_bite(
            black, x0.ravel(), [-1] * n, [1] * n, seed=42, max_evaluations=budget
        )
        work = counter["traversals"] - before
        x = np.array(res[0]).reshape(N, 3)
        x /= np.linalg.norm(x, axis=1)[:, None]
        global_audit = case.check(x.ravel())
        require(global_audit["feasible"])
        record = dict(
            arm=arm,
            native_global=plain(res),
            search_pair_traversals=work,
            check=global_audit,
            energy=global_audit["objective"],
        )
        if arm == "hybrid":
            rr, aa, ww, x, cc, pp = local(x.ravel(), call_cap - budget)
            record.update(
                native_local=plain(rr),
                native_check=aa,
                check=cc,
                projected_after_infeasible_attempt=pp,
                energy=cc["objective"],
                search_pair_traversals=work + ww,
            )
        record["points"] = x
        arms.append(record)
    # Only classify curvature at a checked first-order point. Projected points
    # have useful feasible energies but inherit no native KKT/multiplier claim.
    first_order = [a for a in arms if a["check"].get("first_order")]
    if not first_order:
        return dict(
            charges=N,
            pair_count=pair_count,
            requested_kernel_call_cap=call_cap,
            arms=arms,
            tangent_eigenvalues=[],
            hessian_status="unrun_no_first_order_point",
            invalid_collisions=counter["invalid"],
            reference=49.1652530576288 if N == 12 else 1543.830400976,
            scope="Feasible normalized candidates only; projected points have no first-order claim. Common callback caps, actual pair work differs; audits excluded.",
        )
    selected = min(first_order, key=lambda a: a["check"]["objective"])
    x = np.array(selected["points"])
    basis = []
    for i, p in enumerate(x):
        _, _, vh = np.linalg.svd(p.reshape(1, 3))
        b = np.zeros((n, 2))
        b[3 * i : 3 * i + 3] = vh[1:].T
        basis.append(b)
    Z = np.hstack(basis)
    H = np.zeros((n, n))
    eps = 1e-5
    for j in range(n):
        e = np.zeros(n)
        e[j] = eps
        H[:, j] = (kernel(x.ravel() + e)[1] - kernel(x.ravel() - e)[1]) / (2 * eps)
    native = selected.get("native", selected.get("native_local"))
    H -= np.diag(np.repeat(2 * np.array(native["multipliers"]["constraints"]), 3))
    eigen = np.linalg.eigvalsh(Z.T @ ((H + H.T) / 2) @ Z)
    # Classify with a relative tolerance: rotations give three zero modes.
    tol = 1e-6 * float(np.max(np.abs(eigen)))
    zero_modes = int(np.sum(np.abs(eigen) <= tol))
    if np.any(eigen < -tol):
        verdict = "not_a_local_minimum"
    elif zero_modes == 3:
        verdict = "local_minimum"
    else:
        verdict = "inconclusive"
    return dict(
        charges=N,
        pair_count=pair_count,
        requested_kernel_call_cap=call_cap,
        arms=arms,
        tangent_eigenvalues=eigen,
        curvature=dict(
            tolerance=tol,
            zero_modes=zero_modes,
            smallest_nonzero=float(np.min(eigen[np.abs(eigen) > tol])),
            verdict=verdict,
        ),
        hessian_status="checked_first_order_point",
        hessian_arm=selected["arm"],
        invalid_collisions=counter["invalid"],
        reference=49.1652530576288 if N == 12 else 1543.830400976,
        scope="Feasible candidates only. Common callback caps; actual search pair traversals include convergence differences; independent checks and projection audits excluded. Descriptive, not an equal-consumed-work ranking.",
    )
