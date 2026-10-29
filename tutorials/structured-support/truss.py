"""Yield-only LP bounds and smooth sizing, checked against the native FEM."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np


def geometry():
    nodes = np.array([[c * 2.4, r * 2.0] for c in range(6) for r in range(3)])
    members = [
        (a, b)
        for a in range(18)
        for b in range(a + 1, 18)
        if np.linalg.norm(nodes[b] - nodes[a]) <= 5 + 1e-12
    ]
    forces = np.zeros((2, 36))
    forces[0, [17, 23]] = -180000
    forces[1, [16, 22]] = 45000
    forces[1, [17, 23]] = -135000
    free = np.array([i for i in range(36) if i not in (0, 1, 31)])
    return nodes, members, forces, free


def matrices(nodes, members, free):
    lengths = []
    B = np.zeros((len(members), len(free)))
    for k, (a, b) in enumerate(members):
        delta = nodes[b] - nodes[a]
        L = np.linalg.norm(delta)
        lengths.append(L)
        v = np.zeros(36)
        v[2 * a : 2 * a + 2] = -delta / L
        v[2 * b : 2 * b + 2] = delta / L
        B[k] = v[free]
    return np.array(lengths), B


def run(preset):
    sources = Path(__file__).resolve().parents[3]
    root = sources / "fcmaes-rust/tutorials/truss-sizing"
    sys.path.insert(0, str(sources / "LP-IP-craft/tutorials/_support"))
    sys.path.insert(0, str(sources / "SQP-craft/tutorials/_support"))
    from linear_model import LinearModel
    from numerics import Case, plain
    from lp_ip_craft import safe_dual_bound

    nodes, members, loads, free = geometry()
    nominal = nodes.copy()
    catalogue = list(csv.DictReader((root / "results/publication/sections.csv").open()))
    catA = np.array([float(s["area_m2"]) for s in catalogue])
    catI = np.array([float(s["inertia_m4"]) for s in catalogue])
    published = list(
        csv.DictReader((root / "results/publication/so/best_members.csv").open())
    )
    indices = []
    sections = []
    for row in published:
        a = int(row["node_a"])
        b = int(row["node_b"])
        indices.append(int(row["member_index"]))
        sections.append(int(row["section_index"]))
        nodes[a] = [float(row["x_a_m"]), float(row["y_a_m"])]
        nodes[b] = [float(row["x_b_m"]), float(row["y_b_m"])]
    bounds = []
    for coordinates in [nominal, nodes]:
        L, B = matrices(coordinates, members, free)
        m = LinearModel()
        areas = [
            m.var(f"A{k}", 7850 * L[k], upper=float(catA[-1]))
            for k in range(len(members))
        ]
        for load in range(2):
            f = [
                m.var(
                    f"force{load}.{k}", lower=-215e6 * catA[-1], upper=215e6 * catA[-1]
                )
                for k in range(len(members))
            ]
            for d in range(len(free)):
                m.row(
                    f"equilibrium{load}.{d}",
                    {f[k]: B[k, d] for k in range(len(f))},
                    lower=loads[load, free[d]],
                    upper=loads[load, free[d]],
                )
            for k in range(len(f)):
                m.row(f"tension{load}.{k}", {f[k]: 1, areas[k]: -215e6}, upper=0)
                m.row(f"compression{load}.{k}", {f[k]: -1, areas[k]: -215e6}, upper=0)
        r, rec = m.run()
        assert r.status == "optimal", r.status
        cert = safe_dual_bound(m.model(), r.duals.row_duals)
        bounds.append(dict(result=rec, safe_dual_bound=plain(cert)))
    assert abs(bounds[0]["result"]["evaluation"]["objective"] - 249.16995) < 0.001
    topology = [members[i] for i in indices]
    L, B = matrices(nodes, topology, free)
    E = 210e9
    rho = 7850.0
    forces = loads[:, free]
    area0 = catA[sections]

    def fem(area):
        K = B.T @ ((E * area / L)[:, None] * B)
        u = np.linalg.solve(K, forces.T).T
        axial = E * area / L * (u @ B.T)
        return u, axial, K

    u0, f0, _ = fem(area0)
    native_bin = os.environ.get("TRUSS_REPLAY_BIN")
    parity = None
    if native_bin:
        result = subprocess.run(
            [native_bin],
            input=json.dumps(
                dict(nodes=nodes.tolist(), active=list(zip(indices, sections)))
            ),
            text=True,
            capture_output=True,
            check=True,
        )
        parity = json.loads(result.stdout)
        assert parity["status"] == "checked", parity
        assert np.max(np.abs(np.array(parity["displacements"])[:, free] - u0)) < 1e-9
    beta = 0.8
    shape = (1 + beta**2) / (4 * np.pi * (1 - beta**2))
    count = len(L)

    def response(a, derivatives=False):
        area = a * 0.001
        u, axial, K = fem(area)
        stress = axial / area / 215e6
        critical = np.pi**2 * E * shape * area**2 / L**2
        buckling = -axial / critical
        full = np.zeros((2, 36))
        full[:, free] = u
        displacement = np.sum(full.reshape(2, 18, 2) ** 2, axis=2) / 0.05**2
        value = np.r_[stress.ravel(), buckling.ravel(), displacement.ravel()]
        if not derivatives:
            return value
        J = np.zeros((len(value), count))
        for j in range(count):
            dK = E * 0.001 / L[j] * np.outer(B[j], B[j])
            du = np.linalg.solve(K, -dK @ u.T).T
            dforce = E * area / L * (du @ B.T)
            dforce[:, j] += E * 0.001 / L[j] * (u @ B.T)[:, j]
            ds = E / L * (du @ B.T) / 215e6
            db = -dforce / critical
            db[:, j] += 2 * axial[:, j] / critical[j] / a[j]
            dfull = np.zeros_like(full)
            dfull[:, free] = du
            dd = (
                2
                * np.sum(full.reshape(2, 18, 2) * dfull.reshape(2, 18, 2), axis=2)
                / 0.05**2
            )
            J[:, j] = np.r_[ds.ravel(), db.ravel(), dd.ravel()]
        return J

    def fun(a):
        return rho * np.dot(L, a * 0.001) / 1000, response(a)

    def derivative(a):
        return rho * L * 0.001 / 1000, response(a, True)

    lower = np.full(count, 0.1)
    upper = np.full(count, catA[-1] * 1000)
    case = Case(
        lower,
        upper,
        np.r_[np.full(2 * count, -1), np.full(2 * count + 36, -np.inf)],
        np.ones(4 * count + 36),
        fun,
        derivative,
        np.full(count, catA[-1] * 900),
    )
    r, rec = case.run(scaling="jacobian", iterations=500, seconds=120)
    attempts = [rec]
    if r.status != "converged":
        r, rec = case.run(backend="clarabel", iterations=500, seconds=120)
        attempts.append(rec)
    assert r.status == "converged", r.status
    area = np.array(r.candidate) * 0.001
    rounded = np.array([int(np.searchsorted(catA, v - 1e-12)) for v in area])
    assert np.all(rounded < len(catA))
    ur, fr, _ = fem(catA[rounded])
    buck = -fr / (np.pi**2 * E * catI[rounded] / L**2)
    full = np.zeros((2, 36))
    full[:, free] = ur
    rounded_check = dict(
        mass_kg=float(rho * L @ catA[rounded]),
        stress_ratio=float(np.max(np.abs(fr / catA[rounded] / 215e6))),
        buckling_ratio=float(np.max(buck)),
        displacement=float(np.max(np.linalg.norm(full.reshape(2, 18, 2), axis=2))),
    )
    rounded_native = None
    if native_bin:
        reply = subprocess.run(
            [native_bin],
            input=json.dumps(
                dict(nodes=nodes.tolist(), active=list(zip(indices, map(int, rounded))))
            ),
            text=True,
            capture_output=True,
            check=True,
        )
        rounded_native = json.loads(reply.stdout)
        assert rounded_native["status"] == "checked"
        assert (
            np.max(np.abs(np.array(rounded_native["displacements"])[:, free] - ur))
            < 1e-9
        )
    return dict(
        nominal_nodes=nominal,
        moved_nodes=nodes,
        nominal_members=members,
        active_members=topology,
        bounds=bounds,
        published_mass_kg=float(rho * L @ area0),
        published_native_parity=parity,
        rounded_native_parity=rounded_native,
        attempts=attempts,
        continuous=rec,
        continuous_mass_kg=float(r.evaluation.objective * 1000),
        catalogue_rounding=rounded_check,
        bound_scope="Yield only, fixed geometry, areas capped by largest catalogue section and forces by its yield load; nominal and moved-geometry bounds kept separate. Safe binary64 dual bounds apply to these bounded models.",
    )
