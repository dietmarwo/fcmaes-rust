#!/usr/bin/env python3
"""Recheck the retained certificates in exact rational arithmetic, without Julia.

Independent of RadiiPolynomial and IntervalArithmetic: every binary64 number in
the candidate and certificate is converted to a Fraction exactly, and pi enters
only through a rational enclosure of width 1e-60. The script recomputes Y, Z1
(finite part and tail), the norm of A, Z2, the radii polynomial and the
contraction bound with the formulas of THEOREM.md, and requires that

- every recorded interval encloses the exactly computed value, and
- every theorem condition holds exactly at the recorded radius.

It checks the arithmetic of the retained evidence; it does not replace the
verifier, and it uses the same theorem and bound formulas.
Usage: exact_check.py [EVIDENCE_DIRECTORY]
"""
from fractions import Fraction as Fr
import json
from pathlib import Path
import struct
import sys

PI_DIGITS = "314159265358979323846264338327950288419716939937510582097494459"
PI_LO = Fr(int(PI_DIGITS[:61]), 10**60)
PI_HI = PI_LO + Fr(1, 10**60)
PI2 = (PI_LO * PI_LO, PI_HI * PI_HI)
TARGET = Fr(1, 10**10)


def exact(word):
    """The exact rational value of a binary64 hex word."""
    return Fr(struct.unpack(">d", bytes.fromhex(word))[0])


def recorded(interval):
    return exact(interval["lo"]), exact(interval["hi"])


def pi2_linear(p, q):
    """Exact range of p*pi^2 + q over the enclosure of pi^2."""
    a, b = p * PI2[0] + q, p * PI2[1] + q
    return (a, b) if a <= b else (b, a)


def absolute(lo, hi):
    if lo >= 0:
        return lo, hi
    if hi <= 0:
        return -hi, -lo
    return Fr(0), max(-lo, hi)


class Report:
    def __init__(self):
        self.failures = []

    def encloses(self, name, record, value):
        lo, hi = record
        ok = lo <= value[0] and value[1] <= hi
        print(f"{name:28s} recorded [{float(lo):.17g}, {float(hi):.17g}] exact ≈ {float(value[1]):.17g}"
              f"{'' if ok else '  NOT ENCLOSED'}")
        ok or self.failures.append(name + " is not enclosed by its recorded interval")

    def holds(self, name, ok):
        print(f"{name:28s} {'holds' if ok else 'FAILS'}")
        ok or self.failures.append(name + " fails")


def load(directory, problem):
    folder = Path(directory) / problem
    return (json.loads((folder / "verification.json").read_text()),
            json.loads((folder / "candidate.json").read_text()))


def scalar(directory, report):
    v, c = load(directory, "scalar-cuberoot")
    u, A = exact(c["coefficients"][0]), exact(c["inverse_rows"][0][0])
    bounds, R = v["proof"]["bounds"], exact(v["proof"]["existence_radius"])
    Y = abs(A * (u**3 - 2))
    # |1 - 3 A x^2| over the ball is largest at an end point (monotone for x > 0).
    Z1 = max(abs(1 - 3 * A * (u - R) ** 2), abs(1 - 3 * A * (u + R) ** 2))
    P = Y + (Z1 - 1) * R
    print("scalar cube root")
    report.encloses("Y", recorded(bounds["Y"]), (Y, Y))
    report.encloses("Z1 over the ball", recorded(bounds["Z1"]), (Z1, Z1))
    report.encloses("P(R)", recorded(v["proof"]["polynomial_at_radius"]), (P, P))
    report.holds("R = 2 sup(Y)", R == 2 * exact(bounds["Y"]["hi"]))
    report.holds("P(R) <= 0", P <= 0)
    report.holds("Z1 < 1", Z1 < 1)
    report.holds("A != 0", A != 0)
    report.holds("ball is positive", u - R > 0)
    report.holds("0 < R <= 1e-10", 0 < R <= TARGET)


def cahn_hilliard(directory, report):
    v, c = load(directory, "cahn-hilliard")
    K, beta = 32, Fr(1285, 32)
    u = [exact(h) for h in c["coefficients"]]
    A = [[exact(h) for h in row] for row in c["inverse_rows"]]  # A[row][column]
    weight = lambda k: 1 if k == 0 else 2
    bounds, R = v["proof"]["bounds"], exact(v["proof"]["existence_radius"])

    def lattice(a):  # an even sequence on the full integer lattice
        return {s * k: x for k, x in enumerate(a) if x for s in (1, -1)}

    def convolve(x, y):
        z = {}
        for i, xi in x.items():
            for j, yj in y.items():
                z[i + j] = z.get(i + j, 0) + xi * yj
        return z

    U = lattice(u)
    U2 = convolve(U, U)
    U3 = convolve(U2, U)
    norm_u = abs(u[0]) + 2 * sum(abs(x) for x in u[1:])
    print("Cahn-Hilliard steady state")

    # Y = ||A F(u)||, F_k = -(k pi)^2 u_k + beta (u - u^3)_k with c = 0, k <= 3K;
    # the inverse Laplacian acts on modes above K.
    G = [beta * (U.get(k, 0) - U3.get(k, 0)) for k in range(3 * K + 1)]
    Y = [Fr(0), Fr(0)]
    for r in range(K + 1):
        p = -sum(A[r][i] * i * i * u[i] for i in range(K + 1))
        q = sum(A[r][i] * G[i] for i in range(K + 1))
        lo, hi = absolute(*pi2_linear(p, q))
        Y[0] += weight(r) * lo
        Y[1] += weight(r) * hi
    for k in range(K + 1, 3 * K + 1):
        g = abs(G[k]) / (k * k)
        Y[0] += 2 * g / PI2[1]
        Y[1] += 2 * g / PI2[0]
    report.encloses("Y", recorded(bounds["Y"]), Y)

    # Z1 finite: input modes up to 3K+1 and every output mode up to 5K+1.
    V = {m: beta * ((1 if m == 0 else 0) - 3 * U2.get(m, 0)) for m in range(-2 * K, 2 * K + 1)}
    multiply = lambda i, j: V.get(i, 0) if j == 0 else V.get(i - j, 0) + V.get(i + j, 0)
    finite = [Fr(0), Fr(0)]
    for j in range(3 * K + 2):
        column = [Fr(0), Fr(0)]
        for r in range(K + 1):
            p = A[r][j] * j * j if j <= K else Fr(0)
            q = (1 if r == j else 0) - sum(A[r][i] * multiply(i, j) for i in range(K + 1))
            lo, hi = absolute(*pi2_linear(p, q))
            column[0] += weight(r) * lo
            column[1] += weight(r) * hi
        for k in range(K + 1, 5 * K + 2):
            m = abs(multiply(k, j)) / (k * k)
            column[0] += 2 * m / PI2[1]
            column[1] += 2 * m / PI2[0]
        finite = [max(finite[0], column[0] / weight(j)), max(finite[1], column[1] / weight(j))]
    report.encloses("Z1 finite part", recorded(bounds["Z1_finite"]), finite)
    tail = [beta * (1 + 3 * norm_u**2) / (K + 2) ** 2 / PI2[1], beta * (1 + 3 * norm_u**2) / (K + 2) ** 2 / PI2[0]]
    report.encloses("Z1 tail", recorded(bounds["Z1_tail"]), tail)
    Z1 = [max(finite[0], tail[0]), max(finite[1], tail[1])]
    report.encloses("Z1", recorded(bounds["Z1"]), Z1)
    finite_norm = max(sum(weight(i) * abs(A[i][j]) for i in range(K + 1)) / weight(j) for j in range(K + 1))
    norm_A = [max(finite_norm, 1 / PI2[1]), max(finite_norm, 1 / PI2[0])]
    report.encloses("norm of A", recorded(bounds["norm_A"]), norm_A)
    Z2 = [3 * beta * n * (2 * norm_u + R) for n in norm_A]
    report.encloses("Z2", recorded(bounds["Z2"]), Z2)
    P = [Y[i] + (Z1[i] - 1) * R + Z2[i] * R * R / 2 for i in (0, 1)]
    report.encloses("P(R)", recorded(v["proof"]["polynomial_at_radius"]), P)
    contraction = [Z1[i] + Z2[i] * R for i in (0, 1)]
    report.encloses("Z1 + Z2 R", recorded(v["proof"]["contraction_at_radius"]), contraction)
    report.holds("R = 10 sup(Y)", float(R) == 10 * float(exact(bounds["Y"]["hi"])))
    report.holds("P(R) <= 0", P[1] <= 0)
    report.holds("Z1 + Z2 R < 1", contraction[1] < 1)
    report.holds("Z1 finite part < 1", finite[1] < 1)
    report.holds("first mode a1 - R/2 > 0", u[1] - R / 2 > 0)
    report.holds("0 < R <= 1e-10", 0 < R <= TARGET)


def main():
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "evidence"
    report = Report()
    scalar(directory, report)
    cahn_hilliard(directory, report)
    if report.failures:
        print("FAILED: " + "; ".join(report.failures))
        return 1
    print("all recorded intervals enclose the exact values, and every condition holds exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
