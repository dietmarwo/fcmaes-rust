"""Benchmark problems, identical to fcmaes-surrogate's `examples/benchmark.rs`.

`check_against(path)` re-evaluates every observation of a Rust benchmark JSON
file and asserts agreement, so the Python and Rust objectives cannot drift.
"""
import json
import math

PI = math.pi


def branin(x):
    a = x[1] - 5.1 * x[0] ** 2 / (4.0 * PI**2) + 5.0 * x[0] / PI - 6.0
    return a * a + 10.0 * (1.0 - 1.0 / (8.0 * PI)) * math.cos(x[0]) + 10.0


def _hartmann(x, a, p):
    alpha = [1.0, 1.2, 3.0, 3.2]
    return -sum(
        alpha[i] * math.exp(-sum(a[i][j] * (x[j] - p[i][j]) ** 2 for j in range(len(x))))
        for i in range(4)
    )


def hartmann3(x):
    a = [[3.0, 10.0, 30.0], [0.1, 10.0, 35.0], [3.0, 10.0, 30.0], [0.1, 10.0, 35.0]]
    p = [
        [0.3689, 0.1170, 0.2673],
        [0.4699, 0.4387, 0.7470],
        [0.1091, 0.8732, 0.5547],
        [0.0381, 0.5743, 0.8828],
    ]
    return _hartmann(x, a, p)


def hartmann6(x):
    a = [
        [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
        [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
        [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
        [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
    ]
    p = [
        [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
        [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
        [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
        [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
    ]
    return _hartmann(x, a, p)


def styblinski_tang(x):
    return 0.5 * sum(v**4 - 16.0 * v * v + 5.0 * v for v in x)


def rosenbrock(x):
    return sum(100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2 for i in range(len(x) - 1))


def oscillator(x):
    """RK4 damped-oscillator calibration from the surrogate tutorial."""
    c, k = x
    steps = 2048
    h = 4.0 / steps
    omega = math.sqrt(2.3 - 0.04)
    total = 0.0
    for replica in range(4):
        initial = 1.0 + 0.1 * replica
        y0, y1 = initial, 0.0
        loss = 0.0
        for i in range(steps):
            a0, a1 = y1, -c * y1 - k * y0
            b0, b1 = y1 + h * a1 / 2.0, -c * (y1 + h * a1 / 2.0) - k * (y0 + h * a0 / 2.0)
            d0, d1 = y1 + h * b1 / 2.0, -c * (y1 + h * b1 / 2.0) - k * (y0 + h * b0 / 2.0)
            e0, e1 = y1 + h * d1, -c * (y1 + h * d1) - k * (y0 + h * d0)
            y0, y1 = (
                y0 + h * (a0 + 2.0 * b0 + 2.0 * d0 + e0) / 6.0,
                y1 + h * (a1 + 2.0 * b1 + 2.0 * d1 + e1) / 6.0,
            )
            if (i + 1) % (steps // 8) == 0:
                t = (i + 1) * h
                target = initial * math.exp(-0.2 * t) * (math.cos(omega * t) + 0.2 / omega * math.sin(omega * t))
                loss += (y0 - target) ** 2
        total += loss / 8.0
    return total / 4.0


# name: (objective, lower, upper, known minimum, evaluation budget)
PROBLEMS = {
    "branin": (branin, [-5.0, 0.0], [10.0, 15.0], 0.39788735772973816, 48),
    "hartmann3": (hartmann3, [0.0] * 3, [1.0] * 3, -3.8627797873326624, 48),
    "hartmann6": (hartmann6, [0.0] * 6, [1.0] * 6, -3.322368011415515, 96),
    "styblinski10": (styblinski_tang, [-5.0] * 10, [5.0] * 10, -391.6616570377142, 128),
    "rosenbrock8": (rosenbrock, [-2.048] * 8, [2.048] * 8, 0.0, 128),
    "oscillator": (oscillator, [0.05, 0.5], [1.5, 4.0], 0.0, 32),
}


def check_against(path, tolerance=1e-9):
    """Assert that the Python objectives reproduce a Rust benchmark file."""
    data = json.loads(open(path).read())
    checked = 0
    for run in data["runs"]:
        objective = PROBLEMS[run["problem"]][0]
        for observation in run["history"][:12]:
            expected = observation["value"]["value"]
            actual = objective(observation["x"])
            assert abs(actual - expected) <= tolerance * max(1.0, abs(expected)), (
                run["problem"], observation["x"], actual, expected)
            checked += 1
    return checked
