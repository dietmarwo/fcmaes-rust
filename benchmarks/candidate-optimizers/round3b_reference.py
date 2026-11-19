#!/usr/bin/env python3
"""Reference code of the round-3b problems and the check of the Rust ports.

The harness evaluates all three round-3b problems in Rust
(`harness/src/round3b.rs`). This file holds faithful, attributed copies of
the two reference functions that exist as code, and compares the harness
values with them through `candidate-harness serve`:

- `pest_control_score`: `_pest_control_score` of Bounce
  (`bounce/util/pest_control.py`), byte-for-byte in its logic; Bounce's
  benchmark calls it with `seed=0`. Casmopolitan's `test_funcs/pest.py`
  has the same function. Required: bit-identical values.
- `cantilever_deflection`: `MixedCantileverBeam._evaluate` of SMT 2.15.1
  (`smt/problems/mixed_cantilever_beam.py`) with its option defaults.
  NumPy evaluates `L**3` with a SIMD power routine on hosts that have one
  (AVX-512 here), which can differ from the C library's `pow` by one unit in
  the last place; the Rust port uses the C library's `pow`. Required:
  bit-identical values against the same formula with `math.pow`, and at
  most 2 units in the last place against the NumPy formula.

The welded beam has no reference code; its definition is checked against
the published solution of Deb and Goyal in `harness/tests/round3b.rs`.

Usage (from this directory, after `cargo build --release -p
candidate-harness --features candidates`):

    python3 round3b_reference.py verify [--pest 500] [--cantilever 200]
        [--harness PATH] [--smt DIR] [--bounce FILE]

`--smt DIR` (a directory that contains the `smt` package, version 2.15.1)
and `--bounce FILE` (Bounce's `pest_control.py`) additionally compare the
copies here with the upstream code bit for bit. Every check raises an
exception; the script never uses `assert`.

License notices of the copied code
----------------------------------

Bounce (`_pest_control_score`, `_pest_spread`):

    MIT License

    Copyright (c) 2024 Leonard Papenmeier

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to permit
    persons to whom the Software is furnished to do so, subject to the
    following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
    NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
    USE OR OTHER DEALINGS IN THE SOFTWARE.

The pest-control model originates in COMBO
(`COMBO/experiments/test_functions/multiple_categorical.py`):

    Copyright (c) 2019, QUVA-Lab, University of Amsterdam
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    1. Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
    CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

SMT (`MixedCantileverBeam`; author P. Saves):

    BSD 3-Clause License

    Copyright (c) 2017, SMT developers
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
    CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

import run_catcmawm as driver

PEST_SEED = 0
SMT_VERSION = "2.15.1"


class ReferenceError(Exception):
    """A reference check failed."""


# ------------------------------------------------- Bounce (MIT), copied

def _pest_spread(curr_pest_frac, spread_rate, control_rate, apply_control):
    if apply_control:
        next_pest_frac = (1.0 - control_rate) * curr_pest_frac
    else:
        next_pest_frac = spread_rate * (1 - curr_pest_frac) + curr_pest_frac
    return next_pest_frac


def pest_control_score(x, seed=None):
    """`_pest_control_score` of Bounce (`bounce/util/pest_control.py`),
    without its `logging.debug` call."""
    U = 0.1
    n_stages = x.size
    n_simulations = 100

    init_pest_frac_alpha = 1.0
    init_pest_frac_beta = 30.0
    spread_alpha = 1.0
    spread_beta = 17.0 / 3.0

    control_alpha = 1.0
    control_price_max_discount = {1: 0.2, 2: 0.3, 3: 0.3, 4: 0.0}
    tolerance_develop_rate = {1: 1.0 / 7.0, 2: 2.5 / 7.0, 3: 2.0 / 7.0, 4: 0.5 / 7.0}
    control_price = {1: 1.0, 2: 0.8, 3: 0.7, 4: 0.5}
    # below two changes over stages according to x
    control_beta = {1: 2.0 / 7.0, 2: 3.0 / 7.0, 3: 3.0 / 7.0, 4: 5.0 / 7.0}

    payed_price_sum = 0
    above_threshold = 0

    if seed is not None:
        init_pest_frac = np.random.RandomState(seed).beta(
            init_pest_frac_alpha, init_pest_frac_beta, size=(n_simulations,)
        )
    else:
        init_pest_frac = np.random.beta(
            init_pest_frac_alpha, init_pest_frac_beta, size=(n_simulations,)
        )
    curr_pest_frac = init_pest_frac
    for i in range(n_stages):
        if seed is not None:
            spread_rate = np.random.RandomState(seed).beta(
                spread_alpha, spread_beta, size=(n_simulations,)
            )
        else:
            spread_rate = np.random.beta(
                spread_alpha, spread_beta, size=(n_simulations,)
            )
        do_control = x[i] > 0
        if do_control:
            if seed is not None:
                control_rate = np.random.RandomState(seed).beta(
                    control_alpha, control_beta[x[i]], size=(n_simulations,)
                )
            else:
                control_rate = np.random.beta(
                    control_alpha, control_beta[x[i]], size=(n_simulations,)
                )
            next_pest_frac = _pest_spread(
                curr_pest_frac, spread_rate, control_rate, True
            )
            # torelance has been developed for pesticide type 1
            control_beta[x[i]] += tolerance_develop_rate[x[i]] / float(n_stages)
            # you will get discount
            payed_price = control_price[x[i]] * (
                1.0
                - control_price_max_discount[x[i]]
                / float(n_stages)
                * float(np.sum(x == x[i]))
            )
        else:
            next_pest_frac = _pest_spread(curr_pest_frac, spread_rate, 0, False)
            payed_price = 0
        payed_price_sum += payed_price
        above_threshold += np.mean(curr_pest_frac > U)
        curr_pest_frac = next_pest_frac

    return payed_price_sum + above_threshold


# --------------------------------------------------- SMT (BSD-3), copied

SMT_LIST_I = [0.0833, 0.139, 0.380, 0.0796, 0.133, 0.363, 0.0859, 0.136, 0.360,
              0.0922, 0.138, 0.369]
SMT_P = 50e3
SMT_E = 200e9


def cantilever_deflection(x):
    """`MixedCantileverBeam._evaluate` of SMT 2.15.1 for rows (category
    index 0..11, L, S) with the default options P = 50e3, E = 200e9."""
    P = SMT_P
    E = SMT_E
    I = np.int64(x[:, 0]) - 1  # noqa: E741
    L = x[:, 1]
    S = x[:, 2]
    Ival = np.array([SMT_LIST_I[i] for i in I])
    y = (P * L**3) / (3 * E * S**2 * Ival)
    return y


def cantilever_deflection_libm(x):
    """The same formula and operation order with the C library's `pow`."""
    return np.array([(SMT_P * math.pow(length, 3.0))
                     / (3 * SMT_E * (area * area) * SMT_LIST_I[int(c) - 1])
                     for c, length, area in x])


# ------------------------------------------------------------- checks

def bits_equal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).view(np.uint64) == \
        np.asarray(b, dtype=np.float64).view(np.uint64)


def ulps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(a, dtype=np.float64).view(np.int64)
                  - np.asarray(b, dtype=np.float64).view(np.int64))


def pest_points(count: int) -> list[np.ndarray]:
    rng = np.random.default_rng(driver.fnv1a64("pest_control") & 0xFFFFFFFF)
    points = [rng.integers(0, 5, 25) for _ in range(count)]
    points += [np.zeros(25, dtype=np.int64), np.full(25, 4), np.arange(25) % 5]
    return points


def cantilever_points(count: int) -> np.ndarray:
    rng = np.random.default_rng(driver.fnv1a64("cantilever_mixed") & 0xFFFFFFFF)
    rows = np.column_stack([rng.integers(0, 12, count), rng.uniform(10, 20, count),
                            rng.uniform(1, 2, count)]).astype(np.float64)
    corners = [[c, length, area] for c in range(12) for length, area in ((10.0, 1.0),
                                                                        (20.0, 2.0))]
    return np.vstack([rows, np.array(corners, dtype=np.float64)])


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ReferenceError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify(harness: str, pest: int, cantilever: int, smt: str | None = None,
           bounce: str | None = None, verbose: bool = True) -> list[str]:
    """Compare the harness with the reference copies (and optionally the
    upstream code); returns the failures."""
    failures = []
    serve = driver.Serve(harness)
    try:
        # Pest control: choices x map to the centers of their intervals.
        xs = pest_points(pest)
        vectors = [(x + 0.5) / 5.0 for x in xs]
        rust = serve.evaluate("pest_control", vectors)
        reference = np.array([pest_control_score(x, seed=PEST_SEED) for x in xs])
        wrong = int(np.sum(~bits_equal(rust, reference)))
        if wrong:
            failures.append(f"pest_control: {wrong} of {len(xs)} values differ")
        if bounce is not None:
            upstream = load_module(bounce, "bounce_pest_control")
            same = bits_equal(reference, [upstream._pest_control_score(x, seed=PEST_SEED)
                                          for x in xs])
            if not same.all():
                failures.append("pest_control: the copy differs from Bounce")
        if verbose:
            print(f"pest_control: {len(xs)} points, harness == Bounce copy bit for bit: "
                  f"{wrong == 0}{'' if bounce is None else '; copy == upstream'}", flush=True)

        # Cantilever: category index k maps to the center of interval k.
        rows = cantilever_points(cantilever)
        vectors = [np.array([(c + 0.5) / 12.0, length, area]) for c, length, area in rows]
        rust = serve.evaluate("cantilever_mixed", vectors)
        libm = cantilever_deflection_libm(rows)
        numpy_values = cantilever_deflection(rows)
        exact = int(np.sum(~bits_equal(rust, libm)))
        distance = int(ulps(rust, numpy_values).max())
        if exact:
            failures.append(f"cantilever_mixed: {exact} values differ from the libm formula")
        if distance > 2:
            failures.append(f"cantilever_mixed: {distance} ulp from the NumPy formula")
        if smt is not None:
            sys.path.insert(0, smt)
            from smt import __version__ as smt_version
            from smt.problems import MixedCantileverBeam
            if smt_version != SMT_VERSION:
                failures.append(f"smt {smt_version} found, {SMT_VERSION} expected")
            upstream = MixedCantileverBeam()(rows).ravel()
            if not bits_equal(upstream, numpy_values).all():
                failures.append("cantilever_mixed: the copy differs from SMT")
        if verbose:
            identical = int(np.sum(bits_equal(rust, numpy_values)))
            print(f"cantilever_mixed: {len(rows)} points, harness == libm formula bit for "
                  f"bit: {exact == 0}; against the NumPy formula {identical} identical, "
                  f"max {distance} ulp{'' if smt is None else '; copy == SMT'}", flush=True)
    finally:
        serve.close()
    return failures


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("verify")
    p.add_argument("--pest", type=int, default=500)
    p.add_argument("--cantilever", type=int, default=200)
    p.add_argument("--harness", default=None)
    p.add_argument("--smt", default=None, help="directory containing the smt package")
    p.add_argument("--bounce", default=None, help="Bounce's bounce/util/pest_control.py")
    args = parser.parse_args(argv)
    failures = verify(args.harness or driver.default_harness(), args.pest, args.cantilever,
                      args.smt, args.bounce)
    if failures:
        for failure in failures:
            print(f"FAILED {failure}", file=sys.stderr)
        return 1
    print("round-3b references verified")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except (ReferenceError, driver.DriverError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
