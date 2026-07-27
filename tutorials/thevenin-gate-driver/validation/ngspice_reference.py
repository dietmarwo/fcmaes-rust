#!/usr/bin/env python3
"""Replay the checked-in validation grid with synchronous libngspice.

This script is deliberately outside the optimization hot path. It provides an
independent implementation for cross-simulator validation and requires the
ngspice shared library (`libngspice0` on Debian/Ubuntu).
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (ROOT / "netlists" / "gate-driver.cir").read_text(encoding="utf-8")
DRIVE_VOLTAGE_V = 10.0
EDGE_START_S = 5.0e-9
TRACE_INDUCTANCE_H = 8.0e-9
BASE_GATE_CAPACITANCE_F = 4.0e-9
SNUBBER_CAPACITANCE_F = 2.0e-9
STEP_S = 50.0e-12
STOP_S = 120.0e-9


class VectorInfo(ctypes.Structure):
    """Subset of sharedspice.h's vector_info used by the harness."""

    _fields_ = [
        ("v_name", ctypes.c_char_p),
        ("v_type", ctypes.c_int),
        ("v_flags", ctypes.c_short),
        ("v_realdata", ctypes.POINTER(ctypes.c_double)),
        ("v_compdata", ctypes.c_void_p),
        ("v_length", ctypes.c_int),
    ]


SEND_CHAR = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p
)
SEND_STAT = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p
)
CONTROLLED_EXIT = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_bool,
    ctypes.c_bool,
    ctypes.c_int,
    ctypes.c_void_p,
)
SEND_DATA = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p
)
SEND_INIT_DATA = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p
)
BG_THREAD_RUNNING = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_bool, ctypes.c_int, ctypes.c_void_p
)


@SEND_CHAR
def _send_char(_message, _identifier, _user):
    return 0


@SEND_STAT
def _send_stat(_message, _identifier, _user):
    return 0


@CONTROLLED_EXIT
def _controlled_exit(status, _immediate, _quit, _identifier, _user):
    if status:
        raise RuntimeError(f"ngspice requested controlled exit with status {status}")
    return 0


@SEND_DATA
def _send_data(_values, _count, _identifier, _user):
    return 0


@SEND_INIT_DATA
def _send_init_data(_values, _identifier, _user):
    return 0


@BG_THREAD_RUNNING
def _bg_thread_running(_running, _identifier, _user):
    return 0


class NgSpice:
    """Minimal synchronous shared-ngspice driver."""

    def __init__(self, library: str):
        self.library = ctypes.CDLL(library)
        self.library.ngSpice_Init.argtypes = [
            SEND_CHAR,
            SEND_STAT,
            CONTROLLED_EXIT,
            SEND_DATA,
            SEND_INIT_DATA,
            BG_THREAD_RUNNING,
            ctypes.c_void_p,
        ]
        self.library.ngSpice_Init.restype = ctypes.c_int
        self.library.ngSpice_Circ.argtypes = [
            ctypes.POINTER(ctypes.c_char_p)
        ]
        self.library.ngSpice_Circ.restype = ctypes.c_int
        self.library.ngSpice_Command.argtypes = [ctypes.c_char_p]
        self.library.ngSpice_Command.restype = ctypes.c_int
        self.library.ngSpice_CurPlot.argtypes = []
        self.library.ngSpice_CurPlot.restype = ctypes.c_char_p
        self.library.ngGet_Vec_Info.argtypes = [ctypes.c_char_p]
        self.library.ngGet_Vec_Info.restype = ctypes.POINTER(VectorInfo)
        result = self.library.ngSpice_Init(
            _send_char,
            _send_stat,
            _controlled_exit,
            _send_data,
            _send_init_data,
            _bg_thread_running,
            None,
        )
        if result != 0:
            raise RuntimeError(f"ngSpice_Init returned {result}")

    def command(self, command: str) -> None:
        result = self.library.ngSpice_Command(command.encode("utf-8"))
        if result != 0:
            raise RuntimeError(f"ngSpice_Command({command!r}) returned {result}")

    def vector(self, plot: str, name: str) -> list[float]:
        qualified = f"{plot}.{name}".encode("utf-8")
        pointer = self.library.ngGet_Vec_Info(qualified)
        if not pointer:
            raise RuntimeError(f"ngspice did not expose {qualified.decode()}")
        vector = pointer.contents
        if not vector.v_realdata or vector.v_length < 1:
            raise RuntimeError(f"{qualified.decode()} is not a non-empty real vector")
        return [vector.v_realdata[index] for index in range(vector.v_length)]

    def simulate(self, netlist: str) -> tuple[list[float], ...]:
        encoded = [line.encode("utf-8") for line in netlist.splitlines()]
        lines = (ctypes.c_char_p * (len(encoded) + 1))()
        for index, line in enumerate(encoded):
            lines[index] = line
        lines[len(encoded)] = None
        result = self.library.ngSpice_Circ(lines)
        if result != 0:
            raise RuntimeError(f"ngSpice_Circ returned {result}")
        self.command("run")
        plot_raw = self.library.ngSpice_CurPlot()
        if not plot_raw:
            raise RuntimeError("ngspice returned no current plot")
        plot = plot_raw.decode("utf-8")
        vectors = tuple(
            self.vector(plot, name)
            for name in ("time", "v(drive)", "v(trace)", "v(gate)")
        )
        self.command("destroy all")
        self.command("remcirc")
        return vectors


def render_netlist(
    resistance_ohm: float, snubber_resistance_ohm: float
) -> str:
    values = {
        "DRIVE_VOLTAGE_V": DRIVE_VOLTAGE_V,
        "EDGE_START_S": EDGE_START_S,
        "RESISTANCE_OHM": resistance_ohm,
        "TRACE_INDUCTANCE_H": TRACE_INDUCTANCE_H,
        "BASE_GATE_CAPACITANCE_F": BASE_GATE_CAPACITANCE_F,
        "SNUBBER_RESISTANCE_OHM": snubber_resistance_ohm,
        "SNUBBER_CAPACITANCE_F": SNUBBER_CAPACITANCE_F,
        "STEP_S": STEP_S,
        "STOP_S": STOP_S,
    }
    rendered = TEMPLATE
    for name, value in values.items():
        rendered = rendered.replace(f"{{{{{name}}}}}", f"{value:.17e}")
    if "{{" in rendered:
        raise ValueError("unresolved netlist template placeholder")
    return rendered


def rising_crossing(
    time: list[float],
    values: list[float],
    threshold: float,
    start_s: float,
) -> float:
    for t0, t1, y0, y1 in zip(
        time[:-1], time[1:], values[:-1], values[1:], strict=True
    ):
        if t1 >= start_s and y0 <= threshold <= y1 and y1 != y0:
            fraction = min(1.0, max(0.0, (threshold - y0) / (y1 - y0)))
            return t0 + fraction * (t1 - t0)
    raise ValueError(f"waveform never crosses {threshold}")


def measure(
    time: list[float],
    drive: list[float],
    trace: list[float],
    gate: list[float],
    resistance_ohm: float,
) -> dict[str, float]:
    lengths = {len(time), len(drive), len(trace), len(gate)}
    if len(lengths) != 1 or len(time) < 3:
        raise ValueError("transient vectors have inconsistent lengths")
    t10 = rising_crossing(time, gate, 0.1 * DRIVE_VOLTAGE_V, EDGE_START_S)
    t90 = rising_crossing(time, gate, 0.9 * DRIVE_VOLTAGE_V, t10)
    maximum = max(
        voltage for sample_time, voltage in zip(time, gate, strict=True)
        if sample_time >= t10
    )
    band = 0.02 * DRIVE_VOLTAGE_V
    outside = [
        sample_time
        for sample_time, voltage in zip(time, gate, strict=True)
        if sample_time >= EDGE_START_S
        and abs(voltage - DRIVE_VOLTAGE_V) > band
    ]
    final_count = min(16, len(gate))
    metrics = {
        "rise_time_ns": (t90 - t10) * 1.0e9,
        "overshoot_percent": max(
            0.0, 100.0 * (maximum - DRIVE_VOLTAGE_V) / DRIVE_VOLTAGE_V
        ),
        "peak_driver_current_a": max(
            abs((source - node) / resistance_ohm)
            for source, node in zip(drive, trace, strict=True)
        ),
        "settling_time_ns": max(
            0.0, ((outside[-1] if outside else EDGE_START_S) - EDGE_START_S)
            * 1.0e9,
        ),
        "final_gate_voltage_v": sum(gate[-final_count:]) / final_count,
        "timepoints": len(time),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("non-finite ngspice measurement")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT / "validation" / "candidates.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "validation" / "ngspice.csv",
    )
    parser.add_argument("--library", default="libngspice.so")
    arguments = parser.parse_args()

    simulator = NgSpice(arguments.library)
    with arguments.candidates.open(newline="", encoding="utf-8") as source:
        candidates = list(csv.DictReader(source))
    fields = [
        "point_id",
        "u_resistance",
        "u_snubber",
        "resistance_ohm",
        "snubber_resistance_ohm",
        "rise_time_ns",
        "overshoot_percent",
        "peak_driver_current_a",
        "settling_time_ns",
        "final_gate_voltage_v",
        "timepoints",
    ]
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            resistance = float(candidate["resistance_ohm"])
            snubber_resistance = float(candidate["snubber_resistance_ohm"])
            vectors = simulator.simulate(
                render_netlist(resistance, snubber_resistance)
            )
            metrics = measure(*vectors, resistance)
            writer.writerow(
                {
                    "point_id": candidate["point_id"],
                    "u_resistance": candidate["u_resistance"],
                    "u_snubber": candidate["u_snubber"],
                    "resistance_ohm": candidate["resistance_ohm"],
                    "snubber_resistance_ohm": candidate[
                        "snubber_resistance_ohm"
                    ],
                    **metrics,
                }
            )
    print(f"wrote {len(candidates)} ngspice reference rows to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
