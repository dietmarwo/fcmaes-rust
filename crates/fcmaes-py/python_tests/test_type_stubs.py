"""The ``.pyi`` stub must describe the extension that is actually built.

A stub that drifts from the binding is worse than no stub: editors and type
checkers report confident, wrong signatures. These tests compare the stub
against the live module so drift fails CI instead of shipping.
"""

from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path

import pytest

import fcmaes_rust
from fcmaes_rust import _fcmaes_ext as ext

STUB = Path(fcmaes_rust.__file__).with_name("_fcmaes_ext.pyi")


def _stub_module() -> ast.Module:
    if not STUB.is_file():
        pytest.skip(f"type stub not installed at {STUB}")
    return ast.parse(STUB.read_text(encoding="utf-8"))


def _defaults(node: ast.FunctionDef) -> dict[str, object]:
    """Literal defaults declared for a stub function, by parameter name."""
    out: dict[str, object] = {}
    args = node.args
    positional = args.posonlyargs + args.args
    for name, default in zip(positional[len(positional) - len(args.defaults) :], args.defaults):
        out[name.arg] = _literal(default)
    for name, default in zip(args.kwonlyargs, args.kw_defaults):
        if default is not None:
            out[name.arg] = _literal(default)
    return out


def _literal(node: ast.expr) -> object:
    """Evaluate the small set of default expressions the stub uses."""
    # ``-np.inf`` / ``np.inf`` are the two non-literal defaults we allow.
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _literal(node.operand)
        return -inner if isinstance(inner, (int, float)) else inner
    if isinstance(node, ast.Attribute) and node.attr == "inf":
        return math.inf
    try:
        return ast.literal_eval(node)
    except ValueError:
        return NotImplemented


def _stub_functions(module: ast.Module) -> dict[str, ast.FunctionDef]:
    found: dict[str, ast.FunctionDef] = {}
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            found[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    key = node.name if item.name == "__init__" else f"{node.name}.{item.name}"
                    found[key] = item
    return found


def _runtime_targets() -> dict[str, object]:
    targets: dict[str, object] = {}
    for name in fcmaes_rust.__all__:
        if name in ("__version__", "native"):
            continue
        obj = getattr(fcmaes_rust, name)
        targets[name] = obj
        if inspect.isclass(obj):
            for member in dir(obj):
                if member.startswith("_"):
                    continue
                attribute = getattr(obj, member)
                if callable(attribute):
                    targets[f"{name}.{member}"] = attribute
    return targets


def test_stub_covers_every_exported_name() -> None:
    declared = _stub_functions(_stub_module())
    missing = sorted(name for name in _runtime_targets() if name not in declared)
    assert not missing, f"names exported by the extension but absent from the stub: {missing}"


def test_stub_declares_no_names_the_extension_lacks() -> None:
    module = _stub_module()
    runtime = set(_runtime_targets())
    extra = []
    for node in module.body:
        if isinstance(node, ast.ClassDef) and not hasattr(ext, node.name):
            extra.append(node.name)
        if isinstance(node, ast.FunctionDef) and not hasattr(ext, node.name):
            extra.append(node.name)
    assert not extra, f"stub declares names the extension does not provide: {extra}"
    assert runtime, "sanity: the extension exported nothing"


def test_stub_parameter_names_match_the_binding() -> None:
    declared = _stub_functions(_stub_module())
    problems = []
    for name, obj in _runtime_targets().items():
        node = declared.get(name)
        if node is None:
            continue
        try:
            signature = inspect.signature(obj)
        except (TypeError, ValueError):
            continue
        runtime_names = [p for p in signature.parameters if p != "self"]
        stub_names = [
            a.arg
            for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            if a.arg != "self"
        ]
        if runtime_names != stub_names:
            problems.append(f"{name}: binding={runtime_names} stub={stub_names}")
    assert not problems, "stub parameter names drifted:\n" + "\n".join(problems)


def test_stub_defaults_match_the_binding_where_introspectable() -> None:
    """Defaults must agree wherever PyO3 can report one.

    PyO3 renders non-literal Rust defaults as ``Ellipsis``; those are exactly
    the values the stub exists to document, so they are checked separately in
    :func:`test_stub_documents_the_defaults_pyo3_hides`.
    """
    declared = _stub_functions(_stub_module())
    problems = []
    for name, obj in _runtime_targets().items():
        node = declared.get(name)
        if node is None:
            continue
        try:
            signature = inspect.signature(obj)
        except (TypeError, ValueError):
            continue
        stub_defaults = _defaults(node)
        for parameter in signature.parameters.values():
            if parameter.default is inspect.Parameter.empty:
                continue
            if parameter.default is Ellipsis:
                continue
            if parameter.name not in stub_defaults:
                problems.append(f"{name}.{parameter.name}: stub declares no default")
                continue
            expected = stub_defaults[parameter.name]
            if expected is NotImplemented:
                continue
            if expected != parameter.default:
                problems.append(
                    f"{name}.{parameter.name}: binding={parameter.default!r} stub={expected!r}"
                )
    assert not problems, "stub defaults drifted:\n" + "\n".join(problems)


HIDDEN_DEFAULTS = {
    "stop_fitness": -math.inf,
    "value_limit": math.inf,
    "stop_hist": -1.0,
    "update_gap": -1,
}


def test_stub_documents_the_defaults_pyo3_hides() -> None:
    """Every ``Ellipsis`` default must be spelled out in the stub."""
    declared = _stub_functions(_stub_module())
    problems = []
    for name, obj in _runtime_targets().items():
        node = declared.get(name)
        if node is None:
            continue
        try:
            signature = inspect.signature(obj)
        except (TypeError, ValueError):
            continue
        stub_defaults = _defaults(node)
        for parameter in signature.parameters.values():
            if parameter.default is not Ellipsis:
                continue
            assert parameter.name in HIDDEN_DEFAULTS, (
                f"{name}.{parameter.name} hides a default this test does not know about; "
                "add it to HIDDEN_DEFAULTS with the value from the Rust signature"
            )
            expected = HIDDEN_DEFAULTS[parameter.name]
            actual = stub_defaults.get(parameter.name, "<missing>")
            if actual != expected:
                problems.append(f"{name}.{parameter.name}: stub={actual!r} expected={expected!r}")
    assert not problems, "stub misreports a hidden default:\n" + "\n".join(problems)


def test_package_is_marked_typed() -> None:
    marker = Path(fcmaes_rust.__file__).with_name("py.typed")
    assert marker.is_file(), "PEP 561 requires a py.typed marker for stubs to be used"
