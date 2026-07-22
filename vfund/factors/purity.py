"""AST purity gate — structurally reject alphas that could cheat.

Why a gate at all
-----------------
The operator layer makes adding alphas cheap. That is the point, and it is also
the danger: a fast way to write factors is a fast way to write *subtly broken*
factors. VFund's existing look-ahead defence (``tests/test_lookahead.py``) proves
the **engine** cannot see the future. It says nothing about whether a given
alpha's own code reaches around the engine — by indexing a panel directly,
shifting the wrong way, importing something that reads a file, or calling
``eval``.

So alphas are parsed, not trusted. This module inspects an alpha's source as an
AST and refuses anything outside a deliberately small vocabulary. It is a
structural guarantee, checked before the code ever runs.

What is rejected
----------------
* imports of any kind (an alpha needs nothing beyond the operator surface);
* ``eval`` / ``exec`` / ``compile`` / ``__import__`` / ``open`` / ``globals``;
* dunder attribute access (``__class__``, ``__globals__``, … — the usual
  sandbox-escape ladder);
* negative-step slicing such as ``close[::-1]``, which reverses time;
* any call to a name that is not a whitelisted operator or builtin.

What is deliberately *not* claimed
----------------------------------
This is not a security sandbox and does not pretend to be — it guards against
*accidental* look-ahead and sloppy imports in first-party research code, not a
determined attacker. Do not use it to run untrusted third-party alphas.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass
from typing import Callable

from vfund.factors import operators

#: Names an alpha body may call: the operator surface plus arithmetic helpers.
ALLOWED_CALLS: frozenset[str] = frozenset(operators.__all__) | frozenset(
    {"abs", "min", "max", "float", "int", "len"}
)

#: Callables that are never acceptable inside an alpha.
FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {"eval", "exec", "compile", "__import__", "open", "globals", "locals",
     "vars", "getattr", "setattr", "delattr", "input", "exit", "quit"}
)

#: numpy functions an alpha may call as ``np.<fn>(...)``. Every one is
#: element-wise and shape-preserving, so none can move a value between bars —
#: which is what makes them safe. Time-shuffling functions (``roll``, ``flip``,
#: ``sort``, ``argsort``) are absent on purpose: those *could* express
#: look-ahead, so they stay outside the vocabulary.
ALLOWED_NUMPY: frozenset[str] = frozenset(
    {"where", "sign", "abs", "log", "log1p", "log10", "exp", "sqrt", "square",
     "minimum", "maximum", "clip", "isnan", "isfinite", "full", "full_like",
     "zeros_like", "ones_like", "nan_to_num", "power"}
)

#: Modules whose whitelisted functions may be called.
_NUMPY_ALIASES: frozenset[str] = frozenset({"np", "numpy"})


class PurityError(Exception):
    """Raised when an alpha's source violates the purity rules."""


@dataclass(frozen=True)
class Violation:
    """One rule breach, with the line it occurred on."""

    line: int
    rule: str
    detail: str

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"line {self.line}: {self.rule} - {self.detail}"


class _Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: list[Violation] = []

    def _flag(self, node: ast.AST, rule: str, detail: str) -> None:
        self.violations.append(Violation(getattr(node, "lineno", 0), rule, detail))

    def visit_Import(self, node: ast.Import) -> None:
        self._flag(node, "import", "alphas may not import; use the operator surface")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._flag(node, "import", "alphas may not import; use the operator surface")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__") and node.attr.endswith("__"):
            self._flag(node, "dunder", f"dunder attribute access: {node.attr}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in FORBIDDEN_NAMES:
            self._flag(node, "forbidden", f"forbidden name: {node.id}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        fn = node.func
        if isinstance(fn, ast.Name):
            if fn.id in FORBIDDEN_NAMES:
                self._flag(node, "forbidden", f"forbidden call: {fn.id}")
            elif fn.id not in ALLOWED_CALLS:
                self._flag(node, "unknown-call",
                           f"{fn.id}() is not a whitelisted operator")
        elif isinstance(fn, ast.Attribute):
            base = fn.value
            is_np = isinstance(base, ast.Name) and base.id in _NUMPY_ALIASES
            if is_np and fn.attr in ALLOWED_NUMPY:
                pass  # element-wise, cannot move data between bars
            elif is_np:
                self._flag(node, "numpy-call",
                           f"np.{fn.attr}() is not whitelisted; it may reorder time")
            else:
                # df.shift() / arr.roll() and friends bypass the operator
                # vocabulary and its look-ahead guarantees.
                self._flag(node, "method-call",
                           f"method call .{fn.attr}() bypasses the operator surface")
        self.generic_visit(node)

    def visit_Slice(self, node: ast.Slice) -> None:
        step = node.step
        neg = (
            isinstance(step, ast.UnaryOp) and isinstance(step.op, ast.USub)
        ) or (isinstance(step, ast.Constant) and isinstance(step.value, int)
              and step.value < 0)
        if neg:
            self._flag(node, "time-reversal",
                       "negative-step slice reverses time")
        self.generic_visit(node)


def check_source(src: str, *, name: str = "<alpha>") -> list[Violation]:
    """Parse ``src`` and return every purity violation found (empty == clean).

    When the source is a single function definition, only its **body** is
    inspected — a registration decorator such as ``@alpha(...)`` is scaffolding
    around the formula, not part of it.

    Raises:
        PurityError: if the source cannot be parsed at all.
    """
    try:
        tree = ast.parse(textwrap.dedent(src))
    except SyntaxError as exc:  # pragma: no cover - defensive
        raise PurityError(f"{name}: could not parse alpha source: {exc}") from exc

    nodes: list[ast.AST] = [tree]
    if len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef):
        nodes = list(tree.body[0].body)

    v = _Visitor()
    for node in nodes:
        v.visit(node)
    return v.violations


def assert_pure(fn: Callable, *, name: str | None = None) -> None:
    """Raise :class:`PurityError` unless ``fn``'s source passes every rule.

    Args:
        fn: the alpha compute function to inspect.
        name: label used in the error message; defaults to ``fn.__name__``.

    Raises:
        PurityError: listing every violation at once, so a broken alpha is fixed
            in one pass rather than one error at a time.
    """
    label = name or getattr(fn, "__name__", "<alpha>")
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError) as exc:
        raise PurityError(f"{label}: cannot read source to verify purity") from exc
    violations = check_source(src, name=label)
    if violations:
        joined = "\n  ".join(str(v) for v in violations)
        raise PurityError(f"{label} is not a pure alpha:\n  {joined}")
