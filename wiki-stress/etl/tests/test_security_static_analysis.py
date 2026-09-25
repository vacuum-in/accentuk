"""Static enforcement that dump content can never reach code/shell execution.

design.md and openspec/project.md both state that this package never
executes Wiktionary templates, Lua, or dump-provided code. This test makes
that a checked invariant rather than a claim: any of these tokens appearing
in src/ukstress means either a real security regression or a legitimate new
use that must be reviewed and, if safe, added to the narrow allow-list below
with a comment explaining why.
"""

from __future__ import annotations

import re
from pathlib import Path

FORBIDDEN_PATTERNS = {
    "eval(": re.compile(r"\beval\("),
    "exec(": re.compile(r"\bexec\("),
    "subprocess": re.compile(r"\bsubprocess\b"),
    "os.system(": re.compile(r"\bos\.system\("),
    "os.popen(": re.compile(r"\bos\.popen\("),
    "__import__(": re.compile(r"\b__import__\("),
    # The bare builtin compile(), which can turn a string into executable
    # code for later exec()/eval(). Deliberately excludes re.compile(...)
    # and regex.compile(...), which this codebase uses throughout for
    # ordinary pattern compilation.
    "compile(": re.compile(r"(?<!re\.)(?<!regex\.)\bcompile\("),
}

# Empty by design. Add an entry only alongside a code review that confirms
# the match cannot be reached with dump-controlled input, e.g.
# {"src/ukstress/some_module.py": {"eval("}}.
ALLOWED_MATCHES: dict[str, set[str]] = {}


def _source_files() -> list[Path]:
    package_root = Path(__file__).resolve().parents[1] / "src" / "ukstress"
    return sorted(package_root.rglob("*.py"))


def test_no_dynamic_code_execution_primitives_in_source() -> None:
    violations: list[str] = []
    for path in _source_files():
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(Path(__file__).resolve().parents[1]).as_posix()
        allowed = ALLOWED_MATCHES.get(relative, set())
        for name, pattern in FORBIDDEN_PATTERNS.items():
            if name in allowed:
                continue
            for match in pattern.finditer(text):
                line_number = text.count("\n", 0, match.start()) + 1
                violations.append(f"{relative}:{line_number}: forbidden pattern {name!r}")
    assert not violations, "\n".join(violations)


def test_sql_identifiers_are_never_built_from_dump_content() -> None:
    """Every f-string that looks like SQL must not embed a variable that
    could plausibly hold dump-derived text (page titles, template names,
    wikitext). This is a heuristic guard, not a full data-flow analysis: it
    flags any f-string containing "SELECT"/"INSERT"/"UPDATE"/"DELETE" with an
    interpolated name outside a small, reviewed allow-list of known-safe
    identifiers (hardcoded table names, not dump content).
    """
    sql_keyword = re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE)\b", re.IGNORECASE)
    fstring_with_interpolation = re.compile(
        r'f(["\'])(?:(?!\1).)*\{[^}]+\}(?:(?!\1).)*\1', re.DOTALL
    )
    known_safe_identifiers = {"table", "rank_filter"}

    violations: list[str] = []
    for path in _source_files():
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(Path(__file__).resolve().parents[1]).as_posix()
        for match in fstring_with_interpolation.finditer(text):
            snippet = match.group(0)
            if not sql_keyword.search(snippet):
                continue
            interpolated = re.findall(r"\{([^}!:]+)", snippet)
            unexpected = [
                name.strip()
                for name in interpolated
                if name.strip() not in known_safe_identifiers
            ]
            if unexpected:
                line_number = text.count("\n", 0, match.start()) + 1
                violations.append(
                    f"{relative}:{line_number}: f-string SQL interpolates "
                    f"{unexpected!r}, expected only {sorted(known_safe_identifiers)!r}"
                )
    assert not violations, "\n".join(violations)
