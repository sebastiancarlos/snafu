import argparse
import ast
import json
import math
import re
import sys
import textwrap
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from any_llm.types.completion import ChatCompletionMessage

# ANSI terminal colors

BOLD = "\033[1m"
RESET = "\033[0m"
DIM = "\033[90m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"


# --- Symbol extraction ---

# .py deliberately excluded — ast.parse() handles it.
EXTENSION_TO_LANGUAGE = {
    ".rb": "ruby",
    ".cs": "csharp",
    ".java": "java",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".php": "php",
    ".rs": "rust",
    ".go": "go",
}


def split_identifier_into_words(name: str) -> list[str]:
    """Split snake_case / camelCase / PascalCase into lowercase words."""
    s = re.sub(r"_+", " ", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return [w.lower() for w in s.split() if w]


# Symbol-extraction policy
# - The containers declared here will be recursed into.
_INCLUDE_KINDS = {"function", "method", "class", "struct", "interface", "enum", "trait"}
_CONTAINER_KINDS = {"class", "struct", "interface", "enum", "trait", "impl", "module", "namespace"}

# A rename whose new NAN is at most this much worse than the original (delta
# within [-NAN_TOLERANCE, 0)) still goes to the final LLM validation, but is
# reported as UNCHANGED rather than SUCCESS.
NAN_TOLERANCE = 0.2


def extract_python_symbols(source: str) -> list[str]:
    tree = ast.parse(source)
    names: set[str] = set()

    def walk(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(child.name)
            elif isinstance(child, ast.ClassDef):
                names.add(child.name)
                walk(child)

    walk(tree)
    return sorted(names)


def extract_symbols_tree_sitter(source: str, language: str) -> list[str]:
    from tree_sitter_language_pack import ProcessConfig, process

    result = process(source, ProcessConfig(language=language, structure=True))

    def _flatten_structure(items) -> Iterator[str]:
        for item in items:
            kind = str(item.kind).lower()
            if item.name and kind in _INCLUDE_KINDS:
                yield item.name
            if kind in _CONTAINER_KINDS:
                yield from _flatten_structure(item.children)

    return sorted(set(_flatten_structure(result.structure)))


def collect_symbols(path: Path, min_words: int) -> list[str]:
    try:
        source = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as e:
        sys.exit(f"Could not read {path}: {e}")

    if path.suffix == ".py":
        try:
            found = extract_python_symbols(source)
        except SyntaxError as e:
            sys.exit(f"Could not parse {path}: {e}")
    elif language := EXTENSION_TO_LANGUAGE.get(path.suffix):
        try:
            found = extract_symbols_tree_sitter(source, language)
        except Exception as e:
            sys.exit(f"Could not extract symbols from {path} ({language}): {e}")
    else:
        supported = ", ".join(sorted(set(EXTENSION_TO_LANGUAGE)))
        sys.exit(
            f"Unsupported file type: {path.suffix or '(no extension)'} "
            f"(supported: .py, {supported})"
        )

    return sorted(s for s in found if len(split_identifier_into_words(s)) >= min_words)


def read_symbols_file(path: Path) -> list[str]:
    """Read explicitly-listed symbols from a file, one per line."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (UnicodeDecodeError, OSError) as e:
        sys.exit(f"Could not read {path}: {e}")
    symbols = [line.strip() for line in lines if line.strip()]
    if not symbols:
        sys.exit(f"No symbols found in {path}")
    return sorted(set(symbols))


# --- LLM-based estimation step ---

PROMPT_TEMPLATE = """You are an expert at naming and interpreting symbols in codebases.

You are analyzing a single function/symbol name from a codebase, with NO other
context (no function body, no surrounding code, no docstring).

Symbol name: `{symbol}`

List up to 6 plausible, MUTUALLY DISTINCT things this symbol could reasonably
mean or do, based purely on its name. Less than 6 is ok. "Distinct" means: a
reader could not tell which one is correct without reading the implementation.

For each interpretation, give:
- a short description (Up to 15 words. But way less is ok, particularly if you
  want to return a generic description).
- your estimated probability that this is what the symbol actually does, GIVEN
  ONLY THE NAME

If part of the symbol, even just one apparent word within it, has no obvious
meaning of its own, do not force a literal reading. Instead, try to:
- Consider what the most likely acronym or neologism for it would be, given the
  context implied by the rest of the name, and use that for some descriptions
  if it would yield a high enough probability to make the cut.
- treat it as a black-box noun (an opaque token whose meaning is whatever it is
  used for) and use that for some descriptions if it would yield a high enough
  probability to make the cut.

If the symbol name seems to be clear on WHAT it does, but not on WHY or FOR
WHAT REASON it does it, feel free to write interpretations that are concerned
only on the WHAT.

Probabilities must sum to 1.0. If the name is genuinely unambiguous, return a
single interpretation with probability 1.0.

Respond with ONLY valid JSON, no other text, in this format:
{{"interpretations": [{{"description": "...", "probability": 0.0}}, ...]}}
"""


@dataclass
class SymbolAnalysisResult:
    symbol: str
    words: list[str]
    interpretations: list[dict] = field(default_factory=list)
    entropy: float = float("nan")
    nan: float = float("nan")


def chat_completion(
    model: str,
    messages: list[dict[str, Any] | ChatCompletionMessage],
    max_tokens: int,
) -> str:
    """OpenAI-compatible chat-completions call via any-llm."""
    from any_llm import completion
    from any_llm.types.completion import ChatCompletion

    provider: str | None = None
    if ":" in model:
        provider, model = model.split(":", 1)
    elif "/" in model:
        provider, model = model.split("/", 1)
    else:
        provider = "openai"  # bare model name -> OpenAI

    try:
        resp = cast(
            ChatCompletion,
            completion(
                model=model,
                provider=provider,
                messages=messages,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            ),
        )
    except Exception as e:
        sys.exit(f"any-llm error: {e}")

    try:
        return (resp.choices[0].message.content or "").strip()
    except KeyError, IndexError, TypeError, AttributeError:
        sys.exit(f"Unexpected any-llm response: {resp!r}")


def parse_json(text: str) -> dict:
    """Parse a JSON object out of a model response, tolerating fences."""
    text = re.sub(r"^```(json)?|```$", "", text or "", flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def estimate_symbol_name_ambiguity(symbol: str, model: str) -> SymbolAnalysisResult:
    words = split_identifier_into_words(symbol)
    text = chat_completion(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(symbol=symbol)}],
    )

    try:
        data = parse_json(text)
        probs = [max(float(i["probability"]), 1e-9) for i in data["interpretations"]]
    except json.JSONDecodeError, KeyError, ValueError, IndexError:
        return SymbolAnalysisResult(symbol=symbol, words=words)

    total = sum(probs)
    probs = [p / total for p in probs]  # renormalize defensively
    entropy = -sum(p * math.log2(p) for p in probs)

    return SymbolAnalysisResult(
        symbol=symbol,
        words=words,
        interpretations=data["interpretations"],
        entropy=entropy,
        nan=2**entropy,
    )


# --- Interactive confirmation step ---


def prompt_user_to_confirm_interpretation(result: SymbolAnalysisResult) -> str | None:
    """Ask the user which interpretation is correct. Return a description
    string, or None to skip the symbol."""
    print(f"\n{BOLD}{CYAN}Symbol: {result.symbol}{RESET}")
    for i, interp in enumerate(result.interpretations, 1):
        print(
            f"  {BOLD}{i:>2}.{RESET} ({YELLOW}{interp['probability']:.2f}{RESET}) "
            f"{interp['description']}"
        )
    while True:
        try:
            answer = input(
                f"  {BOLD}Correct one?{RESET} {DIM}[number / 'skip' / custom text]{RESET} "
            ).strip()
        except EOFError:
            return None
        if not answer:
            continue  # empty or whitespace-only -> re-ask
        if answer.lower() in ("none", "skip"):
            return None
        if answer.isdigit():
            idx = int(answer)
            if 1 <= idx <= len(result.interpretations):
                return result.interpretations[idx - 1]["description"]
            continue  # out-of-range number -> re-ask
        return answer  # auto-completed custom description


# --- Rename proposal and validation steps ---

PROMPT_RENAME_TEMPLATE = """You are an expert at naming and interpreting symbols in codebases.

The author of a symbol has confirmed its true meaning. Propose a NEW symbol
name that expresses exactly that confirmed meaning, so a reader guessing from
the name alone picks it out immediately.

Original symbol: `{symbol}`

Confirmed meaning (by the author):
{correct}

Guidance:
- Express ONLY this confirmed meaning in the new name; drop other associations.
- Match the original naming style (snake_case, camelCase, PascalCase) and keep
  it concise but descriptive.
- Prefer concrete, unambiguous words; avoid abbreviations or jargon that could
  be misread.
- Prefer to use simpler/more-standard words, all else being equal. For example:
  Unless there's a good reason, prefer "get" over "gather."
- Return a symbol which is LESS AMBIGUOUS than the original one. Meaning that a
  reader should be LESS LIKELY to interpret the symbol in a way other than the
  confirmed meaning, in comparison to the original symbol.
- If the original name already expresses the confirmed meaning clearly, return
  it unchanged rather than inventing something obscure.

Respond with ONLY valid JSON, no other text, in this format:
{{"new_symbol": "..."}}
"""

PROMPT_VALIDATE_TEMPLATE = """You are an expert at naming and interpreting symbols in codebases.

You are validating a proposed rename of a symbol.

Old symbol: `{symbol}`
New symbol: `{new_symbol}`
Confirmed meaning (by the author): {correct}

Top alternative interpretation of the OLD symbol: {old_top}
Top alternative interpretation of the NEW symbol: {new_top}

State whether ALL of the following are true:
1. the new symbol makes sense as a name (coherent, matches meaning)
2. the top alternative interpretation of the new symbol makes sense (coherent,
   plausible, matches meaning).

Answer each honestly. Respond with ONLY valid JSON, no other text, in this
format:
{{"new_symbol_makes_sense": true, "first_alternative_makes_sense": true,
  "reason": "short rationale"}}
"""


def suggest_new_symbol_name(symbol: str, correct: str, model: str) -> str | None:
    """Have the model propose a clearer name for `symbol` given only the
    user-confirmed meaning."""
    text = chat_completion(
        model=model,
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": PROMPT_RENAME_TEMPLATE.format(symbol=symbol, correct=correct),
            }
        ],
    )
    try:
        new = str(parse_json(text)["new_symbol"]).strip()
    except json.JSONDecodeError, KeyError, ValueError:
        return None
    return new or None


def validate_rename(
    symbol: str,
    new_symbol: str,
    correct: str,
    old_top: str,
    new_top: str,
    model: str,
) -> dict:
    """Give the validator everything: both names, the confirmed meaning, and
    each version's top alternative. Returns the verdict dict."""
    text = chat_completion(
        model=model,
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": PROMPT_VALIDATE_TEMPLATE.format(
                    symbol=symbol,
                    new_symbol=new_symbol,
                    correct=correct,
                    old_top=old_top,
                    new_top=new_top,
                ),
            }
        ],
    )
    try:
        data = parse_json(text)
    except json.JSONDecodeError, KeyError, ValueError:
        return {
            "new_symbol_makes_sense": False,
            "first_alternative_makes_sense": False,
            "reason": "validator returned invalid JSON",
        }
    for key in ("new_symbol_makes_sense", "first_alternative_makes_sense"):
        data[key] = bool(data.get(key))
    data.setdefault("reason", "")
    return data


# --- Main logic ---


def main() -> None:
    # --- Argument parsing ---

    parser = argparse.ArgumentParser(
        description=(
            'Compute and improve "Name Ambiguity Numbers (NAN)" for symbols in source code.'
        ),
        epilog=(
            f"{BOLD}{BLUE}Env vars:{RESET}\n"
            f"    {BOLD}{GREEN}OPENAI_API_KEY{RESET}     OpenAI key (required for OpenAI models)\n"
            f"    {BOLD}{GREEN}OPENAI_BASE_URL{RESET}    optional; base URL, e.g. https://my-host/v1\n"
            "    Other providers read their own env vars (ANTHROPIC_API_KEY, ...).\n"
            "    See https://docs.mozilla.ai/any-llm/providers/ for the full list."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("file", type=Path, nargs="?", help="source file to analyze")
    parser.add_argument(
        "--symbols-file",
        type=Path,
        help=(
            "read symbols from this file, one per line, instead of extracting "
            "them from a source file (used as-is, --min-words not applied)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only extract and list symbols, then exit (no LLM calls)",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=2,
        help="only score symbols with >= N words (default 2)",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="model to use (default gpt-4o-mini)")
    parser.add_argument("--limit", type=int, default=None, help="cap number of symbols to process")
    args = parser.parse_args()

    # --- Obtain symbols ---

    if args.symbols_file:
        symbols = read_symbols_file(args.symbols_file)
    elif args.file:
        symbols = collect_symbols(args.file, args.min_words)
    else:
        parser.error("provide either a source file or --symbols-file")
    if args.limit:
        symbols = symbols[: args.limit]

    if not symbols:
        print(f"{YELLOW}No multi-word symbols found.{RESET}")
        return

    print(f"{BOLD}Symbols ({len(symbols)}):{RESET}")
    for sym in symbols:
        print(f"  {CYAN}{sym}{RESET}")

    if args.dry_run:
        return

    # --- Score every symbol ---

    print(
        f"{BOLD}Round 1: estimating ambiguity for each symbol...{RESET}",
        file=sys.stderr,
    )
    results = []
    for i, sym in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] scoring {CYAN}{sym}{RESET}...", file=sys.stderr)
        results.append(estimate_symbol_name_ambiguity(sym, args.model))

    results.sort(key=lambda r: r.symbol)

    # --- prompt user to confirm correct interpretation per symbol ---
    print(
        f"{BOLD}\nRound 2: confirm the correct interpretation for each symbol.{RESET}",
        file=sys.stderr,
    )
    confirmed: dict[str, str] = {}
    for r in results:
        if not r.interpretations:
            continue
        desc = prompt_user_to_confirm_interpretation(r)
        if desc:
            confirmed[r.symbol] = desc

    # --- propose clearer names for each symbol, based on the confirmed meaning ---

    print(f"{BOLD}\nRound 3: proposing clearer names...{RESET}", file=sys.stderr)
    renames: dict[str, str] = {}
    for r in [x for x in results if x.symbol in confirmed]:
        print(f"[{CYAN}{r.symbol}{RESET}] proposing rename...", file=sys.stderr)
        new = suggest_new_symbol_name(r.symbol, confirmed[r.symbol], args.model)
        if new and new != r.symbol:
            renames[r.symbol] = new

    # --- score each new symbol name ---
    print(f"{BOLD}\nRound 4: scoring proposed names...{RESET}", file=sys.stderr)
    scored: dict[str, tuple[str, SymbolAnalysisResult]] = {}
    for sym, new in renames.items():
        print(f"[{CYAN}{sym}{RESET}] scoring {CYAN}{new}{RESET}...", file=sys.stderr)
        scored[sym] = (new, estimate_symbol_name_ambiguity(new, args.model))

    # Drop new names whose score got worse beyond the tolerance
    by_name = {r.symbol: r for r in results}
    candidates: dict[str, tuple[str, SymbolAnalysisResult]] = {}
    for sym, (new, nr) in scored.items():
        old = by_name[sym]
        if math.isnan(old.nan) or math.isnan(nr.nan):
            continue  # unscorable
        if nr.nan - old.nan <= NAN_TOLERANCE:
            candidates[sym] = (new, nr)

    # ---- final LLM validation of each surviving new name ---

    print(f"{BOLD}\nRound 5: validating renames...{RESET}", file=sys.stderr)
    validations: dict[str, tuple[SymbolAnalysisResult, dict]] = {}
    for sym, (new, nr) in candidates.items():
        old = by_name[sym]
        old_top = old.interpretations[0]["description"] if old.interpretations else "(none)"
        new_top = nr.interpretations[0]["description"] if nr.interpretations else "(none)"
        delta = old.nan - nr.nan
        print(
            f"[{CYAN}{sym}{RESET}] validating -> {CYAN}{new}{RESET} "
            f"(delta {YELLOW}{delta:+.2f}{RESET})...",
            file=sys.stderr,
        )
        verdict = validate_rename(old.symbol, new, confirmed[sym], old_top, new_top, args.model)
        validations[sym] = (nr, verdict)

    # --- present results ---

    print(f"\n{DIM}{'=' * 100}{RESET}")
    print(f"{BOLD}{MAGENTA}RESULTS{RESET}")
    print(f"{DIM}{'=' * 100}{RESET}")
    for r in results:
        if r.symbol not in validations:
            continue
        new_result, verdict = validations[r.symbol]
        new = candidates[r.symbol][0]
        nan_old, nan_new = r.nan, new_result.nan
        if math.isnan(nan_old) or math.isnan(nan_new):
            continue
        delta = nan_old - nan_new
        ok = all(verdict[k] for k in ("new_symbol_makes_sense", "first_alternative_makes_sense"))
        improved = delta > 0
        if ok and improved:
            status = f"{GREEN}SUCCESS{RESET}"
            color = GREEN
        elif ok:
            status = f"{YELLOW}UNCHANGED{RESET}"
            color = YELLOW
        else:
            status = f"{RED}FAILED validation{RESET}"
            color = RED
        print(f"\n{BOLD}{CYAN}{r.symbol}{RESET} -> {BOLD}{CYAN}{new}{RESET}")
        print(f"  {'Name Ambiguity Number (NAN):':<28}  {nan_old:5.2f} -> {nan_new:5.2f}")
        print(f"  {'NAN delta:':<28}  {color}{delta:+5.2f}{RESET} [{status}]")
        prefix = "  Validation reasoning: "
        reason_lines = textwrap.wrap(verdict["reason"], width=100 - len(prefix))
        print(f"{prefix}{reason_lines[0]}")
        for line in reason_lines[1:]:
            print(f"{' ' * len(prefix)}{line}")

    successes = 0
    unchanged = 0
    for sym, (nr, verdict) in validations.items():
        if not all(verdict[k] for k in ("new_symbol_makes_sense", "first_alternative_makes_sense")):
            continue
        if by_name[sym].nan - nr.nan > 0:
            successes += 1
        else:
            unchanged += 1

    print(f"\n{DIM}{'-' * 100}{RESET}")
    print(
        f"{CYAN}Renames proposed: {len(renames)}{RESET}  |  "
        f"{YELLOW}passed numerical filter: {len(candidates)}{RESET}"
    )
    print(
        f"{GREEN}fully validated as success: {successes}{RESET}  |  "
        f"{YELLOW}validated as unchanged: {unchanged}{RESET}"
    )
    print(f"{DIM}{'=' * 100}{RESET}")


if __name__ == "__main__":
    main()
