"""The ``humpday`` console command, and ``python -m humpday``.

pyproject.toml has declared ``humpday = "humpday.__main__:main"`` since the entry point was added,
and there was no such module: an installed wheel put a command on the PATH that raised
ModuleNotFoundError on every invocation. What it should do was never written down, so this is the
smallest command that is worth having -- it reads out the recorded evidence the library is built
around, and does not try to be a second way to run an optimization, which is what the Python API
is for.
"""

from __future__ import annotations

import argparse
import json
import sys

DESCRIPTION = "Read out what humpday has measured about derivative-free optimizers."


def _suggest(args) -> dict:
    from humpday import suggest

    smooth = True if args.smooth else (False if args.rugged else None)
    ranked = suggest(n_dim=args.dim, n_trials=args.trials, smooth=smooth)
    names = [name for _, _, name in ranked][: args.limit]
    scores = {name: score for score, _, name in ranked}
    suite = "surfaces" if smooth else ("engineering" if smooth is False else None)
    return {
        "n_dim": args.dim,
        "n_trials": args.trials,
        "suite": suite,
        "order": names,
        # nan where the ordering is by worst rank across both suites, which no single rating
        # describes; the JSON form says null rather than inventing one.
        "ratings": {n: (None if scores[n] != scores[n] else scores[n]) for n in names},
    }


def _render_suggest(result: dict) -> str:
    suite = {
        "surfaces": "the analytic surfaces",
        "engineering": "the engineering suite",
        None: "both suites, by worst rank",
    }[result["suite"]]
    lines = [f"Ranked at d={result['n_dim']}, budget {result['n_trials']}, on {suite}:"]
    for position, name in enumerate(result["order"], start=1):
        rating = result["ratings"][name]
        lines.append(
            f"{position:3d}  {name}" + (f"  {rating:.0f}" if rating is not None else "")
        )
    if not result["order"]:
        lines.append("  nothing recorded, and no fallback ordering either")
    return "\n".join(lines)


def _recommend(args) -> dict:
    from humpday.eligibility import recommend

    name = recommend(
        n_dim=args.dim,
        n_trials=args.trials,
        eval_time=args.eval_time,
        cost_weight=args.cost_weight,
    )
    return {
        "n_dim": args.dim,
        "n_trials": args.trials,
        "eval_time": args.eval_time,
        "cost_weight": args.cost_weight,
        "recommended": name,
    }


def _render_recommend(result: dict) -> str:
    return result["recommended"]


def _ratings(args) -> dict:
    from humpday import ratings

    dims = ratings.recorded_dimensions() if args.dim is None else [args.dim]
    out = {}
    for n_dim in dims:
        for suite, cell in ratings.cells_at(n_dim, args.trials).items():
            out.setdefault(str(n_dim), {})[suite] = {
                "problems": int(cell.get("problems", 0)),
                "ratings": {
                    n: round(float(r), 1) for n, r in cell.get("ratings", {}).items()
                },
                "timed_out": list(cell.get("timed_out", [])),
            }
    return {"n_trials": args.trials, "recorded": out}


def _render_ratings(result: dict) -> str:
    if not result["recorded"]:
        return "nothing recorded at that dimension"
    lines = []
    for n_dim in sorted(result["recorded"], key=int):
        for suite, cell in sorted(result["recorded"][n_dim].items()):
            best = sorted(cell["ratings"].items(), key=lambda kv: -kv[1])[:5]
            lines.append(
                f"d={n_dim:<4} {suite:<12} {cell['problems']:>3} problems  "
                + ", ".join(f"{n} {r:.0f}" for n, r in best)
            )
    return "\n".join(lines)


def _optimizers(args) -> dict:
    from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS

    names = sorted(PURE_OPTIMIZERS)
    if args.dim is None:
        return {"optimizers": names}

    from humpday.eligibility import min_trials, passes_dim, passes_trials

    out = {}
    for name in names:
        if not passes_dim(name, args.dim):
            out[name] = "dimension cap"
        elif not passes_trials(name, args.dim, args.trials):
            out[name] = f"needs {min_trials(name, args.dim)} evaluations"
        else:
            out[name] = None
    return {"n_dim": args.dim, "n_trials": args.trials, "eligibility": out}


def _render_optimizers(result: dict) -> str:
    if "optimizers" in result:
        return "\n".join(result["optimizers"])
    lines = []
    for name, reason in sorted(result["eligibility"].items()):
        lines.append(f"{name:<26} {'eligible' if reason is None else reason}")
    return "\n".join(lines)


_COMMANDS = {
    "suggest": (_suggest, _render_suggest),
    "recommend": (_recommend, _render_recommend),
    "ratings": (_ratings, _render_ratings),
    "optimizers": (_optimizers, _render_optimizers),
}


def _parser() -> argparse.ArgumentParser:
    from humpday import __version__

    parser = argparse.ArgumentParser(prog="humpday", description=DESCRIPTION)
    parser.add_argument("--version", "-V", action="version", version=__version__)
    parser.add_argument(
        "--json", action="store_true", help="machine-readable output for every command"
    )
    sub = parser.add_subparsers(dest="command")

    suggest = sub.add_parser(
        "suggest",
        help="optimizers for a problem, best first, from the recorded tournament",
    )
    suggest.add_argument("--dim", type=int, required=True, help="problem dimension")
    suggest.add_argument("--trials", type=int, default=100, help="evaluation budget")
    suggest.add_argument("--limit", "-n", type=int, default=10, help="how many to list")
    character = suggest.add_mutually_exclusive_group()
    character.add_argument(
        "--smooth", action="store_true", help="rank on the analytic surfaces"
    )
    character.add_argument(
        "--rugged", action="store_true", help="rank on the engineering suite"
    )

    recommend = sub.add_parser(
        "recommend", help="the single optimizer the recommendation grid picks"
    )
    recommend.add_argument("--dim", type=int, required=True)
    recommend.add_argument("--trials", type=int, default=100)
    recommend.add_argument(
        "--eval-time",
        type=float,
        default=None,
        dest="eval_time",
        help="seconds one objective evaluation costs you",
    )
    recommend.add_argument(
        "--cost-weight",
        default=0.0,
        dest="cost_weight",
        help="0 for the quality-only pick, a positive number, or 'auto'",
    )

    rated = sub.add_parser(
        "ratings", help="what was recorded, and off how many problems"
    )
    rated.add_argument("--dim", type=int, default=None, help="one dimension, or all")
    rated.add_argument("--trials", type=int, default=100)

    roster = sub.add_parser("optimizers", help="the roster, with eligibility if asked")
    roster.add_argument("--dim", type=int, default=None)
    roster.add_argument("--trials", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.command is None:
        parser.print_help()
        return 0

    run, render = _COMMANDS[args.command]
    if args.command == "recommend" and args.cost_weight != "auto":
        try:
            args.cost_weight = float(args.cost_weight)
        except ValueError:
            parser.error("--cost-weight takes a number or 'auto'")

    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
