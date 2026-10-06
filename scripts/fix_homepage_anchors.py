"""Point every code link in the homepage's algorithm table at its class.

Each table row in docs/index.html names an algorithm (`<code>Name</code>`) and
links its Python and JavaScript source by line number. Line numbers drift
whenever a file above the class changes, so this script finds `class Name` in
each linked file and rewrites the link's `#L<n>` and its text. Run it after
merging anything that moves those classes:

    python scripts/fix_homepage_anchors.py          # rewrite in place
    python scripts/fix_homepage_anchors.py --check  # exit 1 if any link is stale
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "docs" / "index.html"
LINK = re.compile(
    r'(href="https://github\.com/microprediction/humpday/blob/main/)([^"#]+)#L(\d+)(")'
    r"([^>]*>)([^<#]+)#L(\d+)(</a>)"
)


def class_line(path, name):
    # `}class Name` occurs where a previous class body ends on the same line
    pattern = re.compile(rf"^\s*}}?\s*(export\s+)?class\s+{re.escape(name)}\b")
    for number, text in enumerate((ROOT / path).read_text().splitlines(), start=1):
        if pattern.match(text):
            return number
    return None


def main(check):
    rows = re.split(r"(<tr>)", PAGE.read_text())
    stale, missing = [], []
    for i, row in enumerate(rows):
        name = re.search(r"<td><code>(\w+)</code></td>", row)
        if not name:
            continue

        algo = name.group(1)

        def fix(m):
            path, old = m.group(2), int(m.group(3))
            new = class_line(path, algo)
            if new is None:
                missing.append(f"{algo}: no class in {path}")
                return m.group(0)
            if new != old:
                stale.append(f"{algo}: {path} L{old} -> L{new}")
            return (
                f"{m.group(1)}{path}#L{new}{m.group(4)}{m.group(5)}"
                f"{m.group(6)}#L{new}{m.group(8)}"
            )

        rows[i] = LINK.sub(fix, row)
    for line in stale + missing:
        print(line)
    if check:
        return 1 if stale or missing else 0
    PAGE.write_text("".join(rows))
    print(f"{len(stale)} links rewritten, {len(missing)} without a class")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main("--check" in sys.argv))
