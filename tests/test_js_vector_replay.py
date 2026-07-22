"""JS optimizers replay the Python transition vectors bit-for-bit.

JS_EXACT lists the optimizers whose JavaScript implementations have been
upgraded to statement-level twins of the Python `_run()` generators,
drawing from the shared portable PCG32 stream. The runner drives each
one via suggestNext/receiveUpdate and compares every point and value as
IEEE-754 bit patterns against parity/transition_vectors.json. The list
grows batch by batch until it covers the whole roster.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
RUNNER = Path(__file__).parent / "js_vector_replay_runner.js"

JS_EXACT = [
    "RandomSearch",
    "GridSearch",
    "Rechenberg",
]


@pytest.mark.skipif(not NODE, reason="node not on PATH")
def test_js_replays_transition_vectors_bit_exactly():
    result = subprocess.run(
        [NODE, str(RUNNER), ",".join(JS_EXACT)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    verdicts = json.loads(result.stdout)
    assert len(verdicts) == 4 * len(JS_EXACT)
    failures = [v for v in verdicts if not v["ok"]]
    assert not failures, failures
