import importlib.util
import os

import pytest


def _expected_backend():
    if os.environ.get("HUMPDAY_FORCE_PURE_ARRAY") == "1":
        return "pure"
    return "numpy" if importlib.util.find_spec("numpy") else "pure"


@pytest.fixture(autouse=True)
def _backend_matches_environment():
    """Every test runs on the backend the environment asked for, and leaves it there.

    humpday._array is one module object shared by every optimizer, so a test that reloads it
    onto the other backend and does not put it back silently switches the rest of the session,
    and a run launched to validate the pure backend can finish having validated numpy.
    """
    from humpday import _array

    expected = _expected_backend()
    assert _array.BACKEND == expected, (
        f"test started on the {_array.BACKEND!r} backend but the environment selects "
        f"{expected!r}: an earlier test reloaded humpday._array and did not restore it"
    )
    yield
    assert _array.BACKEND == expected, (
        f"test left humpday._array on the {_array.BACKEND!r} backend; the environment "
        f"selects {expected!r}"
    )
