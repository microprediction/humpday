"""
Simple test to debug JavaScript loading
"""

import os
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
JS_SURFACES_PATH = REPO_ROOT / "docs/js/surfaces.js"


def test_js_loading():
    js_code = f"""
// Load the surfaces implementation
const fs = require('fs');
const surfacesCode = fs.readFileSync('{JS_SURFACES_PATH}', 'utf8');
eval(surfacesCode);

// Test that TestSurfaces is defined
console.log('TestSurfaces defined:', typeof TestSurfaces !== 'undefined');
console.log('Available functions:', Object.keys(TestSurfaces));

// Test a simple sphere function call
const sphereFunc = TestSurfaces.sphere();
const result = sphereFunc([0.1, 0.2]);
console.log('Sphere([0.1, 0.2]) =', result);
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
        f.write(js_code)
        temp_file = f.name

    try:
        result = subprocess.run(
            ["node", temp_file], capture_output=True, text=True, timeout=10
        )
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)
        print("returncode:", result.returncode)
    finally:
        os.unlink(temp_file)


if __name__ == "__main__":
    test_js_loading()
