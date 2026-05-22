#!/usr/bin/env python3
"""
Release helper script for humpday.

Usage:
    python scripts/release.py --version 0.8.1 --type patch
    python scripts/release.py --version 0.9.0 --type minor
    python scripts/release.py --version 1.0.0 --type major
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def update_version_in_pyproject(version: str) -> None:
    """Update version in pyproject.toml"""
    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()

    # Update version line
    new_content = re.sub(
        r'^version = "[^"]*"',
        f'version = "{version}"',
        content,
        flags=re.MULTILINE
    )

    pyproject_path.write_text(new_content)
    print(f"Updated version to {version} in pyproject.toml")


def update_version_in_init(version: str) -> None:
    """Update version in __init__.py if it exists"""
    init_path = Path("humpday/__init__.py")
    if init_path.exists():
        content = init_path.read_text()

        # Check if __version__ exists
        if "__version__" in content:
            new_content = re.sub(
                r'^__version__ = "[^"]*"',
                f'__version__ = "{version}"',
                content,
                flags=re.MULTILINE
            )
        else:
            # Add version if it doesn't exist
            lines = content.split('\n')
            # Insert after docstring or at beginning
            insert_pos = 0
            for i, line in enumerate(lines):
                if '"""' in line:
                    # Find end of docstring
                    for j in range(i+1, len(lines)):
                        if '"""' in lines[j]:
                            insert_pos = j + 1
                            break
                    break

            lines.insert(insert_pos, f'__version__ = "{version}"')
            new_content = '\n'.join(lines)

        init_path.write_text(new_content)
        print(f"Updated version to {version} in __init__.py")


def run_command(cmd: list[str]) -> bool:
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(cmd)}")
        print(f"Error: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Release helper for humpday")
    parser.add_argument("--version", required=True, help="New version number")
    parser.add_argument("--type", choices=["patch", "minor", "major"],
                       required=True, help="Release type")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without making changes")

    args = parser.parse_args()

    if args.dry_run:
        print(f"DRY RUN: Would update version to {args.version}")
        return

    # Update version files
    update_version_in_pyproject(args.version)
    update_version_in_init(args.version)

    # Run tests
    print("Running tests...")
    if not run_command(["python", "-m", "pytest", "tests/", "-v"]):
        print("Tests failed! Aborting release.")
        sys.exit(1)

    # Build package
    print("Building package...")
    if not run_command(["python", "-m", "build"]):
        print("Build failed! Aborting release.")
        sys.exit(1)

    # Check package
    print("Checking package...")
    if not run_command(["python", "-m", "twine", "check", "dist/*"]):
        print("Package check failed! Aborting release.")
        sys.exit(1)

    print(f"""
Release preparation complete for version {args.version}!

Next steps:
1. Review changes: git diff
2. Commit changes: git add -A && git commit -m "Release {args.version}"
3. Create tag: git tag v{args.version}
4. Push: git push && git push --tags
5. Create GitHub release to trigger PyPI deployment

Or test first:
1. Upload to Test PyPI: twine upload --repository testpypi dist/*
2. Test install: pip install --index-url https://test.pypi.org/simple/ humpday=={args.version}
""")


if __name__ == "__main__":
    main()