#!/usr/bin/env python3
"""
Link verification script for HumpDay implementation table.
Checks that all GitHub links in the implementation table are valid and accessible.
"""

import re
import time
from pathlib import Path

import requests


def check_github_link(url):
    """Check if a GitHub link is accessible."""
    try:
        # Convert GitHub blob links to raw content for checking
        if "github.com" in url and "/blob/" in url:
            # Check if the file exists by making a HEAD request
            response = requests.head(url, timeout=10)
            return response.status_code == 200
        return False
    except Exception as e:
        print(f"Error checking {url}: {e}")
        return False


def extract_links_from_html():
    """Extract GitHub links from the implementation table."""
    html_file = Path("/Users/petercotton/github/humpday/docs/index.html")

    if not html_file.exists():
        print("index.html not found")
        return []

    content = html_file.read_text()

    # Find all GitHub links in the implementation table
    github_links = re.findall(
        r'href="(https://github\.com/microprediction/humpday/blob/main/[^"]*)"', content
    )

    return list(set(github_links))  # Remove duplicates


def verify_all_links():
    """Verify all GitHub links in the implementation table."""
    print("HumpDay Implementation Table Link Verification")
    print("=" * 50)

    links = extract_links_from_html()

    if not links:
        print("No GitHub links found in implementation table")
        return

    print(f"Found {len(links)} unique GitHub links to verify\n")

    working_links = []
    broken_links = []

    for i, link in enumerate(links, 1):
        print(f"{i:2d}. Checking: {link.split('/')[-1]}")

        if check_github_link(link):
            print("    ✓ Working")
            working_links.append(link)
        else:
            print("    ✗ Broken")
            broken_links.append(link)

        # Be nice to GitHub's servers
        time.sleep(0.5)

    print("\n" + "=" * 50)
    print(f"SUMMARY: {len(working_links)} working, {len(broken_links)} broken")

    if broken_links:
        print("\nBroken links that need fixing:")
        for link in broken_links:
            print(f"  ✗ {link}")
    else:
        print("\n✓ All implementation table links are working correctly!")


def verify_local_files():
    """Verify that local JavaScript files exist."""
    print("\nLocal JavaScript File Verification")
    print("=" * 50)

    js_files = [
        "docs/js/modules/base-optimizer.js",
        "docs/js/modules/prima-algorithms.js",
        "docs/js/modules/scipy-algorithms.js",
        "docs/js/modules/evolutionary-algorithms.js",
        "docs/js/modules/search-algorithms.js",
        "docs/js/modules/optimizer-factory.js",
        "docs/js/modules/index.js",
    ]

    base_path = Path("/Users/petercotton/github/humpday")

    for js_file in js_files:
        file_path = base_path / js_file
        if file_path.exists():
            size_kb = file_path.stat().st_size / 1024
            print(f"✓ {js_file} ({size_kb:.1f} KB)")
        else:
            print(f"✗ {js_file} - NOT FOUND")


if __name__ == "__main__":
    verify_local_files()
    verify_all_links()
