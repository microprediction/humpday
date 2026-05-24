#!/usr/bin/env python3
"""
Comprehensive test suite for HumpDay repository and website.
Tests Python implementation, JavaScript modules, website functionality, and links.
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
import requests
from urllib.parse import urljoin

class HumpDayTester:
    def __init__(self):
        self.base_path = Path('/Users/petercotton/github/humpday')
        self.docs_path = self.base_path / 'docs'
        self.results = {
            'python_tests': {},
            'javascript_tests': {},
            'website_tests': {},
            'link_tests': {},
            'file_integrity': {}
        }
        self.server_process = None
        self.server_port = 8083

    def test_python_implementation(self):
        """Test Python optimization algorithms."""
        print("🐍 TESTING PYTHON IMPLEMENTATION")
        print("=" * 50)

        try:
            # Test basic import
            sys.path.insert(0, str(self.base_path))
            from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS, get_optimizer

            print(f"✓ Successfully imported {len(PURE_OPTIMIZERS)} Python algorithms")

            # Test each algorithm
            def sphere_function(x):
                return sum(xi**2 for xi in x)

            working_algorithms = []
            failed_algorithms = []

            for name in PURE_OPTIMIZERS.keys():
                try:
                    optimizer_func = get_optimizer(name)
                    if optimizer_func:
                        # Quick test with minimal evaluations
                        best_val, best_x = optimizer_func(sphere_function, n_dim=2, n_trials=10)
                        if isinstance(best_val, (int, float)) and len(best_x) == 2:
                            working_algorithms.append(name)
                            print(f"  ✓ {name}: Working (result: {best_val:.6f})")
                        else:
                            failed_algorithms.append(name)
                            print(f"  ✗ {name}: Invalid result format")
                    else:
                        failed_algorithms.append(name)
                        print(f"  ✗ {name}: get_optimizer returned None")

                except Exception as e:
                    failed_algorithms.append(name)
                    print(f"  ✗ {name}: Error - {e}")

            self.results['python_tests'] = {
                'total_algorithms': len(PURE_OPTIMIZERS),
                'working': len(working_algorithms),
                'failed': len(failed_algorithms),
                'working_list': working_algorithms,
                'failed_list': failed_algorithms
            }

            print(f"\nPython Summary: {len(working_algorithms)}/{len(PURE_OPTIMIZERS)} algorithms working")

        except Exception as e:
            print(f"✗ Python implementation test failed: {e}")
            self.results['python_tests'] = {'error': str(e)}

    def test_file_integrity(self):
        """Test that all required files exist and have reasonable sizes."""
        print("\n📁 TESTING FILE INTEGRITY")
        print("=" * 50)

        required_files = {
            # HTML pages
            'docs/index.html': (10000, 50000),  # 10-50KB
            'docs/contest.html': (10000, 100000),  # 10-100KB
            'docs/test-modular.html': (1000, 20000),  # 1-20KB
            'docs/contest-modular.html': (5000, 50000),  # 5-50KB

            # JavaScript modules
            'docs/js/modules/base-optimizer.js': (1000, 5000),  # 1-5KB
            'docs/js/modules/prima-algorithms.js': (30000, 80000),  # 30-80KB
            'docs/js/modules/scipy-algorithms.js': (8000, 20000),  # 8-20KB
            'docs/js/modules/evolutionary-algorithms.js': (25000, 60000),  # 25-60KB
            'docs/js/modules/search-algorithms.js': (5000, 15000),  # 5-15KB
            'docs/js/modules/optimizer-factory.js': (2000, 10000),  # 2-10KB
            'docs/js/modules/index.js': (2000, 10000),  # 2-10KB

            # Support files
            'docs/js/stochastic_surfaces.js': (5000, 30000),  # 5-30KB
            'docs/js/contest-controller.js': (30000, 100000),  # 30-100KB

            # Python files
            'humpday/optimizers/alloptimizers.py': (2000, 10000),  # 2-10KB
            'humpday/optimizers/prima_algorithms.py': (5000, 30000),  # 5-30KB
            'humpday/optimizers/scipy_algorithms.py': (5000, 30000),  # 5-30KB
            'humpday/optimizers/evolutionary_algorithms.py': (5000, 50000),  # 5-50KB
        }

        file_status = {}
        for file_path, (min_size, max_size) in required_files.items():
            full_path = self.base_path / file_path
            if full_path.exists():
                size = full_path.stat().st_size
                if min_size <= size <= max_size:
                    print(f"  ✓ {file_path} ({size/1024:.1f} KB)")
                    file_status[file_path] = {'status': 'ok', 'size': size}
                else:
                    print(f"  ⚠ {file_path} ({size/1024:.1f} KB) - Size outside expected range")
                    file_status[file_path] = {'status': 'size_warning', 'size': size}
            else:
                print(f"  ✗ {file_path} - NOT FOUND")
                file_status[file_path] = {'status': 'missing', 'size': 0}

        self.results['file_integrity'] = file_status

        missing_files = [f for f, s in file_status.items() if s['status'] == 'missing']
        if missing_files:
            print(f"\n⚠ {len(missing_files)} files missing!")
        else:
            print(f"\n✓ All {len(required_files)} required files present")

    def start_local_server(self):
        """Start a local HTTP server for testing."""
        print(f"\n🌐 STARTING LOCAL SERVER ON PORT {self.server_port}")
        print("=" * 50)

        try:
            # Kill any existing server on this port
            subprocess.run(['pkill', '-f', f'http.server {self.server_port}'],
                         capture_output=True, cwd=self.docs_path)
            time.sleep(1)

            # Start new server
            self.server_process = subprocess.Popen(
                ['python3', '-m', 'http.server', str(self.server_port), '--bind', '127.0.0.1'],
                cwd=self.docs_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Give server time to start
            time.sleep(2)

            # Test if server is responding
            try:
                response = requests.get(f'http://127.0.0.1:{self.server_port}/', timeout=5)
                if response.status_code == 200:
                    print(f"✓ Server started successfully on http://127.0.0.1:{self.server_port}")
                    return True
                else:
                    print(f"✗ Server responded with status {response.status_code}")
                    return False
            except requests.exceptions.RequestException as e:
                print(f"✗ Server not responding: {e}")
                return False

        except Exception as e:
            print(f"✗ Failed to start server: {e}")
            return False

    def test_website_pages(self):
        """Test that all website pages load correctly."""
        print("\n🌐 TESTING WEBSITE PAGES")
        print("=" * 50)

        if not self.start_local_server():
            self.results['website_tests'] = {'error': 'Could not start local server'}
            return

        base_url = f'http://127.0.0.1:{self.server_port}'
        pages_to_test = [
            '/',
            '/contest.html',
            '/test-modular.html',
            '/contest-modular.html',
            '/js/modules/index.js',
            '/js/modules/base-optimizer.js',
            '/js/modules/prima-algorithms.js',
            '/js/modules/scipy-algorithms.js',
            '/js/modules/evolutionary-algorithms.js',
            '/js/modules/search-algorithms.js',
            '/js/modules/optimizer-factory.js',
        ]

        page_results = {}
        for page in pages_to_test:
            try:
                url = urljoin(base_url, page)
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    content_length = len(response.content)
                    print(f"  ✓ {page} ({content_length/1024:.1f} KB)")
                    page_results[page] = {'status': 'ok', 'size': content_length}
                else:
                    print(f"  ✗ {page} - Status {response.status_code}")
                    page_results[page] = {'status': f'error_{response.status_code}', 'size': 0}

            except requests.exceptions.RequestException as e:
                print(f"  ✗ {page} - Request failed: {e}")
                page_results[page] = {'status': 'request_failed', 'size': 0}

        self.results['website_tests'] = page_results

        successful_pages = [p for p, r in page_results.items() if r['status'] == 'ok']
        print(f"\n✓ {len(successful_pages)}/{len(pages_to_test)} pages loaded successfully")

    def test_external_links(self):
        """Test external GitHub and academic links."""
        print("\n🔗 TESTING EXTERNAL LINKS")
        print("=" * 50)

        # Extract links from index.html
        index_file = self.docs_path / 'index.html'
        if not index_file.exists():
            self.results['link_tests'] = {'error': 'index.html not found'}
            return

        content = index_file.read_text()

        # Find GitHub links
        import re
        github_links = re.findall(r'href="(https://github\.com/[^"]*)"', content)
        doi_links = re.findall(r'href="(https://doi\.org/[^"]*)"', content)
        arxiv_links = re.findall(r'href="(https://arxiv\.org/[^"]*)"', content)

        all_links = list(set(github_links + doi_links + arxiv_links))

        print(f"Found {len(all_links)} external links to test:")
        print(f"  - GitHub: {len(github_links)}")
        print(f"  - DOI: {len(doi_links)}")
        print(f"  - arXiv: {len(arxiv_links)}")

        link_results = {}
        working_count = 0

        for i, link in enumerate(all_links[:20], 1):  # Test first 20 to avoid rate limits
            try:
                response = requests.head(link, timeout=10, allow_redirects=True)
                if response.status_code in [200, 301, 302]:
                    print(f"  ✓ {link.split('/')[-1]}")
                    link_results[link] = 'ok'
                    working_count += 1
                else:
                    print(f"  ✗ {link.split('/')[-1]} - Status {response.status_code}")
                    link_results[link] = f'error_{response.status_code}'

                # Be nice to servers
                time.sleep(0.5)

            except requests.exceptions.RequestException as e:
                print(f"  ✗ {link.split('/')[-1]} - {str(e)[:50]}...")
                link_results[link] = 'request_failed'

        self.results['link_tests'] = {
            'total_found': len(all_links),
            'tested': len(link_results),
            'working': working_count,
            'results': link_results
        }

        print(f"\n✓ {working_count}/{len(link_results)} tested links working")

    def cleanup(self):
        """Clean up test resources."""
        if self.server_process:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()

    def generate_report(self):
        """Generate comprehensive test report."""
        print("\n" + "=" * 60)
        print("🧪 COMPREHENSIVE TEST REPORT")
        print("=" * 60)

        # Python implementation summary
        if 'python_tests' in self.results and 'working' in self.results['python_tests']:
            py = self.results['python_tests']
            print(f"🐍 Python: {py['working']}/{py['total_algorithms']} algorithms working")

        # File integrity summary
        if 'file_integrity' in self.results:
            files = self.results['file_integrity']
            ok_files = sum(1 for f in files.values() if f['status'] == 'ok')
            total_files = len(files)
            print(f"📁 Files: {ok_files}/{total_files} files present and valid")

        # Website summary
        if 'website_tests' in self.results and isinstance(self.results['website_tests'], dict):
            pages = self.results['website_tests']
            ok_pages = sum(1 for p in pages.values() if isinstance(p, dict) and p.get('status') == 'ok')
            total_pages = len(pages)
            print(f"🌐 Website: {ok_pages}/{total_pages} pages loading correctly")

        # Links summary
        if 'link_tests' in self.results and 'working' in self.results['link_tests']:
            links = self.results['link_tests']
            print(f"🔗 Links: {links['working']}/{links['tested']} external links working")

        # Overall status
        print("\n" + "=" * 60)

        # Save detailed report
        report_file = self.base_path / 'test_report.json'
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"📊 Detailed report saved to: {report_file}")

        return self.results

    def run_all_tests(self):
        """Run all tests in sequence."""
        print("🚀 STARTING COMPREHENSIVE HUMPDAY TEST SUITE")
        print("=" * 60)

        try:
            self.test_file_integrity()
            self.test_python_implementation()
            self.test_website_pages()
            self.test_external_links()
            return self.generate_report()

        except KeyboardInterrupt:
            print("\n⚠ Tests interrupted by user")
        except Exception as e:
            print(f"\n✗ Test suite failed: {e}")
        finally:
            self.cleanup()

if __name__ == "__main__":
    tester = HumpDayTester()
    tester.run_all_tests()