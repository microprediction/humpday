#!/usr/bin/env python3
"""
Comprehensive HumpDay Site Audit & Testing Suite
Tests functionality, performance, links, and prepares style consistency report.
"""

import os
import sys
import subprocess
import json
import time
import re
from pathlib import Path
import requests
from urllib.parse import urljoin, urlparse
import concurrent.futures
from datetime import datetime

class ComprehensiveSiteAuditor:
    def __init__(self):
        self.base_path = Path('/Users/petercotton/github/humpday')
        self.docs_path = self.base_path / 'docs'
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'algorithm_tests': {},
            'performance_tests': {},
            'link_tests': {},
            'style_audit': {},
            'accessibility_tests': {},
            'browser_tests': {},
            'security_tests': {}
        }
        self.server_process = None
        self.server_port = 8084

    def test_all_python_algorithms(self):
        """Comprehensive Python algorithm testing with performance metrics."""
        print("PYTHON ALGORITHM COMPREHENSIVE TESTING")
        print("=" * 60)

        try:
            sys.path.insert(0, str(self.base_path))
            from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS, get_optimizer
            import numpy as np

            test_functions = {
                'sphere': lambda x: sum(xi**2 for xi in x),
                'rosenbrock': lambda x: sum(100.0*(x[i+1]-x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1)),
                'rastrigin': lambda x: 10*len(x) + sum(xi**2 - 10*np.cos(2*np.pi*xi) for xi in x)
            }

            algorithm_results = {}

            for alg_name in sorted(PURE_OPTIMIZERS.keys()):
                print(f"\nTesting {alg_name}...")
                algorithm_results[alg_name] = {}

                try:
                    optimizer_func = get_optimizer(alg_name)
                    if not optimizer_func:
                        algorithm_results[alg_name]['status'] = 'failed_to_get_optimizer'
                        continue

                    for func_name, test_func in test_functions.items():
                        try:
                            # Test with different dimensions
                            for n_dim in [2, 5, 10]:
                                start_time = time.time()
                                best_val, best_x = optimizer_func(test_func, n_dim=n_dim, n_trials=50)
                                end_time = time.time()

                                test_key = f"{func_name}_{n_dim}d"
                                algorithm_results[alg_name][test_key] = {
                                    'best_value': float(best_val),
                                    'solution_length': len(best_x),
                                    'execution_time': end_time - start_time,
                                    'success': True
                                }

                                print(f"  {test_key}: {best_val:.6f} ({(end_time-start_time)*1000:.1f}ms)")

                        except Exception as e:
                            algorithm_results[alg_name][f"{func_name}_{n_dim}d"] = {
                                'error': str(e),
                                'success': False
                            }

                except Exception as e:
                    algorithm_results[alg_name]['error'] = str(e)
                    print(f"  ERROR: {e}")

            self.results['algorithm_tests']['python'] = algorithm_results

            # Summary
            working_algorithms = sum(1 for alg in algorithm_results.values()
                                   if any(test.get('success', False) for test in alg.values() if isinstance(test, dict)))

            print(f"\nPython Algorithm Summary: {working_algorithms}/{len(PURE_OPTIMIZERS)} algorithms working")

        except Exception as e:
            self.results['algorithm_tests']['python_error'] = str(e)
            print(f"Python testing failed: {e}")

    def start_test_server(self):
        """Start local server for comprehensive web testing."""
        print(f"\nSTARTING TEST SERVER ON PORT {self.server_port}")
        print("=" * 60)

        try:
            # Kill existing server
            subprocess.run(['pkill', '-f', f'http.server {self.server_port}'],
                         capture_output=True, cwd=self.docs_path)
            time.sleep(1)

            self.server_process = subprocess.Popen(
                ['python3', '-m', 'http.server', str(self.server_port), '--bind', '127.0.0.1'],
                cwd=self.docs_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            time.sleep(3)

            # Verify server
            response = requests.get(f'http://127.0.0.1:{self.server_port}/', timeout=10)
            if response.status_code == 200:
                print(f"Server started successfully: http://127.0.0.1:{self.server_port}")
                return True
            else:
                print(f"Server error: {response.status_code}")
                return False

        except Exception as e:
            print(f"Server startup failed: {e}")
            return False

    def test_page_performance(self):
        """Test page load performance and resource loading."""
        print("\nPAGE PERFORMANCE TESTING")
        print("=" * 60)

        base_url = f'http://127.0.0.1:{self.server_port}'
        pages = ['/', '/contest.html', '/test-modular.html', '/contest-modular.html']

        performance_results = {}

        for page in pages:
            print(f"Testing {page}...")
            try:
                start_time = time.time()
                response = requests.get(urljoin(base_url, page), timeout=15)
                load_time = time.time() - start_time

                performance_results[page] = {
                    'status_code': response.status_code,
                    'load_time_ms': load_time * 1000,
                    'content_size_kb': len(response.content) / 1024,
                    'content_type': response.headers.get('content-type', ''),
                    'success': response.status_code == 200
                }

                print(f"  Status: {response.status_code}, Size: {len(response.content)/1024:.1f}KB, Time: {load_time*1000:.0f}ms")

            except Exception as e:
                performance_results[page] = {
                    'error': str(e),
                    'success': False
                }
                print(f"  ERROR: {e}")

        self.results['performance_tests'] = performance_results

    def test_javascript_modules(self):
        """Test JavaScript module loading and functionality."""
        print("\nJAVASCRIPT MODULE TESTING")
        print("=" * 60)

        base_url = f'http://127.0.0.1:{self.server_port}'
        js_modules = [
            '/js/modules/base-optimizer.js',
            '/js/modules/prima-algorithms.js',
            '/js/modules/scipy-algorithms.js',
            '/js/modules/evolutionary-algorithms.js',
            '/js/modules/search-algorithms.js',
            '/js/modules/optimizer-factory.js',
            '/js/modules/index.js'
        ]

        js_results = {}

        for module in js_modules:
            print(f"Testing {module}...")
            try:
                start_time = time.time()
                response = requests.get(urljoin(base_url, module), timeout=10)
                load_time = time.time() - start_time

                # Basic syntax check (look for obvious errors)
                content = response.text
                syntax_issues = []

                if 'undefined' in content.lower() and 'typeof' not in content:
                    syntax_issues.append('potential undefined variables')

                if response.status_code == 200:
                    # Count classes and functions
                    class_count = len(re.findall(r'class\s+\w+\s+extends', content))
                    function_count = len(re.findall(r'function\s+\w+\s*\(', content))

                    js_results[module] = {
                        'status_code': response.status_code,
                        'load_time_ms': load_time * 1000,
                        'size_kb': len(content) / 1024,
                        'class_count': class_count,
                        'function_count': function_count,
                        'syntax_issues': syntax_issues,
                        'success': True
                    }

                    print(f"  OK: {class_count} classes, {function_count} functions, {len(content)/1024:.1f}KB")
                else:
                    js_results[module] = {
                        'status_code': response.status_code,
                        'success': False
                    }
                    print(f"  ERROR: {response.status_code}")

            except Exception as e:
                js_results[module] = {
                    'error': str(e),
                    'success': False
                }
                print(f"  ERROR: {e}")

        self.results['algorithm_tests']['javascript'] = js_results

    def comprehensive_link_testing(self):
        """Test all links across all pages."""
        print("\nCOMPREHENSIVE LINK TESTING")
        print("=" * 60)

        # Get all HTML files
        html_files = list(self.docs_path.glob('*.html'))
        base_url = f'http://127.0.0.1:{self.server_port}'

        all_links = set()
        internal_links = set()
        external_links = set()

        # Extract links from all HTML files
        for html_file in html_files:
            try:
                content = html_file.read_text()
                # Find all href links
                links = re.findall(r'href="([^"]*)"', content)

                for link in links:
                    if link.startswith('http'):
                        external_links.add(link)
                    elif link.startswith('/') or link.endswith('.html'):
                        internal_links.add(link)
                    all_links.add(link)

            except Exception as e:
                print(f"Error reading {html_file}: {e}")

        print(f"Found {len(all_links)} total links:")
        print(f"  Internal: {len(internal_links)}")
        print(f"  External: {len(external_links)}")

        # Test internal links
        internal_results = {}
        print("\nTesting internal links...")
        for link in list(internal_links)[:20]:  # Test first 20
            try:
                test_url = urljoin(base_url, link)
                response = requests.head(test_url, timeout=5)
                internal_results[link] = {
                    'status': response.status_code,
                    'success': response.status_code == 200
                }
                print(f"  {link}: {response.status_code}")
            except Exception as e:
                internal_results[link] = {
                    'error': str(e),
                    'success': False
                }

        # Test external links (sample)
        external_results = {}
        print("\nTesting external links (sample)...")
        external_sample = list(external_links)[:15]  # Test 15 external links

        for link in external_sample:
            try:
                response = requests.head(link, timeout=10, allow_redirects=True)
                external_results[link] = {
                    'status': response.status_code,
                    'success': response.status_code in [200, 301, 302]
                }
                status_icon = "✓" if response.status_code in [200, 301, 302] else "✗"
                print(f"  {status_icon} {link}: {response.status_code}")
                time.sleep(0.5)  # Be nice to servers
            except Exception as e:
                external_results[link] = {
                    'error': str(e),
                    'success': False
                }
                print(f"  ✗ {link}: {str(e)[:50]}...")

        self.results['link_tests'] = {
            'internal': internal_results,
            'external': external_results,
            'totals': {
                'internal_found': len(internal_links),
                'external_found': len(external_links),
                'internal_tested': len(internal_results),
                'external_tested': len(external_results)
            }
        }

    def audit_style_consistency(self):
        """Audit CSS and style consistency across pages."""
        print("\nSTYLE CONSISTENCY AUDIT")
        print("=" * 60)

        html_files = list(self.docs_path.glob('*.html'))
        style_audit = {
            'fonts_found': set(),
            'color_schemes': set(),
            'style_inconsistencies': [],
            'emoji_usage': [],
            'font_issues': []
        }

        for html_file in html_files:
            try:
                content = html_file.read_text()

                # Check for fonts
                font_matches = re.findall(r'font-family:\s*["\']?([^;"\']+)["\']?', content, re.IGNORECASE)
                for font in font_matches:
                    style_audit['fonts_found'].add(font.strip())

                # Check for colors
                color_matches = re.findall(r'color:\s*([^;]+)', content, re.IGNORECASE)
                for color in color_matches:
                    style_audit['color_schemes'].add(color.strip())

                # Check for emojis (user wants NO emojis!)
                emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000026FF\U00002700-\U000027BF]')
                emoji_matches = emoji_pattern.findall(content)
                if emoji_matches:
                    style_audit['emoji_usage'].append({
                        'file': str(html_file.name),
                        'emojis': emoji_matches,
                        'count': len(emoji_matches)
                    })

                # Check font consistency
                font_declarations = re.findall(r'font-family:\s*([^;]+)', content, re.IGNORECASE)
                expected_fonts = ["'Times New Roman'", "Times", "serif"]

                for font_decl in font_declarations:
                    if not any(expected in font_decl for expected in expected_fonts):
                        style_audit['font_issues'].append({
                            'file': html_file.name,
                            'font': font_decl.strip(),
                            'issue': 'non-standard font family'
                        })

            except Exception as e:
                style_audit['style_inconsistencies'].append({
                    'file': html_file.name,
                    'error': str(e)
                })

        # Convert sets to lists for JSON serialization
        style_audit['fonts_found'] = list(style_audit['fonts_found'])
        style_audit['color_schemes'] = list(style_audit['color_schemes'])

        self.results['style_audit'] = style_audit

        # Print style audit results
        print(f"Fonts found: {len(style_audit['fonts_found'])}")
        for font in style_audit['fonts_found']:
            print(f"  - {font}")

        if style_audit['emoji_usage']:
            print(f"\nEMOJIS FOUND (NEED TO REMOVE): {len(style_audit['emoji_usage'])} files")
            for emoji_file in style_audit['emoji_usage']:
                print(f"  {emoji_file['file']}: {emoji_file['emojis']} ({emoji_file['count']} emojis)")

        if style_audit['font_issues']:
            print(f"\nFONT INCONSISTENCIES: {len(style_audit['font_issues'])}")
            for issue in style_audit['font_issues']:
                print(f"  {issue['file']}: {issue['font']}")

    def test_accessibility(self):
        """Basic accessibility testing."""
        print("\nACCESSIBILITY TESTING")
        print("=" * 60)

        html_files = list(self.docs_path.glob('*.html'))
        accessibility_results = {}

        for html_file in html_files:
            try:
                content = html_file.read_text()

                # Check for accessibility issues
                issues = []

                # Check for alt text on images
                img_tags = re.findall(r'<img[^>]*>', content, re.IGNORECASE)
                for img in img_tags:
                    if 'alt=' not in img:
                        issues.append('Image missing alt text')

                # Check for proper heading structure
                headings = re.findall(r'<h([1-6])[^>]*>', content, re.IGNORECASE)
                if headings and headings[0] != '1':
                    issues.append('Page does not start with h1')

                # Check for link descriptions
                link_texts = re.findall(r'<a[^>]*>([^<]*)</a>', content, re.IGNORECASE)
                generic_links = [text.strip() for text in link_texts if text.strip().lower() in ['click here', 'here', 'link', 'more']]
                if generic_links:
                    issues.append(f'Generic link text found: {generic_links}')

                accessibility_results[html_file.name] = {
                    'issues': issues,
                    'issue_count': len(issues)
                }

                print(f"{html_file.name}: {len(issues)} issues")
                for issue in issues:
                    print(f"  - {issue}")

            except Exception as e:
                accessibility_results[html_file.name] = {
                    'error': str(e)
                }

        self.results['accessibility_tests'] = accessibility_results

    def generate_comprehensive_report(self):
        """Generate final comprehensive audit report."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE SITE AUDIT REPORT")
        print("=" * 80)

        # Python Algorithm Summary
        if 'python' in self.results['algorithm_tests']:
            py_results = self.results['algorithm_tests']['python']
            working_algs = [name for name, data in py_results.items()
                          if isinstance(data, dict) and any(test.get('success', False) for test in data.values() if isinstance(test, dict))]
            print(f"Python Algorithms: {len(working_algs)}/{len(py_results)} working")

        # JavaScript Module Summary
        if 'javascript' in self.results['algorithm_tests']:
            js_results = self.results['algorithm_tests']['javascript']
            working_modules = sum(1 for module in js_results.values() if module.get('success', False))
            print(f"JavaScript Modules: {working_modules}/{len(js_results)} loading correctly")

        # Performance Summary
        if 'performance_tests' in self.results:
            perf_results = self.results['performance_tests']
            working_pages = sum(1 for page in perf_results.values() if page.get('success', False))
            avg_load_time = sum(page.get('load_time_ms', 0) for page in perf_results.values() if 'load_time_ms' in page) / len(perf_results)
            print(f"Page Performance: {working_pages}/{len(perf_results)} pages, avg load: {avg_load_time:.0f}ms")

        # Link Testing Summary
        if 'link_tests' in self.results:
            link_results = self.results['link_tests']
            internal_ok = sum(1 for link in link_results['internal'].values() if link.get('success', False))
            external_ok = sum(1 for link in link_results['external'].values() if link.get('success', False))
            print(f"Link Testing: {internal_ok}/{len(link_results['internal'])} internal, {external_ok}/{len(link_results['external'])} external working")

        # Style Audit Summary
        if 'style_audit' in self.results:
            style_results = self.results['style_audit']
            emoji_files = len(style_results.get('emoji_usage', []))
            font_issues = len(style_results.get('font_issues', []))
            print(f"Style Audit: {emoji_files} files with emojis (NEED FIXING), {font_issues} font inconsistencies")

        # Accessibility Summary
        if 'accessibility_tests' in self.results:
            acc_results = self.results['accessibility_tests']
            total_issues = sum(result.get('issue_count', 0) for result in acc_results.values() if isinstance(result, dict))
            print(f"Accessibility: {total_issues} issues found across {len(acc_results)} pages")

        # Save detailed report
        report_file = self.base_path / 'comprehensive_audit_report.json'
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\nDetailed report saved: {report_file}")
        return self.results

    def cleanup(self):
        """Clean up resources."""
        if self.server_process:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()

    def run_full_audit(self):
        """Run complete audit suite."""
        print("STARTING COMPREHENSIVE HUMPDAY SITE AUDIT")
        print("=" * 80)

        try:
            # Start server for web tests
            if not self.start_test_server():
                print("Cannot start server - skipping web tests")

            # Run all test suites
            self.test_all_python_algorithms()
            self.test_page_performance()
            self.test_javascript_modules()
            self.comprehensive_link_testing()
            self.audit_style_consistency()
            self.test_accessibility()

            return self.generate_comprehensive_report()

        except KeyboardInterrupt:
            print("\nAudit interrupted by user")
        except Exception as e:
            print(f"\nAudit failed: {e}")
        finally:
            self.cleanup()

if __name__ == "__main__":
    auditor = ComprehensiveSiteAuditor()
    auditor.run_full_audit()