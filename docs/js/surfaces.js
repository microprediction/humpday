/**
 * Test surface functions for optimization contests
 * JavaScript implementations of various optimization landscapes
 */

const TestSurfaces = {
    // Sphere function - simple smooth landscape
    sphere(params = {}) {
        const center = params.center || 0.5;
        const scale = params.scale || 1.0;

        return function(x) {
            let sum = 0;
            for (let i = 0; i < x.length; i++) {
                sum += Math.pow(x[i] - center, 2);
            }
            return scale * sum;
        };
    },

    // Rosenbrock function - valley landscape
    rosenbrock(params = {}) {
        const a = params.a || 1.0;
        const b = params.b || 100.0;

        return function(x) {
            if (x.length < 2) return Math.pow(x[0] - a, 2);

            let sum = 0;
            for (let i = 0; i < x.length - 1; i++) {
                sum += b * Math.pow(x[i + 1] - x[i] * x[i], 2) + Math.pow(a - x[i], 2);
            }
            return sum;
        };
    },

    // Rastrigin function - multimodal landscape
    rastrigin(params = {}) {
        const A = params.A || 10.0;
        const omega = params.omega || 2 * Math.PI;

        return function(x) {
            const n = x.length;
            let sum = A * n;
            for (let i = 0; i < n; i++) {
                // Transform to [-5.12, 5.12] range for standard Rastrigin
                const xi = (x[i] - 0.5) * 10.24;
                sum += xi * xi - A * Math.cos(omega * xi);
            }
            return sum;
        };
    },

    // Ackley function - multimodal landscape with global structure
    ackley(params = {}) {
        const a = params.a || 20;
        const b = params.b || 0.2;
        const c = params.c || 2 * Math.PI;

        return function(x) {
            const n = x.length;
            if (n === 0) return 0;

            // Transform to [-32.768, 32.768] range for standard Ackley
            const transformedX = x.map(xi => (xi - 0.5) * 65.536);

            let sum1 = 0, sum2 = 0;
            for (let i = 0; i < n; i++) {
                sum1 += transformedX[i] * transformedX[i];
                sum2 += Math.cos(c * transformedX[i]);
            }

            return -a * Math.exp(-b * Math.sqrt(sum1 / n)) - Math.exp(sum2 / n) + a + Math.E;
        };
    },

    // Griewank function - multimodal with correlation
    griewank(params = {}) {
        return function(x) {
            const n = x.length;
            if (n === 0) return 0;

            // Transform to [-600, 600] range for standard Griewank
            const transformedX = x.map(xi => (xi - 0.5) * 1200);

            let sum = 0;
            let prod = 1;

            for (let i = 0; i < n; i++) {
                sum += transformedX[i] * transformedX[i];
                prod *= Math.cos(transformedX[i] / Math.sqrt(i + 1));
            }

            return 1 + sum / 4000 - prod;
        };
    },

    // Powell's function - separable landscape
    powell(params = {}) {
        return function(x) {
            const n = x.length;
            if (n < 4) {
                // For low dimensions, use a simplified version
                let sum = 0;
                for (let i = 0; i < n; i++) {
                    sum += Math.pow(x[i] - 0.5, 4);
                }
                return sum;
            }

            let sum = 0;
            for (let i = 0; i < n - 3; i += 4) {
                const x1 = x[i] - 0.5;
                const x2 = x[i + 1] - 0.5;
                const x3 = i + 2 < n ? x[i + 2] - 0.5 : 0;
                const x4 = i + 3 < n ? x[i + 3] - 0.5 : 0;

                sum += Math.pow(x1 + 10 * x2, 2) +
                       5 * Math.pow(x3 - x4, 2) +
                       Math.pow(x2 - 2 * x3, 4) +
                       10 * Math.pow(x1 - x4, 4);
            }
            return sum;
        };
    },

    // Styblinski-Tang function - multimodal
    styblinski(params = {}) {
        return function(x) {
            let sum = 0;
            for (let i = 0; i < x.length; i++) {
                // Transform to [-5, 5] range for standard Styblinski-Tang
                const xi = (x[i] - 0.5) * 10;
                sum += Math.pow(xi, 4) - 16 * xi * xi + 5 * xi;
            }
            return sum / 2; // Normalize
        };
    },

    // Rotated benchmarks — Rosenbrock, Rastrigin, and Ackley
    // pre-multiplied by a deterministic uniform-random orthogonal Q.
    // Each strips the axis-alignment bonus that coordinate-wise methods
    // collect on the standard benchmarks above. Q is computed once per
    // n_dim by a seeded Linear Congruential Generator + Mezzadri-2007
    // sign-corrected QR (see _rotationFor below), so the landscape is
    // reproducible and parity-test-safe but no longer coordinate-
    // aligned. Matches `humpday.objectives.classic.rotated_*_on_cube`.
    rotatedRosenbrock(params = {}) {
        return function(x) {
            const n = x.length;
            const Q = TestSurfaces._rotationFor(n);
            const s = x.map(xi => 4.0 * (xi - 0.5)); // [-2, 2]^n
            const y = TestSurfaces._matvec(Q, s);
            let sum = 0;
            for (let i = 0; i < n - 1; i++) {
                sum += 100.0 * Math.pow(y[i + 1] - y[i] * y[i], 2)
                     + Math.pow(1 - y[i], 2);
            }
            return sum;
        };
    },

    rotatedRastrigin(params = {}) {
        return function(x) {
            const n = x.length;
            const Q = TestSurfaces._rotationFor(n);
            const s = x.map(xi => 10.24 * (xi - 0.5)); // [-5.12, 5.12]^n
            const y = TestSurfaces._matvec(Q, s);
            let sum = 10.0 * n;
            for (let i = 0; i < n; i++) {
                sum += y[i] * y[i] - 10.0 * Math.cos(2.0 * Math.PI * y[i]);
            }
            return sum;
        };
    },

    rotatedAckley(params = {}) {
        return function(x) {
            const n = x.length;
            if (n === 0) return 0;
            const Q = TestSurfaces._rotationFor(n);
            const s = x.map(xi => 65.536 * (xi - 0.5)); // [-32.768, 32.768]^n
            const y = TestSurfaces._matvec(Q, s);
            const a = 20.0, b = 0.2, c = 2.0 * Math.PI;
            let sum1 = 0, sum2 = 0;
            for (let i = 0; i < n; i++) {
                sum1 += y[i] * y[i];
                sum2 += Math.cos(c * y[i]);
            }
            return -a * Math.exp(-b * Math.sqrt(sum1 / n))
                 - Math.exp(sum2 / n)
                 + a + Math.E;
        };
    },

    // Cache of deterministic rotation matrices, keyed by n_dim. Uses a
    // simple LCG (Park-Miller, period 2^31 - 2) seeded by 12345 + n_dim
    // so JS and Python don't need to share an RNG — each side just
    // needs a deterministic-per-n_dim orthogonal matrix. The matrix Q
    // is uniformly distributed on O(n) via Mezzadri's (2007)
    // sign-corrected QR construction: take Q from QR of a Gaussian
    // matrix, then sign-correct by diag(sign(diag(R))).
    _rotationCache: {},

    _rotationFor(n) {
        if (TestSurfaces._rotationCache[n]) return TestSurfaces._rotationCache[n];

        // Seeded RNG.
        let s = (12345 + n) | 0;
        if (s === 0) s = 1;
        const rand = () => {
            s = (s * 48271) % 2147483647;
            return s / 2147483647;
        };
        // Box-Muller for standard normal.
        const gauss = () => {
            const u1 = Math.max(rand(), 1e-12);
            const u2 = rand();
            return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        };

        // Random Gaussian matrix A (n × n).
        const A = new Array(n);
        for (let i = 0; i < n; i++) {
            A[i] = new Array(n);
            for (let j = 0; j < n; j++) A[i][j] = gauss();
        }

        // QR via Linalg.householderQR (loaded as a script before this
        // file in the browser; in Node we require it manually below).
        const _Linalg = (typeof Linalg !== 'undefined')
            ? Linalg
            : require('./modules/linalg.js');
        const { Q, R } = _Linalg.householderQR(A);

        // Sign-correct so Q is uniform on O(n) per Mezzadri (2007).
        for (let j = 0; j < n; j++) {
            const sign = R[j][j] >= 0 ? 1 : -1;
            for (let i = 0; i < n; i++) Q[i][j] *= sign;
        }

        TestSurfaces._rotationCache[n] = Q;
        return Q;
    },

    _matvec(M, v) {
        const out = new Array(M.length).fill(0);
        for (let i = 0; i < M.length; i++) {
            for (let j = 0; j < v.length; j++) out[i] += M[i][j] * v[j];
        }
        return out;
    },
};

// Surface generator for contests
const SurfaceGenerator = {
    generateChallengeSuite(spec) {
        const surfaces = [];
        const nSurfaces = spec.surface_count || 10;

        switch (spec.surfaceType) {
            case 'smooth':
                for (let i = 0; i < nSurfaces; i++) {
                    surfaces.push({
                        name: `sphere_variant_${i}`,
                        func: TestSurfaces.sphere({
                            center: 0.3 + i * 0.2,
                            scale: 0.5 + i * 0.3
                        })
                    });
                }
                break;

            case 'valley':
                for (let i = 0; i < nSurfaces; i++) {
                    surfaces.push({
                        name: `valley_${i}`,
                        func: TestSurfaces.rosenbrock({
                            a: 0.8 + i * 0.1,
                            b: 80 + i * 20
                        })
                    });
                }
                break;

            case 'multimodal':
                surfaces.push({
                    name: 'rastrigin_0',
                    func: TestSurfaces.rastrigin({A: 8})
                });
                surfaces.push({
                    name: 'ackley_1',
                    func: TestSurfaces.ackley()
                });
                surfaces.push({
                    name: 'griewank_2',
                    func: TestSurfaces.griewank()
                });
                break;

            case 'noisy':
                // Add noise to base functions
                for (let i = 0; i < nSurfaces; i++) {
                    const baseFuncs = [TestSurfaces.sphere(), TestSurfaces.rosenbrock()];
                    const baseFunc = baseFuncs[i % baseFuncs.length];

                    surfaces.push({
                        name: `noisy_${i}`,
                        func: function(x) {
                            const noise = (Math.random() - 0.5) * 0.1;
                            return baseFunc(x) + noise;
                        }
                    });
                }
                break;

            default: // mixed
                surfaces.push({
                    name: 'mixed_sphere',
                    func: TestSurfaces.sphere({center: 0.7})
                });
                surfaces.push({
                    name: 'mixed_rosenbrock',
                    func: TestSurfaces.rosenbrock()
                });
                surfaces.push({
                    name: 'mixed_rastrigin',
                    func: TestSurfaces.rastrigin({A: 5})
                });
                break;
        }

        return surfaces;
    }
};