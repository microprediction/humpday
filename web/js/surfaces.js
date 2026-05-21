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
    }
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