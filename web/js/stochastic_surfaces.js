/**
 * JavaScript port of the sophisticated StochasticSurfaceGenerator
 * Proper stochastic surface generation for valid benchmarking
 * Each run creates truly random surfaces to avoid bias from fixed landscapes
 */

class StochasticSurfaceGenerator {
    constructor(seed = null) {
        // Simple seedable random number generator
        this.seed = seed || Date.now();
        this.rng_state = this.seed;

        // Generate random parameters for this run
        this._generateRandomParameters();
    }

    // Simple LCG random number generator for reproducible results
    _random() {
        this.rng_state = (this.rng_state * 1664525 + 1013904223) % (2**32);
        return this.rng_state / (2**32);
    }

    _randomNormal(mean = 0, std = 1) {
        // Box-Muller transform for normal distribution
        if (this._spare !== undefined) {
            const val = this._spare * std + mean;
            delete this._spare;
            return val;
        }

        const u = this._random();
        const v = this._random();
        const mag = std * Math.sqrt(-2 * Math.log(u));
        this._spare = mag * Math.cos(2 * Math.PI * v);
        return mag * Math.sin(2 * Math.PI * v) + mean;
    }

    _randomUniform(min, max) {
        return min + this._random() * (max - min);
    }

    _randomChoice(choices, probabilities = null) {
        if (!probabilities) {
            return choices[Math.floor(this._random() * choices.length)];
        }

        const r = this._random();
        let cumSum = 0;
        for (let i = 0; i < choices.length; i++) {
            cumSum += probabilities[i];
            if (r <= cumSum) return choices[i];
        }
        return choices[choices.length - 1];
    }

    _generateRandomParameters() {
        // Random shifts (different for each dimension)
        this.globalShift = this._randomUniform(-0.3, 0.3);
        this.dimensionShifts = {}; // Will be generated per function call

        // Random rotations
        this.useRotation = this._randomChoice([true, false], [0.7, 0.3]); // 70% chance

        // Random scaling factors
        this.scaleFactor = this._randomUniform(0.5, 2.0);

        // Random noise level
        this.noiseLevel = this._randomUniform(0.0, 0.05); // Up to 5% noise

        // Random conditioning (for appropriate functions)
        this.conditioningFactor = this._randomUniform(1.0, 100.0);

        // Random multimodal density
        this.modalFrequency = this._randomUniform(0.5, 2.0);
    }

    _hashString(str) {
        // Simple hash function for strings
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32bit integer
        }
        return Math.abs(hash);
    }

    _getDimensionShifts(nDim, functionName) {
        const key = `${functionName}_${nDim}`;

        if (!(key in this.dimensionShifts)) {
            // Create deterministic but random shifts based on function name
            const seedString = `${functionName}_${nDim}_${this.globalShift}`;
            const seedHash = this._hashString(seedString);

            // Save current state
            const tempState = this.rng_state;
            this.rng_state = seedHash;

            const shifts = [];
            for (let i = 0; i < nDim; i++) {
                shifts.push(this._randomUniform(-0.2, 0.2));
            }
            this.dimensionShifts[key] = shifts;

            // Restore state
            this.rng_state = tempState;
        }

        return this.dimensionShifts[key];
    }

    _randomOrthogonalMatrix(n) {
        // Generate random orthogonal matrix using QR decomposition
        const A = [];
        for (let i = 0; i < n; i++) {
            A[i] = [];
            for (let j = 0; j < n; j++) {
                A[i][j] = this._randomNormal();
            }
        }

        // Simple QR decomposition for small matrices
        if (n <= 3) {
            return this._gramSchmidt(A);
        } else {
            // For larger matrices, use a simpler rotation approach
            return this._generateSimpleRotation(n);
        }
    }

    _gramSchmidt(matrix) {
        const n = matrix.length;
        const Q = [];

        for (let i = 0; i < n; i++) {
            Q[i] = [...matrix[i]];

            // Subtract projections onto previous vectors
            for (let j = 0; j < i; j++) {
                const dot = this._dotProduct(Q[i], Q[j]);
                for (let k = 0; k < n; k++) {
                    Q[i][k] -= dot * Q[j][k];
                }
            }

            // Normalize
            const norm = this._vectorNorm(Q[i]);
            if (norm > 1e-10) {
                for (let k = 0; k < n; k++) {
                    Q[i][k] /= norm;
                }
            }
        }

        return Q;
    }

    _generateSimpleRotation(n) {
        // Generate a rotation matrix using random angles
        const matrix = [];
        for (let i = 0; i < n; i++) {
            matrix[i] = Array(n).fill(0);
            matrix[i][i] = 1;
        }

        // Apply random 2D rotations in different planes
        for (let iter = 0; iter < n; iter++) {
            const i = Math.floor(this._random() * n);
            const j = Math.floor(this._random() * n);
            if (i !== j) {
                const angle = this._randomUniform(0, 2 * Math.PI);
                this._applyPlaneRotation(matrix, i, j, angle);
            }
        }

        return matrix;
    }

    _applyPlaneRotation(matrix, i, j, angle) {
        const cos_a = Math.cos(angle);
        const sin_a = Math.sin(angle);
        const n = matrix.length;

        for (let k = 0; k < n; k++) {
            const temp_i = matrix[k][i];
            const temp_j = matrix[k][j];
            matrix[k][i] = cos_a * temp_i - sin_a * temp_j;
            matrix[k][j] = sin_a * temp_i + cos_a * temp_j;
        }
    }

    _dotProduct(a, b) {
        return a.reduce((sum, val, i) => sum + val * b[i], 0);
    }

    _vectorNorm(vec) {
        return Math.sqrt(vec.reduce((sum, val) => sum + val * val, 0));
    }

    _matrixVectorMultiply(matrix, vector) {
        return matrix.map(row => this._dotProduct(row, vector));
    }

    _applyRotation(x, functionId) {
        if (!this.useRotation || x.length === 1) {
            return x;
        }

        // Generate rotation matrix deterministically for this instance
        const tempState = this.rng_state;
        this.rng_state = this._hashString(functionId);

        const n_dim = x.length;
        let rotated;

        if (n_dim === 2) {
            // 2D rotation
            const theta = this._randomUniform(0, 2 * Math.PI);
            const cos_t = Math.cos(theta);
            const sin_t = Math.sin(theta);
            rotated = [
                cos_t * x[0] - sin_t * x[1],
                sin_t * x[0] + cos_t * x[1]
            ];
        } else {
            // Higher dimensions: random orthogonal matrix
            const rotationMatrix = this._randomOrthogonalMatrix(n_dim);
            rotated = this._matrixVectorMultiply(rotationMatrix, x);
        }

        this.rng_state = tempState;
        return rotated;
    }

    _addNoise(value) {
        if (this.noiseLevel > 0) {
            const noise = this._randomNormal(0, this.noiseLevel * Math.abs(value));
            return value + noise;
        }
        return value;
    }

    stochasticSphere(functionId = null) {
        if (functionId === null) {
            functionId = `sphere_${Math.floor(this._random() * 1000000)}`;
        }

        return (x) => {
            const n_dim = x.length;

            // Get deterministic but random shifts for this function
            const dimShifts = this._getDimensionShifts(n_dim, functionId);

            // Transform to optimization domain with random scaling
            const scaled_x = x.map((xi, i) =>
                this.scaleFactor * (10 * xi - 5) + dimShifts[i] + this.globalShift
            );

            // Apply random rotation
            const rotated_x = this._applyRotation(scaled_x, functionId);

            // Compute sphere function
            const result = rotated_x.reduce((sum, xi) => sum + xi * xi, 0);

            // Add noise
            return this._addNoise(result);
        };
    }

    stochasticRastrigin(functionId = null) {
        if (functionId === null) {
            functionId = `rastrigin_${Math.floor(this._random() * 1000000)}`;
        }

        return (x) => {
            const n_dim = x.length;

            // Get random parameters for this function instance
            const dimShifts = this._getDimensionShifts(n_dim, functionId);

            // Transform with random parameters
            const scaled_x = x.map((xi, i) =>
                this.scaleFactor * (10.24 * xi - 5.12) + dimShifts[i] + this.globalShift
            );

            // Apply rotation
            const rotated_x = this._applyRotation(scaled_x, functionId);

            // Rastrigin with random frequency modulation
            const freq = this.modalFrequency;
            let result = 10 * n_dim;
            for (let i = 0; i < n_dim; i++) {
                result += rotated_x[i] * rotated_x[i] - 10 * Math.cos(2 * Math.PI * freq * rotated_x[i]);
            }

            return this._addNoise(result);
        };
    }

    stochasticRosenbrock(functionId = null) {
        if (functionId === null) {
            functionId = `rosenbrock_${Math.floor(this._random() * 1000000)}`;
        }

        return (x) => {
            const n_dim = x.length;

            if (n_dim < 2) {
                // Fallback for 1D
                return (x[0] - 1) * (x[0] - 1);
            }

            const dimShifts = this._getDimensionShifts(n_dim, functionId);

            // Transform with random conditioning
            const scaled_x = x.map((xi, i) =>
                this.scaleFactor * (4.096 * xi - 2.048) + dimShifts[i] + this.globalShift
            );

            // Apply rotation
            const rotated_x = this._applyRotation(scaled_x, functionId);

            // Rosenbrock with random conditioning factor
            const a = 1.0;
            const b = 100.0 * this.conditioningFactor;

            let result = 0;
            for (let i = 0; i < n_dim - 1; i++) {
                const term1 = b * Math.pow(rotated_x[i + 1] - rotated_x[i] * rotated_x[i], 2);
                const term2 = Math.pow(a - rotated_x[i], 2);
                result += term1 + term2;
            }

            return this._addNoise(result);
        };
    }

    stochasticAckley(functionId = null) {
        if (functionId === null) {
            functionId = `ackley_${Math.floor(this._random() * 1000000)}`;
        }

        return (x) => {
            const n_dim = x.length;

            const dimShifts = this._getDimensionShifts(n_dim, functionId);

            // Transform with random scaling
            const scaled_x = x.map((xi, i) =>
                this.scaleFactor * (65.536 * xi - 32.768) + dimShifts[i] + this.globalShift
            );

            // Apply rotation
            const rotated_x = this._applyRotation(scaled_x, functionId);

            // Ackley with random parameters
            const a = 20 * this._randomUniform(0.8, 1.2);
            const b = 0.2 * this._randomUniform(0.8, 1.2);
            const c = 2 * Math.PI * this.modalFrequency;

            const sum_sq = rotated_x.reduce((sum, xi) => sum + xi * xi, 0);
            const sum_cos = rotated_x.reduce((sum, xi) => sum + Math.cos(c * xi), 0);

            const term1 = -a * Math.exp(-b * Math.sqrt(sum_sq / n_dim));
            const term2 = -Math.exp(sum_cos / n_dim);
            const result = term1 + term2 + a + Math.E;

            return this._addNoise(result);
        };
    }

    stochasticGriewank(functionId = null) {
        if (functionId === null) {
            functionId = `griewank_${Math.floor(this._random() * 1000000)}`;
        }

        return (x) => {
            const n_dim = x.length;

            const dimShifts = this._getDimensionShifts(n_dim, functionId);

            // Transform with random scaling
            const scaled_x = x.map((xi, i) =>
                this.scaleFactor * (1200 * xi - 600) + dimShifts[i] + this.globalShift
            );

            // Apply rotation
            const rotated_x = this._applyRotation(scaled_x, functionId);

            // Griewank function
            const sum_sq = rotated_x.reduce((sum, xi) => sum + xi * xi, 0) / 4000;
            const prod_cos = rotated_x.reduce((prod, xi, i) =>
                prod * Math.cos(xi / Math.sqrt(i + 1)), 1);
            const result = sum_sq - prod_cos + 1;

            return this._addNoise(result);
        };
    }

    getRandomFunctionSuite(nFunctions = 10) {
        const baseFunctions = [
            ['sphere', this.stochasticSphere.bind(this)],
            ['rastrigin', this.stochasticRastrigin.bind(this)],
            ['rosenbrock', this.stochasticRosenbrock.bind(this)],
            ['ackley', this.stochasticAckley.bind(this)],
            ['griewank', this.stochasticGriewank.bind(this)]
        ];

        const suite = [];

        for (let i = 0; i < nFunctions; i++) {
            // Randomly select base function type
            const [baseName, baseFunc] = this._randomChoice(baseFunctions);

            // Create unique instance
            const instanceId = `${baseName}_instance_${i}_${Math.floor(this._random() * 1000000)}`;

            suite.push({
                name: instanceId,
                func: baseFunc(instanceId),
                type: baseName
            });
        }

        return suite;
    }

    getBenchmarkMetadata() {
        return {
            globalShift: this.globalShift,
            useRotation: this.useRotation,
            scaleFactor: this.scaleFactor,
            noiseLevel: this.noiseLevel,
            conditioningFactor: this.conditioningFactor,
            modalFrequency: this.modalFrequency,
            seed: this.seed
        };
    }
}

// Factory function for easy use
function createFairBenchmarkRun(nFunctions = 20, seed = null) {
    console.log(`🎲 Generating ${nFunctions} random function instances for fair benchmarking...`);

    // Create stochastic generator
    const generator = new StochasticSurfaceGenerator(seed);

    // Generate random function suite
    const functionSuite = generator.getRandomFunctionSuite(nFunctions);

    // Get metadata
    const metadata = generator.getBenchmarkMetadata();

    console.log(`✅ Random surfaces generated:`);
    console.log(`   Global shift: ${metadata.globalShift.toFixed(3)}`);
    console.log(`   Rotation enabled: ${metadata.useRotation}`);
    console.log(`   Scale factor: ${metadata.scaleFactor.toFixed(3)}`);
    console.log(`   Noise level: ${metadata.noiseLevel.toFixed(3)}`);
    console.log(`   Modal frequency: ${metadata.modalFrequency.toFixed(3)}`);

    return { functionSuite, metadata };
}

// Export for use in other modules
window.StochasticSurfaceGenerator = StochasticSurfaceGenerator;
window.createFairBenchmarkRun = createFairBenchmarkRun;