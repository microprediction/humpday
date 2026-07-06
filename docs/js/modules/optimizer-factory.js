/**
 * OptimizerFactory - Centralized creation of optimization algorithms
 *
 * Provides a factory interface for creating optimizer instances by name.
 * This maintains compatibility with existing contest and testing systems
 * while using the modular algorithm implementations.
 */

// OptimizerFactory for creating optimizers by name
const OptimizerFactory = {
    /**
     * Create an optimizer instance by algorithm name
     * @param {string} algorithmName - Name of the algorithm to create
     * @param {Function} objective - Objective function to optimize
     * @param {number} nTrials - Number of evaluations allowed
     * @param {number} nDim - Dimensionality of the problem
     * @returns {Optimizer} Optimizer instance
     */
    create(algorithmName, objective, nTrials, nDim) {
        // Map internal names to algorithm classes
        const algorithmMap = {
            // PRIMA algorithms (exact match with contest controller)
            'PRIMA_UOBYQA': window.PRIMA_UOBYQA,
            'PRIMA_NEWUOA': window.PRIMA_NEWUOA,
            'PRIMA_BOBYQA': window.PRIMA_BOBYQA,

            // SciPy algorithms
            'NelderMead': window.NelderMead,
            'Powell': window.Powell,
            'LBFGSB': window.LBFGSB,

            // Evolutionary algorithms
            'DifferentialEvolution': window.DifferentialEvolution,
            'ParticleSwarm': window.ParticleSwarm,
            'SimulatedAnnealing': window.SimulatedAnnealing,
            'GeneticAlgorithm': window.GeneticAlgorithm,
            'RandomSearch': window.RandomSearch,
            'BayesianOpt': window.BayesianOpt,
            'CMAEvolutionStrategy': window.CMAEvolutionStrategy,
            'FireflyAlgorithm': window.FireflyAlgorithm,
            'AntColonyOpt': window.AntColonyOpt,
            'HarmonySearch': window.HarmonySearch,
            'EvolutionStrategy': window.EvolutionStrategy,

            // Search algorithms
            'Rechenberg': window.Rechenberg,
            'AdaptiveRandomSearch': window.AdaptiveRandomSearch,  // backwards-compat alias
            'CoordinateDescent': window.CoordinateDescent,
            'PatternSearch': window.PatternSearch,
            'HillClimbing': window.HillClimbing,
            'GridSearch': window.GridSearch,

            // Discovered algorithms
            'Alloy': window.Alloy
        };

        const AlgorithmClass = algorithmMap[algorithmName];
        if (!AlgorithmClass) {
            console.error(`OptimizerFactory: Unknown algorithm "${algorithmName}"`);
            console.log('Available algorithms:', Object.keys(algorithmMap));
            throw new Error(`Unknown algorithm: ${algorithmName}`);
        }

        try {
            return new AlgorithmClass(objective, nTrials, nDim);
        } catch (error) {
            console.error(`OptimizerFactory: Error creating ${algorithmName}:`, error);
            throw error;
        }
    },

    /**
     * Get list of all available algorithm names
     * @returns {string[]} Array of algorithm names
     */
    getAvailableAlgorithms() {
        return [
            // PRIMA algorithms
            'PRIMA_UOBYQA', 'PRIMA_NEWUOA', 'PRIMA_BOBYQA',
            // SciPy algorithms
            'NelderMead', 'Powell', 'LBFGSB',
            // Evolutionary algorithms
            'DifferentialEvolution', 'ParticleSwarm', 'SimulatedAnnealing',
            'GeneticAlgorithm', 'RandomSearch', 'BayesianOpt', 'CMAEvolutionStrategy',
            'FireflyAlgorithm', 'AntColonyOpt', 'HarmonySearch', 'EvolutionStrategy',
            // Search algorithms
            'Rechenberg', 'CoordinateDescent', 'PatternSearch', 'HillClimbing', 'GridSearch',
            // Discovered algorithms
            'Alloy'
        ];
    },

    /**
     * Verify that all expected algorithms are available
     * @returns {boolean} True if all algorithms are available
     */
    verifyAll() {
        const expected = this.getAvailableAlgorithms();
        const missing = [];

        for (const name of expected) {
            try {
                const dummy = () => 0;
                const optimizer = this.create(name, dummy, 10, 2);
                if (!optimizer) {
                    missing.push(name);
                }
            } catch (error) {
                missing.push(name);
            }
        }

        if (missing.length > 0) {
            console.error('Missing algorithms:', missing);
            return false;
        }

        console.log(`All ${expected.length} algorithms verified successfully`);
        return true;
    }
};

// Make OptimizerFactory available globally
if (typeof window !== 'undefined') {
    window.OptimizerFactory = OptimizerFactory;
}

// Also make it available for Node.js environments
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { OptimizerFactory };
}