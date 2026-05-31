/**
 * HumpDay JavaScript Optimization Algorithms - Modular Version
 *
 * Clean imports and exports for all algorithm families.
 * This replaces the monolithic 3131-line optimizers.js file with focused modules.
 */

// Import all algorithm modules
if (typeof module !== 'undefined' && module.exports) {
    // Node.js environment
    const { Optimizer, MathUtils } = require('./base-optimizer.js');
    const { PRIMA_UOBYQA, PRIMA_NEWUOA, PRIMA_BOBYQA } = require('./prima-algorithms.js');
    const { NelderMead, Powell, LBFGSB } = require('./scipy-algorithms.js');
    const { DifferentialEvolution, ParticleSwarm, SimulatedAnnealing, GeneticAlgorithm, RandomSearch,
            BayesianOpt, CMAEvolutionStrategy, FireflyAlgorithm, AntColonyOpt,
            HarmonySearch, EvolutionStrategy } = require('./evolutionary-algorithms.js');
    const { Rechenberg, AdaptiveRandomSearch, CoordinateDescent, PatternSearch, HillClimbing } = require('./search-algorithms.js');

    // Export everything
    module.exports = {
        // Base classes
        Optimizer,
        MathUtils,

        // PRIMA algorithms (state-of-the-art derivative-free)
        PRIMA_UOBYQA,
        PRIMA_NEWUOA,
        PRIMA_BOBYQA,

        // SciPy algorithms (classical methods)
        NelderMead,
        Powell,
        LBFGSB,

        // Evolutionary algorithms
        DifferentialEvolution,
        ParticleSwarm,
        SimulatedAnnealing,
        GeneticAlgorithm,
        RandomSearch,
        BayesianOpt,
        CMAEvolutionStrategy,
        FireflyAlgorithm,
        AntColonyOpt,
        HarmonySearch,
        EvolutionStrategy,

        // Search algorithms
        Rechenberg,
        AdaptiveRandomSearch,  // backwards-compat alias for Rechenberg
        CoordinateDescent,
        PatternSearch,
        HillClimbing,

        // Algorithm registry for factory pattern
        algorithms: {
            'PRIMA_UOBYQA': PRIMA_UOBYQA,
            'PRIMA_NEWUOA': PRIMA_NEWUOA,
            'PRIMA_BOBYQA': PRIMA_BOBYQA,
            'NelderMead': NelderMead,
            'Powell': Powell,
            'LBFGSB': LBFGSB,
            'DifferentialEvolution': DifferentialEvolution,
            'ParticleSwarm': ParticleSwarm,
            'SimulatedAnnealing': SimulatedAnnealing,
            'GeneticAlgorithm': GeneticAlgorithm,
            'RandomSearch': RandomSearch,
            'BayesianOpt': BayesianOpt,
            'CMAEvolutionStrategy': CMAEvolutionStrategy,
            'FireflyAlgorithm': FireflyAlgorithm,
            'AntColonyOpt': AntColonyOpt,
            'HarmonySearch': HarmonySearch,
            'EvolutionStrategy': EvolutionStrategy,
            'Rechenberg': Rechenberg,
            'AdaptiveRandomSearch': AdaptiveRandomSearch,  // backwards-compat alias
            'CoordinateDescent': CoordinateDescent,
            'PatternSearch': PatternSearch,
            'HillClimbing': HillClimbing
        }
    };
} else {
    // Browser environment
    // All classes are already available in global scope from individual module files
    // Create algorithm registry for factory pattern
    window.HumpDayOptimizers = {
        // PRIMA algorithms
        PRIMA_UOBYQA: window.PRIMA_UOBYQA,
        PRIMA_NEWUOA: window.PRIMA_NEWUOA,
        PRIMA_BOBYQA: window.PRIMA_BOBYQA,

        // SciPy algorithms
        NelderMead: window.NelderMead,
        Powell: window.Powell,
        LBFGSB: window.LBFGSB,

        // Evolutionary algorithms
        DifferentialEvolution: window.DifferentialEvolution,
        ParticleSwarm: window.ParticleSwarm,
        SimulatedAnnealing: window.SimulatedAnnealing,
        GeneticAlgorithm: window.GeneticAlgorithm,
        RandomSearch: window.RandomSearch,
        BayesianOpt: window.BayesianOpt,
        CMAEvolutionStrategy: window.CMAEvolutionStrategy,
        FireflyAlgorithm: window.FireflyAlgorithm,
        AntColonyOpt: window.AntColonyOpt,
        HarmonySearch: window.HarmonySearch,
        EvolutionStrategy: window.EvolutionStrategy,

        // Search algorithms
        Rechenberg: window.Rechenberg,
        AdaptiveRandomSearch: window.AdaptiveRandomSearch,  // backwards-compat alias
        CoordinateDescent: window.CoordinateDescent,
        PatternSearch: window.PatternSearch,
        HillClimbing: window.HillClimbing
    };

    // Factory function for creating optimizers by name
    window.createOptimizer = function(algorithmName, objective, nTrials, nDim) {
        const AlgorithmClass = window.HumpDayOptimizers[algorithmName];
        if (!AlgorithmClass) {
            throw new Error(`Unknown algorithm: ${algorithmName}`);
        }
        return new AlgorithmClass(objective, nTrials, nDim);
    };

    // OptimizerFactory compatible with current contest system
    window.OptimizerFactory = {
        create(algorithmName, objective, nTrials, nDim) {
            return window.createOptimizer(algorithmName, objective, nTrials, nDim);
        }
    };
}