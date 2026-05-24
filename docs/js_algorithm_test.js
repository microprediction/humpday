
// Test all 22 algorithms in JavaScript modular implementation
const expectedAlgorithms = [
    // PRIMA algorithms
    'PRIMA_UOBYQA', 'PRIMA_NEWUOA', 'PRIMA_BOBYQA',
    // SciPy algorithms
    'NelderMead', 'Powell', 'LBFGSB',
    // Evolutionary algorithms
    'DifferentialEvolution', 'ParticleSwarm', 'SimulatedAnnealing',
    'GeneticAlgorithm', 'RandomSearch', 'BayesianOpt', 'CMAEvolutionStrategy',
    'TabuSearch', 'FireflyAlgorithm', 'AntColonyOpt', 'HarmonySearch', 'EvolutionStrategy',
    // Search algorithms
    'AdaptiveRandomSearch', 'CoordinateDescent', 'PatternSearch', 'HillClimbing'
];

function sphereFunction(x) {
    return x.reduce((sum, xi) => sum + xi * xi, 0);
}

function testJavaScriptAlgorithms() {
    console.log('='.repeat(60));
    console.log('TESTING JAVASCRIPT ALGORITHMS (MODULAR)');
    console.log('='.repeat(60));

    console.log(`Expected ${expectedAlgorithms.length} algorithms in JavaScript`);

    let successCount = 0;
    const results = [];

    expectedAlgorithms.forEach(algorithmName => {
        try {
            // Test creating optimizer via factory
            const optimizer = OptimizerFactory.create(algorithmName, sphereFunction, 20, 2);

            // Test optimization
            const result = optimizer.optimize();

            if (result && typeof result.bestValue === 'number' && Array.isArray(result.bestX)) {
                console.log(`  ✓ ${algorithmName}: Working (best = ${result.bestValue.toFixed(6)})`);
                results.push({name: algorithmName, success: true, value: result.bestValue});
                successCount++;
            } else {
                console.log(`  ✗ ${algorithmName}: Invalid result format`);
                results.push({name: algorithmName, success: false, error: 'Invalid result'});
            }

        } catch (error) {
            console.log(`  ✗ ${algorithmName}: Error - ${error.message}`);
            results.push({name: algorithmName, success: false, error: error.message});
        }
    });

    console.log(`\nJavaScript Results: ${successCount}/${expectedAlgorithms.length} algorithms working`);

    return {successCount, total: expectedAlgorithms.length, results};
}

// Run the test (call this in browser console after loading modules)
// testJavaScriptAlgorithms();
