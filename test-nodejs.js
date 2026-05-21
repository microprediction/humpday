// Simple Node.js test to verify optimizer loading

const { OptimizerFactory } = require('./docs/js/optimizers.js');

console.log('🔍 Testing Node.js optimizer loading...');

try {
    if (typeof OptimizerFactory === 'undefined') {
        throw new Error('OptimizerFactory is not defined after loading optimizers.js');
    }

    console.log('✅ OptimizerFactory loaded successfully');
    console.log('✅ Available algorithms:', OptimizerFactory.getAvailableOptimizers().slice(0, 5));

    // Make available globally
    global.OptimizerFactory = OptimizerFactory;
    const Factory = OptimizerFactory;

    // Test function
    const testFunc = x => x[0]*x[0] + x[1]*x[1];

    console.log('🧪 Testing NelderMead...');

    // Create optimizer
    const optimizer = Factory.create('SciPy_NelderMead', testFunc, 50, 2);
    console.log('✅ Optimizer created:', optimizer.name);

    // Run optimization
    const result = optimizer.optimize();
    console.log('✅ Optimization result:', {
        success: true,
        x: result.bestX,
        fun: result.bestValue,
        nfev: result.evaluations,
        algorithm: 'SciPy_NelderMead'
    });

} catch (error) {
    console.log('❌ Error:', error.message);
    console.log('Stack:', error.stack);
}