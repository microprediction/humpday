        let currentContest = null;
        let stochasticGenerator = null;

        // JavaScript loaded successfully
        console.log('JavaScript loaded');

        // Algorithm display names and links
        const algorithmInfo = {
            'UOBYQA (PDFO)': {
                displayName: 'UOBYQA (PDFO)',
                paper: 'https://doi.org/10.1007/s101070100290',
                referenceImpl: 'https://www.pdfo.net/',
                humpdayPython: 'https://github.com/microprediction/humpday/blob/main/humpday/optimizers/prima_algorithms.py#L16',
                humpdayJS: 'js/modules/prima-algorithms.js',
                resources: [
                    'https://arxiv.org/abs/2302.13246',
                    'https://github.com/pdfo/pdfo'
                ]
            },
            'NEWUOA (PDFO)': {
                displayName: 'NEWUOA (PDFO)',
                paper: 'https://www.damtp.cam.ac.uk/user/na/NA_papers/NA2004_08.pdf',
                referenceImpl: 'https://www.pdfo.net/',
                humpdayPython: 'https://github.com/microprediction/humpday/blob/main/humpday/optimizers/prima_algorithms.py#L84',
                humpdayJS: 'js/modules/prima-algorithms.js',
                resources: [
                    'https://arxiv.org/abs/2302.13246',
                    'https://github.com/pdfo/pdfo'
                ]
            },
            'BOBYQA (PDFO)': {
                displayName: 'BOBYQA (PDFO)',
                paper: 'https://www.damtp.cam.ac.uk/user/na/NA_papers/NA2009_06.pdf',
                referenceImpl: 'https://www.pdfo.net/',
                humpdayPython: 'https://github.com/microprediction/humpday/blob/main/humpday/optimizers/prima_algorithms.py#L148',
                humpdayJS: 'js/modules/prima-algorithms.js',
                resources: [
                    'https://arxiv.org/abs/2302.13246',
                    'https://github.com/pdfo/pdfo'
                ]
            },
            'Nelder-Mead (SciPy)': {
                displayName: 'Nelder-Mead (SciPy)',
                paper: 'https://doi.org/10.1093/comjnl/7.4.308',
                referenceImpl: 'https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html',
                humpdayPython: 'https://github.com/microprediction/humpday/blob/main/humpday/optimizers/scipy_algorithms.py#L17',
                humpdayJS: 'js/modules/scipy-algorithms.js',
                resources: [
                    'https://github.com/scipy/scipy',
                    'https://optimization.mccormick.northwestern.edu/index.php/Nelder-Mead_method'
                ]
            },
            'Powell (SciPy)': {
                displayName: 'Powell (SciPy)',
                paper: 'https://doi.org/10.1093/comjnl/7.2.155',
                referenceImpl: 'https://docs.scipy.org/doc/scipy/reference/optimize.minimize-powell.html',
                humpdayPython: 'https://github.com/microprediction/humpday/blob/main/humpday/optimizers/scipy_algorithms.py#L83',
                humpdayJS: 'js/modules/scipy-algorithms.js',
                resources: [
                    'https://github.com/scipy/scipy',
                    'https://optimization.mccormick.northwestern.edu/index.php/Powell%27s_method'
                ]
            },
            'Differential Evolution (SciPy)': {
                displayName: 'Differential Evolution (SciPy)',
                paper: 'https://doi.org/10.1023/A:1008202821328',
                referenceImpl: 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html',
                humpdayPython: 'https://github.com/microprediction/humpday/blob/main/humpday/optimizers/evolutionary_algorithms.py',
                humpdayJS: 'js/modules/evolutionary-algorithms.js',
                resources: [
                    'https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python/',
                    'https://optimization.mccormick.northwestern.edu/index.php/Differential_evolution'
                ]
            },
            'CMA-ES (pycma)': {
                displayName: 'CMA-ES (pycma)',
                paper: 'https://arxiv.org/abs/1604.00772',
                referenceImpl: 'https://pypi.org/project/cma/',
                humpdayPython: 'https://github.com/microprediction/humpday/blob/main/humpday/optimizers/evolutionary_algorithms.py',
                humpdayJS: 'js/modules/evolutionary-algorithms.js',
                resources: [
                    'https://cma-es.github.io/',
                    'https://blog.otoro.net/2017/10/29/visual-evolution-strategies/'
                ]
            },
            'Genetic Algorithm (DEAP)': {
                displayName: 'Genetic Algorithm (DEAP)',
                paper: 'https://doi.org/10.7551/mitpress/1090.001.0001',
                referenceImpl: 'https://deap.readthedocs.io/en/master/',
                humpdayPython: 'https://github.com/microprediction/humpday/blob/main/humpday/optimizers/evolutionary_algorithms.py',
                humpdayJS: 'js/modules/evolutionary-algorithms.js',
                resources: [
                    'https://towardsdatascience.com/introduction-to-genetic-algorithms-including-example-code-e396e98d8bf3',
                    'https://optimization.mccormick.northwestern.edu/index.php/Genetic_algorithm'
                ]
            },
            'Particle Swarm (PySwarm)': {
                displayName: 'Particle Swarm (PySwarm)',
                paper: 'https://doi.org/10.1109/ICNN.1995.488968',
                referenceImpl: 'https://pypi.org/project/pyswarm/',
                humpdayPython: 'https://github.com/microprediction/humpday/blob/main/humpday/optimizers/evolutionary_algorithms.py',
                humpdayJS: 'js/modules/evolutionary-algorithms.js',
                resources: [
                    'https://nathanrooy.github.io/posts/2016-08-17/simple-particle-swarm-optimization-with-python/',
                    'https://optimization.mccormick.northwestern.edu/index.php/Particle_swarm_optimization'
                ]
            },
            'Simulated Annealing (SciPy)': {
                displayName: 'Simulated Annealing (SciPy)',
                paper: 'https://doi.org/10.1126/science.220.4598.671',
                referenceImpl: 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html',
                humpdayPython: 'https://github.com/microprediction/humpday/blob/main/humpday/optimizers/evolutionary_algorithms.py',
                humpdayJS: 'js/modules/evolutionary-algorithms.js',
                resources: [
                    'https://optimization.mccormick.northwestern.edu/index.php/Simulated_annealing',
                    'https://towardsdatascience.com/optimization-techniques-simulated-annealing-d6a4785a1de7'
                ]
            },
            'BFGS (SciPy)': {
                displayName: 'BFGS (SciPy)',
                paper: 'https://doi.org/10.1007/BF01589116',
                referenceImpl: 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize',
                humpdayPython: 'https://github.com/microprediction/humpday/blob/main/humpday/optimizers/evolutionary_algorithms.py',
                humpdayJS: 'js/modules/evolutionary-algorithms.js',
                resources: [
                    'https://optimization.mccormick.northwestern.edu/index.php/BFGS_method',
                    'https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm'
                ]
            },
            'Bayesian Optimization (scikit-optimize)': {
                displayName: 'Bayesian Optimization (scikit-optimize)',
                paper: 'https://arxiv.org/abs/1012.2599',
                referenceImpl: 'https://distill.pub/2020/bayesian-optimization/',
                humpdayPython: 'https://github.com/microprediction/humpday/blob/main/humpday/optimizers/evolutionary_algorithms.py',
                humpdayJS: 'js/modules/evolutionary-algorithms.js',
                resources: [
                    'https://distill.pub/2020/bayesian-optimization/',
                    'https://optimization.mccormick.northwestern.edu/index.php/Bayesian_optimization'
                ]
            },
            'Random Search (scikit-learn)': {
                displayName: 'Random Search (scikit-learn)',
                paper: 'https://jmlr.org/papers/v13/bergstra12a.html',
                referenceImpl: 'https://en.wikipedia.org/wiki/Random_search',
                humpdayPython: 'https://github.com/microprediction/humpday/blob/main/humpday/optimizers/evolutionary_algorithms.py',
                humpdayJS: 'js/modules/evolutionary-algorithms.js',
                resources: [
                    'https://optimization.mccormick.northwestern.edu/index.php/Random_search',
                    'https://towardsdatascience.com/hyperparameter-tuning-c5619e7e6624'
                ]
            },
            'Adaptive Random Search (nlopt)': {
                displayName: 'Adaptive Random Search (nlopt)',
                paper: 'https://doi.org/10.1007/BF01581033',
                referenceImpl: 'https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#controlled-random-search-crs-family',
                humpdayPython: 'https://github.com/microprediction/humpday/blob/main/humpday/optimizers/evolutionary_algorithms.py',
                humpdayJS: 'js/modules/evolutionary-algorithms.js',
                resources: [
                    'https://optimization.mccormick.northwestern.edu/index.php/Random_search',
                    'https://link.springer.com/article/10.1007/BF01581033'
                ]
            },
            'Coordinate Descent (scikit-learn)': {
                displayName: 'Coordinate Descent (scikit-learn)',
                paper: 'https://doi.org/10.1007/s10107-015-0892-3',
                referenceImpl: 'https://scikit-learn.org/stable/modules/linear_model.html#coordinate-descent',
                humpdayPython: 'https://github.com/microprediction/humpday/blob/main/humpday/optimizers/evolutionary_algorithms.py',
                humpdayJS: 'js/modules/evolutionary-algorithms.js',
                resources: [
                    'https://optimization.mccormick.northwestern.edu/index.php/Coordinate_descent',
                    'https://web.stanford.edu/~boyd/papers/prox_algs.html'
                ]
            },
            'Pattern Search (SciPy)': {
                displayName: 'Pattern Search (SciPy)',
                paper: 'https://doi.org/10.1137/S1052623493250780',
                referenceImpl: 'https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cobyla.html',
                humpdayPython: 'https://github.com/microprediction/humpday/blob/main/humpday/optimizers/evolutionary_algorithms.py',
                humpdayJS: 'js/modules/evolutionary-algorithms.js',
                resources: [
                    'https://optimization.mccormick.northwestern.edu/index.php/Pattern_search',
                    'https://epubs.siam.org/doi/10.1137/S1052623496303470'
                ]
            },
            'Hill Climbing (SciPy)': {
                displayName: 'Hill Climbing (SciPy)',
                paper: 'https://doi.org/10.1007/s00521-016-2328-2',
                referenceImpl: 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html',
                humpdayPython: 'https://github.com/microprediction/humpday/blob/main/humpday/optimizers/evolutionary_algorithms.py',
                humpdayJS: 'js/modules/evolutionary-algorithms.js',
                resources: [
                    'https://towardsdatascience.com/hill-climbing-optimization-algorithm-8ddd2d8d6b6d',
                    'https://www.geeksforgeeks.org/introduction-hill-climbing-artificial-intelligence/'
                ]
            },
            'Tabu Search (SciPy)': {
                displayName: 'Tabu Search (SciPy)',
                paper: 'https://doi.org/10.1287/ijoc.1.3.190',
                referenceImpl: 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html',
                humpdayPython: 'https://github.com/microprediction/humpday/blob/main/humpday/optimizers/evolutionary_algorithms.py',
                humpdayJS: 'js/modules/evolutionary-algorithms.js',
                resources: [
                    'https://optimization.mccormick.northwestern.edu/index.php/Tabu_search',
                    'https://www.researchgate.net/publication/227061666_Tabu_Search_A_Tutorial'
                ]
            },
            'Firefly Algorithm (SciPy)': {
                displayName: 'Firefly Algorithm (SciPy)',
                paper: 'https://doi.org/10.1007/978-3-642-04944-6_14',
                referenceImpl: 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html',
                humpdayPython: 'https://github.com/microprediction/humpday/blob/main/humpday/optimizers/evolutionary_algorithms.py',
                humpdayJS: 'js/modules/evolutionary-algorithms.js',
                resources: [
                    'https://www.mathworks.com/matlabcentral/fileexchange/29693-firefly-algorithm',
                    'https://towardsdatascience.com/firefly-algorithm-an-overview-e2de0b7c4a75'
                ]
            },
            'Ant Colony Optimization (acopy)': {
                displayName: 'Ant Colony Optimization (acopy)',
                paper: 'https://doi.org/10.1109/3477.484436',
                referenceImpl: 'https://pypi.org/project/acopy/',
                humpdayPython: 'https://github.com/microprediction/humpday/blob/main/humpday/optimizers/evolutionary_algorithms.py',
                humpdayJS: 'js/modules/evolutionary-algorithms.js',
                resources: [
                    'https://optimization.mccormick.northwestern.edu/index.php/Ant_colony_optimization',
                    'https://towardsdatascience.com/ant-colony-optimization-aco-8c0d9de52e1b'
                ]
            },
            'Harmony Search (pyHarmonySearch)': {
                displayName: 'Harmony Search (pyHarmonySearch)',
                paper: 'https://doi.org/10.1177/003754970107600201',
                referenceImpl: 'https://pypi.org/project/pyHarmonySearch/',
                humpdayPython: 'https://github.com/microprediction/humpday/blob/main/humpday/optimizers/evolutionary_algorithms.py',
                humpdayJS: 'js/modules/evolutionary-algorithms.js',
                resources: [
                    'https://optimization.mccormick.northwestern.edu/index.php/Harmony_search',
                    'https://towardsdatascience.com/harmony-search-optimization-algorithm-b8e8e8d0a2d2'
                ]
            },
            'Evolution Strategy (DEAP)': {
                displayName: 'Evolution Strategy (DEAP)',
                paper: 'https://doi.org/10.1007/978-3-662-43505-2_44',
                referenceImpl: 'https://deap.readthedocs.io/en/master/api/algo.html#evolution-strategies',
                humpdayPython: 'https://github.com/microprediction/humpday/blob/main/humpday/optimizers/evolutionary_algorithms.py',
                humpdayJS: 'js/modules/evolutionary-algorithms.js',
                resources: [
                    'https://optimization.mccormick.northwestern.edu/index.php/Evolution_strategy',
                    'https://towardsdatascience.com/evolution-strategies-an-alternative-to-neural-networks-6af3f5b72ac9'
                ]
            }
        };

        function getAlgorithmDisplayName(internalName) {
            return algorithmInfo[internalName]?.displayName || internalName.replace('_', ' ');
        }

        function getAlgorithmLinks(algorithmName) {
            const info = algorithmInfo[algorithmName];
            if (!info) return '';

            let links = [];

            if (info.paper) {
                links.push(`<a href="${info.paper}" target="_blank" title="Original Paper">📄 Paper</a>`);
            }

            if (info.referenceImpl) {
                links.push(`<a href="${info.referenceImpl}" target="_blank" title="Reference Implementation">🔗 Reference</a>`);
            }

            if (info.humpdayPython) {
                links.push(`<a href="${info.humpdayPython}" target="_blank" title="Humpday Python Wrapper">🐍 Humpday</a>`);
            }

            if (info.humpdayJS) {
                links.push(`<a href="${info.humpdayJS}" title="Humpday JavaScript Port (Local File)">🌐 JS Port</a>`);
            }

            return links.length > 0 ? `<div style="font-size: 0.8em; margin-top: 4px;">${links.join(' | ')}</div>` : '';
        }


        function getAlgorithmTimeout(algorithmName, dimensions, baseTimeout) {
            // Algorithms known to be slow in high dimensions get shorter timeouts
            const slowAlgorithms = {
                'BayesianOpt': 0.3,                     // Very slow - cut timeout to 30%
                'CMAEvolutionStrategy': 0.7,            // Can be slow with large populations
                'GeneticAlgorithm': 0.8,                // Large populations can be slow
                'ParticleSwarm': 0.8,                   // Large swarms can be slow
                'TabuSearch': 0.6,                      // Memory operations can be slow
                'FireflyAlgorithm': 0.7,                // Many distance calculations
                'AntColonyOpt': 0.6,                    // Complex pheromone updates
                'HarmonySearch': 0.8                    // Memory search can be slow
            };

            // Fast algorithms get normal or longer timeouts
            const fastAlgorithms = {
                'RandomSearch': 1.5,                    // Very fast, can afford more time
                'HillClimbing': 1.2,                    // Simple and fast
                'CoordinateDescent': 1.2,               // Simple coordinate moves
                'AdaptiveRandomSearch': 1.3             // Fast with adaptation
            };

            // Dimension-based scaling - higher dimensions need more time but capped
            const dimensionScaling = Math.min(2.0, 1.0 + (dimensions - 2) * 0.05);

            let multiplier = 1.0;
            if (slowAlgorithms[algorithmName]) {
                multiplier = slowAlgorithms[algorithmName];
            } else if (fastAlgorithms[algorithmName]) {
                multiplier = fastAlgorithms[algorithmName];
            }

            // Apply both algorithm and dimension scaling, but cap at reasonable limits
            const timeout = baseTimeout * multiplier * dimensionScaling;

            // Hard limits: minimum 1 second, maximum 15 seconds
            return Math.min(15000, Math.max(1000, timeout));
        }

        function showResources(algorithmName) {
            const info = algorithmInfo[algorithmName];
            if (!info || !info.resources) return;

            // Map algorithm names to actual HTML filenames
            const algorithmPageMap = {
                'UOBYQA (PDFO)': 'uobyqa.html',
                'NEWUOA (PDFO)': 'newuoa.html',
                'BOBYQA (PDFO)': 'bobyqa.html',
                'Nelder-Mead (SciPy)': 'nelder-mead.html',
                'Powell (SciPy)': 'powell.html',
                'Differential Evolution (SciPy)': 'differential-evolution.html',
                'CMA-ES (pycma)': 'cma-evolution-strategy.html',
                'Genetic Algorithm (DEAP)': 'genetic-algorithm.html',
                'Particle Swarm (PySwarm)': 'particle-swarm.html',
                'Simulated Annealing (SciPy)': 'simulated-annealing.html',
                'BFGS (SciPy)': 'lbfgsb.html',
                'Bayesian Optimization (scikit-optimize)': 'bayesian-optimization.html',
                'Random Search (scikit-learn)': 'random-search.html',
                'Adaptive Random Search (nlopt)': 'adaptive-random-search.html',
                'Coordinate Descent (scikit-learn)': 'coordinate-descent.html',
                'Pattern Search (SciPy)': 'pattern-search.html',
                'Hill Climbing (SciPy)': 'hill-climbing.html',
                'Tabu Search (SciPy)': 'tabu-search.html',
                'Firefly Algorithm (SciPy)': 'firefly-algorithm.html',
                'Ant Colony Optimization (acopy)': 'ant-colony-optimization.html',
                'Harmony Search (pyHarmonySearch)': 'harmony-search.html',
                'Evolution Strategy (DEAP)': 'cma-evolution-strategy.html'
            };

            const pageFile = algorithmPageMap[algorithmName];
            if (pageFile) {
                const algorithmPageUrl = `algorithms/${pageFile}`;
                window.open(algorithmPageUrl, '_blank');
            } else {
                console.warn(`No page mapping found for algorithm: ${algorithmName}`);
            }
        }

        function resetContest() {
            console.log('Resetting contest completely...');

            // Hide all contest-related areas
            document.getElementById('contestArea').style.display = 'none';
            document.getElementById('interpretation').style.display = 'none';

            // Clear ALL content thoroughly
            const progressSection = document.getElementById('progressSection');
            progressSection.innerHTML = '';
            progressSection.style.display = 'block'; // Ensure it's visible when needed later

            const leaderboardBody = document.getElementById('leaderboardBody');
            leaderboardBody.innerHTML = '';

            const contestInfo = document.getElementById('contestInfo');
            contestInfo.innerHTML = '';

            const interpretationDetails = document.getElementById('interpretationDetails');
            interpretationDetails.innerHTML = '';

            // Reset optimizer Elo ratings and status completely
            optimizers.forEach(optimizer => {
                optimizer.elo = 1500;
                optimizer.status = 'waiting';
                optimizer.testsCompleted = 0;
            });

            // Clear any running timers or intervals
            if (window.contestTimeout) {
                clearTimeout(window.contestTimeout);
                window.contestTimeout = null;
            }

            // Reset all contest state variables
            currentContest = null;
            stochasticGenerator = null;
            leaderboardUpdatePending = false;

            // Scroll back to top
            window.scrollTo({ top: 0, behavior: 'smooth' });

            console.log('Contest reset complete - ready for new challenge');
        }

        function toggleSurface(surfaceIndex) {
            const surfaceDiv = document.getElementById(`surface-${surfaceIndex}`);
            if (surfaceDiv) {
                surfaceDiv.classList.toggle('collapsed');
            }
        }

        function collapseSurface(surfaceIndex, winner, winnerScore) {
            const surfaceDiv = document.getElementById(`surface-${surfaceIndex}`);
            const summaryDiv = document.getElementById(`surface-${surfaceIndex}-summary`);

            if (surfaceDiv && summaryDiv) {
                // Create summary
                summaryDiv.innerHTML = `Complete - Winner: ${winner} (${winnerScore.toFixed(6)})`;

                // Auto-collapse after a short delay
                setTimeout(() => {
                    surfaceDiv.classList.add('collapsed');
                }, 1000);
            }
        }
        let optimizers = [
            // PRIMA algorithms (state-of-the-art derivative-free)
            { name: 'UOBYQA (PDFO)', internalName: 'PRIMA_UOBYQA', elo: 1500, status: 'waiting', testsCompleted: 0 },
            { name: 'NEWUOA (PDFO)', internalName: 'PRIMA_NEWUOA', elo: 1500, status: 'waiting', testsCompleted: 0 },
            { name: 'BOBYQA (PDFO)', internalName: 'PRIMA_BOBYQA', elo: 1500, status: 'waiting', testsCompleted: 0 },

            // Classical derivative-free methods
            { name: 'Nelder-Mead (SciPy)', internalName: 'SciPy_NelderMead', elo: 1500, status: 'waiting', testsCompleted: 0 },
            { name: 'Powell (SciPy)', internalName: 'SciPy_Powell', elo: 1500, status: 'waiting', testsCompleted: 0 },
            { name: 'BFGS (SciPy)', internalName: 'SciPy_BFGS', elo: 1500, status: 'waiting', testsCompleted: 0 },

            // Evolutionary algorithms
            { name: 'Differential Evolution (SciPy)', internalName: 'DifferentialEvolution', elo: 1500, status: 'waiting', testsCompleted: 0 },
            { name: 'CMA-ES (pycma)', internalName: 'CMAEvolutionStrategy', elo: 1500, status: 'waiting', testsCompleted: 0 },
            { name: 'Genetic Algorithm (DEAP)', internalName: 'GeneticAlgorithm', elo: 1500, status: 'waiting', testsCompleted: 0 },

            // Swarm intelligence
            { name: 'Particle Swarm (PySwarm)', internalName: 'ParticleSwarm', elo: 1500, status: 'waiting', testsCompleted: 0 },

            // Metaheuristics
            { name: 'Simulated Annealing (SciPy)', internalName: 'SimulatedAnnealing', elo: 1500, status: 'waiting', testsCompleted: 0 },

            // Modern Bayesian methods
            { name: 'Bayesian Optimization (scikit-optimize)', internalName: 'BayesianOpt', elo: 1500, status: 'waiting', testsCompleted: 0 },

            // Baseline methods
            { name: 'Random Search (scikit-learn)', internalName: 'RandomSearch', elo: 1500, status: 'waiting', testsCompleted: 0 },

            // Additional gradient-free methods
            { name: 'Adaptive Random Search (nlopt)', internalName: 'AdaptiveRandomSearch', elo: 1500, status: 'waiting', testsCompleted: 0 },
            { name: 'Coordinate Descent (scikit-learn)', internalName: 'CoordinateDescent', elo: 1500, status: 'waiting', testsCompleted: 0 },
            { name: 'Pattern Search (SciPy)', internalName: 'PatternSearch', elo: 1500, status: 'waiting', testsCompleted: 0 },
            { name: 'Hill Climbing (SciPy)', internalName: 'HillClimbing', elo: 1500, status: 'waiting', testsCompleted: 0 },
            { name: 'Tabu Search (SciPy)', internalName: 'TabuSearch', elo: 1500, status: 'waiting', testsCompleted: 0 },
            { name: 'Firefly Algorithm (SciPy)', internalName: 'FireflyAlgorithm', elo: 1500, status: 'waiting', testsCompleted: 0 },
            { name: 'Ant Colony Optimization (acopy)', internalName: 'AntColonyOpt', elo: 1500, status: 'waiting', testsCompleted: 0 },
            { name: 'Harmony Search (pyHarmonySearch)', internalName: 'HarmonySearch', elo: 1500, status: 'waiting', testsCompleted: 0 },
            { name: 'Evolution Strategy (DEAP)', internalName: 'EvolutionStrategy', elo: 1500, status: 'waiting', testsCompleted: 0 }
        ];

        let selectedProblemConfig = null;

        function selectProblem(element) {
            try {
                // Get problem configuration
                const configStr = element.getAttribute('data-config');
                if (!configStr) {
                    alert('Problem configuration not found.');
                    return;
                }

                const config = JSON.parse(configStr);
                selectedProblemConfig = config;

                // Get problem name from tooltip
                const tooltip = element.querySelector('.challenge-tooltip');
                const problemName = tooltip ? tooltip.textContent : `${config.dimensions}D ${config.surfaceType}`;

                // Show customization panel
                showCustomizationPanel(config, problemName);

            } catch (error) {
                console.error('Error selecting problem:', error);
                alert('Failed to select problem: ' + error.message);
            }
        }

        function showCustomizationPanel(config, problemName) {
            // Update problem info
            const infoDiv = document.getElementById('selectedProblemInfo');
            const functionClassification = {
                'sphere': 'Unimodal, smooth, well-conditioned (50 stochastic variants)',
                'rosenbrock': 'Unimodal, narrow curved valley, ill-conditioned (50 rotated/shifted variants)',
                'rastrigin': 'Highly multimodal, regular local optima (50 frequency/scale variants)',
                'ackley': 'Multimodal with global structure, steep near optimum (50 parameter variants)',
                'griewank': 'Multimodal, product structure, scale-dependent (50 rotation/scale variants)',
                'mixed': 'Tests robustness across all function types (10 of each: sphere, rastrigin, rosenbrock, ackley, griewank)',
                'noisy': 'Function evaluations corrupted with noise (50 variants with different noise levels)'
            };

            infoDiv.innerHTML = `
                <strong>Selected:</strong> ${problemName}<br>
                <strong>Function:</strong> ${functionClassification[config.surfaceType] || 'Specialized test case'}<br>
                <strong>Dimensions:</strong> ${config.dimensions}D | <strong>Base Difficulty:</strong> ${config.difficulty}
            `;

            // Set default trial budget based on problem
            document.getElementById('customTrials').value = config.budget;

            // Generate function-specific difficulty controls
            showDifficultyControls(config.surfaceType);

            // Show the customization panel
            document.getElementById('customizationPanel').style.display = 'block';

            // Scroll to customization panel
            document.getElementById('customizationPanel').scrollIntoView({ behavior: 'smooth' });
        }

        function getDifficultyDescription(surfaceType, config) {
            if (!config || Object.keys(config).length === 0) return '';

            switch(surfaceType) {
                case 'griewank':
                    const rotIntensity = config.rotationIntensity;
                    const level = rotIntensity < 0.5 ? 'low' : rotIntensity > 1.2 ? 'high' : 'medium';
                    return `, ${level} rotation`;
                case 'rastrigin':
                    const freq = config.modalFrequency;
                    const freqLevel = freq < 0.8 ? 'low' : freq > 1.5 ? 'high' : 'medium';
                    return `, ${freqLevel} frequency`;
                case 'rosenbrock':
                    const cond = config.conditioningFactor;
                    const condLevel = cond < 80 ? 'wide valley' : cond > 150 ? 'narrow valley' : 'standard';
                    return `, ${condLevel}`;
                case 'ackley':
                    const struct = config.structureIntensity;
                    const structLevel = struct < 0.9 ? 'clear' : struct > 1.1 ? 'deceptive' : 'standard';
                    return `, ${structLevel} structure`;
                default:
                    const noise = config.noiseLevel;
                    if (noise === 0) return ', no noise';
                    const noisePercent = Math.round(noise * 100);
                    return `, ${noisePercent}% noise`;
            }
        }

        function createCustomizedSurfaceGenerator(config) {
            const generator = new StochasticSurfaceGenerator(Date.now());

            // Override specific parameters based on user configuration
            if (config.rotationIntensity !== undefined) {
                generator.scaleFactor = config.rotationIntensity;
            }
            if (config.modalFrequency !== undefined) {
                generator.modalFrequency = config.modalFrequency;
            }
            if (config.conditioningFactor !== undefined) {
                generator.conditioningFactor = config.conditioningFactor;
            }
            if (config.structureIntensity !== undefined) {
                generator.scaleFactor = config.structureIntensity;
            }
            if (config.noiseLevel !== undefined) {
                generator.noiseLevel = config.noiseLevel;
            }

            return generator;
        }

        function extractDifficultyConfig(surfaceType) {
            const config = {};

            switch(surfaceType) {
                case 'griewank':
                    const rotationLevel = document.getElementById('rotationLevel')?.value || 'medium';
                    config.rotationIntensity = rotationLevel === 'low' ? 0.3 :
                                             rotationLevel === 'high' ? 1.5 : 1.0;
                    break;
                case 'rastrigin':
                    const frequencyLevel = document.getElementById('frequencyLevel')?.value || 'medium';
                    config.modalFrequency = frequencyLevel === 'low' ? 0.5 :
                                          frequencyLevel === 'high' ? 2.0 : 1.0;
                    break;
                case 'rosenbrock':
                    const conditioningLevel = document.getElementById('conditioningLevel')?.value || 'medium';
                    config.conditioningFactor = conditioningLevel === 'low' ? 50.0 :
                                               conditioningLevel === 'high' ? 200.0 : 100.0;
                    break;
                case 'ackley':
                    const structureLevel = document.getElementById('structureLevel')?.value || 'medium';
                    config.structureIntensity = structureLevel === 'clear' ? 0.7 :
                                               structureLevel === 'deceptive' ? 1.3 : 1.0;
                    break;
                default:
                    const noiseLevel = document.getElementById('noiseLevel')?.value || 'low';
                    config.noiseLevel = noiseLevel === 'none' ? 0.0 :
                                       noiseLevel === 'medium' ? 0.03 :
                                       noiseLevel === 'high' ? 0.05 : 0.01;
                    break;
            }

            return config;
        }

        function showDifficultyControls(surfaceType) {
            const controlsDiv = document.getElementById('difficultyControls');

            switch(surfaceType) {
                case 'griewank':
                    controlsDiv.innerHTML = `
                        <label style="font-weight: 600; display: block; margin-bottom: 8px; color: #495057;">Rotation Intensity:</label>
                        <select id="rotationLevel" style="width: 100%; padding: 8px; border: 1px solid #ced4da; border-radius: 4px;">
                            <option value="low">Low (minor rotations)</option>
                            <option value="medium" selected>Medium (moderate rotations)</option>
                            <option value="high">High (strong rotations)</option>
                        </select>
                        <small style="color: #6c757d;">controls coordinate system alignment</small>
                    `;
                    break;
                case 'rastrigin':
                    controlsDiv.innerHTML = `
                        <label style="font-weight: 600; display: block; margin-bottom: 8px; color: #495057;">Modal Frequency:</label>
                        <select id="frequencyLevel" style="width: 100%; padding: 8px; border: 1px solid #ced4da; border-radius: 4px;">
                            <option value="low">Low (fewer local optima)</option>
                            <option value="medium" selected>Medium (standard)</option>
                            <option value="high">High (many local optima)</option>
                        </select>
                        <small style="color: #6c757d;">controls number of local optima</small>
                    `;
                    break;
                case 'rosenbrock':
                    controlsDiv.innerHTML = `
                        <label style="font-weight: 600; display: block; margin-bottom: 8px; color: #495057;">Valley Narrowness:</label>
                        <select id="conditioningLevel" style="width: 100%; padding: 8px; border: 1px solid #ced4da; border-radius: 4px;">
                            <option value="low">Low (wide valley)</option>
                            <option value="medium" selected>Medium (standard)</option>
                            <option value="high">High (narrow valley)</option>
                        </select>
                        <small style="color: #6c757d;">controls problem conditioning</small>
                    `;
                    break;
                case 'ackley':
                    controlsDiv.innerHTML = `
                        <label style="font-weight: 600; display: block; margin-bottom: 8px; color: #495057;">Global Structure:</label>
                        <select id="structureLevel" style="width: 100%; padding: 8px; border: 1px solid #ced4da; border-radius: 4px;">
                            <option value="clear">Clear (easier to navigate)</option>
                            <option value="medium" selected>Medium (standard)</option>
                            <option value="deceptive">Deceptive (misleading)</option>
                        </select>
                        <small style="color: #6c757d;">controls deceptiveness of landscape</small>
                    `;
                    break;
                default:
                    controlsDiv.innerHTML = `
                        <label style="font-weight: 600; display: block; margin-bottom: 8px; color: #495057;">Noise Level:</label>
                        <select id="noiseLevel" style="width: 100%; padding: 8px; border: 1px solid #ced4da; border-radius: 4px;">
                            <option value="none">None (deterministic)</option>
                            <option value="low" selected>Low (1% noise)</option>
                            <option value="medium">Medium (3% noise)</option>
                            <option value="high">High (5% noise)</option>
                        </select>
                        <small style="color: #6c757d;">adds measurement uncertainty</small>
                    `;
                    break;
            }
        }

        function cancelCustomization() {
            document.getElementById('customizationPanel').style.display = 'none';
            selectedProblemConfig = null;
        }

        function runCustomizedContest() {
            if (!selectedProblemConfig) {
                alert('No problem selected');
                return;
            }

            // Get customized parameters
            const customTrials = parseInt(document.getElementById('customTrials').value);

            // Extract function-specific difficulty parameters
            const surfaceConfig = extractDifficultyConfig(selectedProblemConfig.surfaceType);

            // Create final interpretation with custom parameters
            const interpretation = {
                dimensions: selectedProblemConfig.dimensions,
                surfaceType: selectedProblemConfig.surfaceType,
                difficulty: selectedProblemConfig.difficulty,
                domain: 'unit_hypercube',
                budget: customTrials,
                template_used: 'customized',
                confidence: 1.0,
                surface_config: surfaceConfig,
                enable_visualization: selectedProblemConfig.dimensions === 2
            };

            // Hide customization panel
            document.getElementById('customizationPanel').style.display = 'none';

            // Start the contest with the original startContest logic
            const difficultyDesc = getDifficultyDescription(selectedProblemConfig.surfaceType, surfaceConfig);
            const problemName = `${selectedProblemConfig.dimensions}D ${selectedProblemConfig.surfaceType} (${customTrials} trials${difficultyDesc})`;
            showInterpretation(interpretation, problemName);

            // Start contest
            setTimeout(() => {
                runRealContest(interpretation);
            }, 500);
        }

        function startContest(element) {
            console.log('Contest clicked!', element);

            try {
                // Get problem configuration directly from the clicked element
                const configStr = element.getAttribute('data-config');
                console.log('Config string:', configStr);

                if (!configStr) {
                    console.error('No data-config found on element');
                    alert('Problem configuration not found.');
                    return;
                }

                const config = JSON.parse(configStr);
                console.log('Parsed config:', config);

                // Use custom trial budget if different from default
                const customTrials = parseInt(document.getElementById('customTrials').value);
                const finalBudget = customTrials || config.budget;

                // Get problem name from tooltip
                const tooltip = element.querySelector('.challenge-tooltip');
                let problemName = tooltip ? tooltip.textContent : `${config.dimensions}D ${config.surfaceType}`;

                if (finalBudget !== config.budget) {
                    problemName = `${problemName} (${finalBudget} trials)`;
                }

                const interpretation = {
                    dimensions: config.dimensions,
                    surfaceType: config.surfaceType,
                    difficulty: config.difficulty,
                    domain: 'unit_hypercube',
                    budget: finalBudget,
                    template_used: 'predefined',
                    confidence: 1.0,
                    surface_config: {},
                    enable_visualization: config.dimensions === 2
                };

                console.log('Starting contest with:', interpretation);

                // Show contest area
                showInterpretation(interpretation, problemName);

                // Start contest immediately
                setTimeout(() => {
                    runRealContest(interpretation);
                }, 500);

            } catch (error) {
                console.error('Error starting contest:', error);
                alert('Failed to start contest: ' + error.message);
            }
        }

        function showInterpretation(interpretation, description) {
            const interpDiv = document.getElementById('interpretation');
            const detailsDiv = document.getElementById('interpretationDetails');

            const functionClassification = {
                'sphere': 'Unimodal, smooth, well-conditioned (50 stochastic variants)',
                'rosenbrock': 'Unimodal, narrow curved valley, ill-conditioned (50 rotated/shifted variants)',
                'rastrigin': 'Highly multimodal, regular local optima (50 frequency/scale variants)',
                'ackley': 'Multimodal with global structure, steep near optimum (50 parameter variants)',
                'griewank': 'Multimodal, product structure, scale-dependent (50 rotation/scale variants)',
                'mixed': 'Tests robustness across all function types (10 of each: sphere, rastrigin, rosenbrock, ackley, griewank)',
                'noisy': 'Function evaluations corrupted with noise (50 variants with different noise levels)'
            };

            detailsDiv.innerHTML = `
                <strong>Selected:</strong> "${description}"<br><br>
                <strong>Contest Details:</strong><br>
                • <strong>Function type:</strong> ${interpretation.surfaceType} - ${functionClassification[interpretation.surfaceType] || 'Specialized test case'}<br>
                • <strong>Dimensions:</strong> ${interpretation.dimensions}D optimization problem<br>
                • <strong>Test instances:</strong> 50 stochastic variants of ${interpretation.surfaceType} landscape<br>
                • <strong>Difficulty:</strong> ${interpretation.difficulty} (based on mathematical properties: modality, conditioning, ruggedness)<br>
                • <strong>Domain:</strong> Unit hypercube [0,1]^${interpretation.dimensions}<br>
                • <strong>Evaluation budget:</strong> ${interpretation.budget} evaluations per algorithm per surface (user controlled)
            `;

            interpDiv.style.display = 'block';

            // Show contest area
            const contestArea = document.getElementById('contestArea');
            contestArea.style.display = 'block';

            const contestInfo = document.getElementById('contestInfo');
            const functionTypeDesc = interpretation.surfaceType === 'mixed' ?
                'mixed function types' : `${interpretation.surfaceType} functions`;

            contestInfo.innerHTML = `
                <strong>Challenge:</strong> ${interpretation.dimensions}D ${functionTypeDesc} (${interpretation.difficulty})<br>
                <strong>Format:</strong> 50 stochastic instances, ${interpretation.budget} evaluations per algorithm per instance<br>
                <strong>Competitors:</strong> ${optimizers.length} derivative-free optimization algorithms
            `;

            updateLeaderboard();
        }

        let leaderboardUpdatePending = false;

        function updateLeaderboard(force = false) {
            if (leaderboardUpdatePending && !force) return;
            leaderboardUpdatePending = true;

            // Debounce updates to reduce flicker
            setTimeout(() => {
                const tbody = document.getElementById('leaderboardBody');
                optimizers.sort((a, b) => b.elo - a.elo);

                // Check if any ratings have been updated from initial 1500
                const ratingsUpdated = optimizers.some(opt => opt.elo !== 1500);

                tbody.innerHTML = optimizers.map((opt, index) => `
                    <tr class="${ratingsUpdated && index < 3 ? `rank-${index + 1}` : ''}">
                        <td>
                            <span class="rank-badge">
                                ${index + 1}
                            </span>
                        </td>
                        <td>
                            <div style="display: flex; flex-direction: column; gap: 2px;">
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <strong>
                                        <a href="${algorithmInfo[opt.name]?.explanation || '#'}" target="_blank"
                                           style="color: #333; text-decoration: none; border-bottom: 1px dotted #ccc;"
                                           title="Learn about ${getAlgorithmDisplayName(opt.name)}">
                                            ${getAlgorithmDisplayName(opt.name)}
                                        </a>
                                    </strong>
                                    ${algorithmInfo[opt.name]?.resources ? `<a href="#" onclick="showResources('${opt.name}'); return false;" style="color: #6f42c1; text-decoration: none; font-size: 0.85em;">More</a>` : ''}
                                </div>
                                ${getAlgorithmLinks(opt.name)}
                            </div>
                        </td>
                        <td><span class="elo-rating">${Math.round(opt.elo)}</span></td>
                        <td>
                            <span class="status-indicator status-${opt.status}"></span>
                            ${(currentContest && currentContest.challengeSurfaces) ? `${opt.testsCompleted}/${currentContest.challengeSurfaces.length} tests` : 'pending'}
                        </td>
                    </tr>
                `).join('');

                leaderboardUpdatePending = false;
            }, force ? 0 : 200);
        }

        function runRealContest(interpretation) {
            // Generate many stochastic surfaces for fair competition
            const nSurfaces = 50; // Comprehensive evaluation with 50 surfaces

            // Create generator with custom difficulty parameters if provided
            if (interpretation.surface_config && Object.keys(interpretation.surface_config).length > 0) {
                stochasticGenerator = createCustomizedSurfaceGenerator(interpretation.surface_config);
            } else {
                stochasticGenerator = new StochasticSurfaceGenerator(Date.now());
            }

            const challengeSurfaces = [];

            // Generate surfaces based on the actual problem type selected
            const primaryType = interpretation.surfaceType;

            for (let i = 0; i < nSurfaces; i++) {
                let surfaceFunc;

                // Generate the actual surface type that was selected
                switch(primaryType) {
                    case 'sphere':
                        surfaceFunc = stochasticGenerator.stochasticSphere(`sphere_${i}`);
                        break;
                    case 'rastrigin':
                        surfaceFunc = stochasticGenerator.stochasticRastrigin(`rastrigin_${i}`);
                        break;
                    case 'rosenbrock':
                        surfaceFunc = stochasticGenerator.stochasticRosenbrock(`rosenbrock_${i}`);
                        break;
                    case 'ackley':
                        surfaceFunc = stochasticGenerator.stochasticAckley(`ackley_${i}`);
                        break;
                    case 'griewank':
                        surfaceFunc = stochasticGenerator.stochasticGriewank(`griewank_${i}`);
                        break;
                    case 'mixed':
                        // Only use mixed for problems explicitly labeled as mixed
                        const mixedType = ['sphere', 'rastrigin', 'rosenbrock', 'ackley', 'griewank'][i % 5];
                        switch(mixedType) {
                            case 'sphere':
                                surfaceFunc = stochasticGenerator.stochasticSphere(`mixed_sphere_${i}`);
                                break;
                            case 'rastrigin':
                                surfaceFunc = stochasticGenerator.stochasticRastrigin(`mixed_rastrigin_${i}`);
                                break;
                            case 'rosenbrock':
                                surfaceFunc = stochasticGenerator.stochasticRosenbrock(`mixed_rosenbrock_${i}`);
                                break;
                            case 'ackley':
                                surfaceFunc = stochasticGenerator.stochasticAckley(`mixed_ackley_${i}`);
                                break;
                            case 'griewank':
                                surfaceFunc = stochasticGenerator.stochasticGriewank(`mixed_griewank_${i}`);
                                break;
                        }
                        break;
                    default:
                        // Fallback to sphere
                        surfaceFunc = stochasticGenerator.stochasticSphere(`fallback_${i}`);
                        break;
                }

                const surfaceDisplayType = primaryType === 'mixed' ?
                    ['Sphere', 'Rastrigin', 'Rosenbrock', 'Ackley', 'Griewank'][i % 5] :
                    primaryType.charAt(0).toUpperCase() + primaryType.slice(1);

                challengeSurfaces.push({
                    name: `${surfaceDisplayType} #${i + 1}`,
                    func: surfaceFunc,
                    type: primaryType === 'mixed' ? ['sphere', 'rastrigin', 'rosenbrock', 'ackley', 'griewank'][i % 5] : primaryType
                });
            }

            console.log(`Generated ${challengeSurfaces.length} stochastic surfaces for fair competition`);

            // Set up currentContest object with challengeSurfaces
            currentContest = {
                challengeSurfaces: challengeSurfaces,
                interpretation: interpretation
            };

            const progressSection = document.getElementById('progressSection');
            progressSection.innerHTML = ''; // Clear previous results

            let currentSurface = 0;
            let contestRunning = true;

            async function runNextSurface() {
                if (!contestRunning || currentSurface >= challengeSurfaces.length) {
                    // Contest finished
                    console.log('Contest finished. Current surface:', currentSurface, 'Total surfaces:', challengeSurfaces.length);
                    optimizers.forEach(opt => opt.status = 'completed');
                    updateLeaderboard(true); // Force final update
                    contestRunning = false;
                    return;
                }

                console.log(`Starting surface ${currentSurface + 1}/${challengeSurfaces.length}`);

                const surface = challengeSurfaces[currentSurface];

                // Add visualization for 2D problems
                const visualizationHtml = interpretation.enable_visualization ?
                    `<div style="display: flex; gap: 20px; margin: 15px 0;">
                        <canvas id="surface-canvas-${currentSurface}" width="300" height="300"
                                style="border: 1px solid #ddd; border-radius: 5px; background: white;"></canvas>
                        <div style="flex: 1;">
                            <div style="font-size: 0.9em; color: #666; margin-bottom: 10px;">
                                🎯 2D Surface Visualization<br>
                                <span style="font-size: 0.8em;">Red = high values, Blue = low values<br>
                                Dots show algorithm paths</span>
                            </div>
                        </div>
                    </div>` : '';

                progressSection.innerHTML += `
                    <div class="surface-progress" id="surface-${currentSurface}">
                        <div class="surface-title" onclick="toggleSurface(${currentSurface})">
                            <span>Surface ${currentSurface + 1}: ${surface.name}</span>
                            <span class="collapse-indicator">▼</span>
                        </div>
                        <div class="surface-summary" id="surface-${currentSurface}-summary">
                            <!-- Summary will be populated when completed -->
                        </div>
                        <div class="surface-content" id="surface-${currentSurface}-content">
                            ${visualizationHtml}
                            <div id="surface-${currentSurface}-results"></div>
                        </div>
                    </div>
                `;

                // Reset optimizer status
                optimizers.forEach(opt => opt.status = 'waiting');

                // Run actual optimization
                setTimeout(async () => {
                    try {
                        await runSurfaceOptimization(currentSurface, surface, interpretation);
                        console.log(`Completed surface ${currentSurface + 1}, moving to next`);
                        currentSurface++;
                        setTimeout(runNextSurface, 1000);
                    } catch (error) {
                        console.error('Error in surface optimization:', error);
                        currentSurface++; // Skip failed surface
                        setTimeout(runNextSurface, 1000);
                    }
                }, 500);
            }

            runNextSurface();
        }

        async function runSurfaceOptimization(surfaceIndex, surface, interpretation) {
            const resultsDiv = document.getElementById(`surface-${surfaceIndex}-results`);
            const results = [];

            // Show loading state
            resultsDiv.innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    Running optimization algorithms...
                </div>
            `;

            // Draw initial surface visualization for 2D problems
            if (interpretation.enable_visualization) {
                const canvasId = `surface-canvas-${surfaceIndex}`;
                // Add small delay to ensure canvas is in DOM
                setTimeout(() => {
                    visualize2DSurface(canvasId, surface.func);
                }, 100);
            }

            const algorithmPaths = [];

            // Run optimizers with proper async handling to prevent freezing
            for (let i = 0; i < optimizers.length; i++) {
                const optimizer = optimizers[i];

                // Skip BayesianOpt for high dimensions (>5D) - it consistently hangs
                if ((optimizer.internalName || optimizer.name) === 'BayesianOpt' && interpretation.dimensions > 5) {
                    console.log(`Skipping ${optimizer.name} for ${interpretation.dimensions}D (too slow)`);
                    results.push({
                        name: optimizer.name,
                        score: Infinity,
                        evaluations: 0,
                        success: false,
                        skipped: true
                    });
                    optimizer.status = 'skipped';
                    optimizer.testsCompleted++;
                    continue;
                }

                // Update current optimizer status
                optimizer.status = 'running';

                // Show current progress with running indicator
                results.push({
                    name: optimizer.name,
                    score: null,
                    evaluations: 0,
                    success: false,
                    running: true
                });
                displaySurfaceResults(resultsDiv, results);
                results.pop(); // Remove the temporary running entry

                // Yield control to browser
                await new Promise(resolve => setTimeout(resolve, 50));

                try {
                    console.log(`Running ${optimizer.name} on surface ${surfaceIndex + 1}`);

                    // Create optimizer instance using internal name for factory
                    const opt = OptimizerFactory.create(
                        optimizer.internalName || optimizer.name,
                        surface.func,
                        interpretation.budget,
                        interpretation.dimensions
                    );

                    // Enable path tracking for 2D visualization
                    if (interpretation.enable_visualization) {
                        opt.trackPath = true;
                    }

                    // Dynamic timeout based on problem size and algorithm
                    const baseTimeout = Math.min(5000, Math.max(2000, interpretation.budget * 5)); // 2-5 seconds base
                    const algorithmTimeout = getAlgorithmTimeout(optimizer.internalName || optimizer.name, interpretation.dimensions, baseTimeout);

                    console.log(`Running ${optimizer.name} with ${algorithmTimeout}ms timeout`);

                    // Run optimization with timeout protection
                    const result = await runOptimizerWithTimeout(opt, optimizer.name, algorithmTimeout);

                    if (result) {
                        results.push({
                            name: optimizer.name,
                            score: result.bestValue,
                            evaluations: result.evaluations,
                            success: result.success
                        });

                        // Collect path for visualization
                        if (interpretation.enable_visualization && result.path && result.path.length > 0) {
                            algorithmPaths.push({
                                name: optimizer.name,
                                points: result.path
                            });

                            // Update visualization with current paths
                            const canvasId = `surface-canvas-${surfaceIndex}`;
                            try {
                                visualize2DSurface(canvasId, surface.func, algorithmPaths);
                            } catch (error) {
                                console.error('Error updating visualization:', error);
                                visualize2DSurface(canvasId, surface.func, []);
                            }
                        }

                        optimizer.status = 'completed';
                        optimizer.testsCompleted++;
                    } else {
                        // Timeout occurred
                        results.push({
                            name: optimizer.name,
                            score: Infinity,
                            evaluations: 0,
                            success: false
                        });
                        optimizer.status = 'timeout';
                        optimizer.testsCompleted++;
                    }

                } catch (error) {
                    console.error(`Error running ${optimizer.name}:`, error);
                    results.push({
                        name: optimizer.name,
                        score: Infinity,
                        evaluations: 0,
                        success: false
                    });
                    optimizer.status = 'failed';
                    optimizer.testsCompleted++;
                }

                // Show incremental results
                displaySurfaceResults(resultsDiv, results);

                // Longer delay between optimizers to prevent freezing
                await new Promise(resolve => setTimeout(resolve, 500));
            }

            // Sort results (lower is better)
            results.sort((a, b) => a.score - b.score);

            // Update Elo ratings based on results
            updateEloRatings(results);

            // Only update leaderboard after every few surfaces to reduce flicker
            if (surfaceIndex % 3 === 2 || surfaceIndex === 0) {
                updateLeaderboard(true);
            }

            // Final display
            displaySurfaceResults(resultsDiv, results);

            // Ensure final visualization is preserved for 2D problems
            if (interpretation.enable_visualization && algorithmPaths.length > 0) {
                const canvasId = `surface-canvas-${surfaceIndex}`;
                setTimeout(() => {
                    try {
                        visualize2DSurface(canvasId, surface.func, algorithmPaths);
                        console.log(`Final visualization preserved for surface ${surfaceIndex}`);
                    } catch (error) {
                        console.error('Error in final visualization:', error);
                    }
                }, 100);
            }

            // Mark optimizers as completed for this surface
            optimizers.forEach(opt => opt.status = 'completed');

            // Auto-collapse this surface with winner summary
            if (results.length > 0) {
                const winner = results[0];
                collapseSurface(surfaceIndex, winner.name.replace('_', ' '), winner.score);
            }

            console.log(`Surface ${surfaceIndex + 1} optimization completed`);
        }

        // Helper function to run optimizer with aggressive timeout protection
        async function runOptimizerWithTimeout(optimizer, name, timeoutMs) {
            return new Promise((resolve) => {
                let completed = false;
                let startTime = Date.now();

                console.log(`Starting ${name} with ${timeoutMs}ms timeout`);

                // Aggressive timeout - force completion
                const timeout = setTimeout(() => {
                    if (!completed) {
                        completed = true;
                        console.warn(`${name} FORCE TIMED OUT after ${timeoutMs}ms`);
                        resolve({
                            bestValue: Infinity,
                            bestX: Array(optimizer.nDim).fill(0.5),
                            evaluations: optimizer.evaluations || 0,
                            success: false,
                            timeout: true,
                            path: optimizer.trackPath ? optimizer.path : null
                        });
                    }
                }, timeoutMs);

                // Progress monitoring - check every 100ms
                const progressMonitor = setInterval(() => {
                    if (completed) {
                        clearInterval(progressMonitor);
                        return;
                    }

                    const elapsed = Date.now() - startTime;

                    // Force timeout if taking too long regardless of progress
                    if (elapsed > timeoutMs * 0.9) {
                        console.warn(`${name} approaching timeout at ${elapsed}ms`);
                    }

                    // Extra safety - hard abort after 120% of timeout
                    if (elapsed > timeoutMs * 1.2) {
                        if (!completed) {
                            completed = true;
                            clearTimeout(timeout);
                            clearInterval(progressMonitor);
                            console.error(`${name} EMERGENCY ABORT after ${elapsed}ms`);
                            resolve({
                                bestValue: Infinity,
                                bestX: Array(optimizer.nDim).fill(0.5),
                                evaluations: optimizer.evaluations || 0,
                                success: false,
                                timeout: true,
                                path: optimizer.trackPath ? optimizer.path : null
                            });
                        }
                    }
                }, 100);

                // Run optimizer with immediate timeout checking
                const runOptimizer = () => {
                    try {
                        if (completed) return;

                        console.log(`Actually starting ${name} optimization`);
                        const result = optimizer.optimize();

                        if (!completed) {
                            completed = true;
                            clearTimeout(timeout);
                            clearInterval(progressMonitor);

                            const elapsed = Date.now() - startTime;
                            console.log(`${name} completed in ${elapsed}ms`);
                            resolve(result);
                        }
                    } catch (error) {
                        if (!completed) {
                            completed = true;
                            clearTimeout(timeout);
                            clearInterval(progressMonitor);
                            console.error(`${name} failed:`, error);
                            resolve({
                                bestValue: Infinity,
                                bestX: Array(optimizer.nDim).fill(0.5),
                                evaluations: optimizer.evaluations || 0,
                                success: false,
                                error: error.message,
                                path: optimizer.trackPath ? optimizer.path : null
                            });
                        }
                    }
                };

                // Start immediately but allow UI to update first
                setTimeout(runOptimizer, 5);
            });
        }

        function displaySurfaceResults(resultsDiv, results) {
            // Sort results for display (lower score is better)
            const sortedResults = [...results].sort((a, b) => a.score - b.score);

            let html = '';
            sortedResults.forEach((result, index) => {
                let statusIcon = '❌';
                let scoreText = 'Failed';
                let className = 'optimizer-result';

                if (result.running) {
                    statusIcon = '🏃';
                    scoreText = 'Running...';
                    className = 'optimizer-result running';
                } else if (result.skipped) {
                    statusIcon = '⏭️';
                    scoreText = 'Skipped (>5D)';
                    className = 'optimizer-result skipped';
                } else if (result.success) {
                    statusIcon = isFinite(result.score) ? '✓' : '⚠️';
                    scoreText = isFinite(result.score) ? result.score.toFixed(6) : 'Failed';
                }

                html += `
                    <div class="${className}">
                        <span class="optimizer-name">
                            ${index + 1}.
                            ${result.name.replace('_', ' ')} ${statusIcon}
                        </span>
                        <span class="optimizer-score">
                            ${scoreText}
                            <small> (${result.evaluations} evals)</small>
                        </span>
                    </div>
                `;
            });

            resultsDiv.innerHTML = html;
        }

        function updateEloRatings(results) {
            // Conservative Elo update for stable long-term averages
            // Lower K-factor reduces recency bias and gives better averages over many challenges
            const baseK = 12; // Much lower for stability

            // Diminishing K-factor based on number of completed tests
            const avgTestsCompleted = optimizers.reduce((sum, opt) => sum + opt.testsCompleted, 0) / optimizers.length;
            const K = Math.max(8, baseK * Math.exp(-avgTestsCompleted / 20)); // Decreases as more data accumulated

            for (let i = 0; i < results.length; i++) {
                for (let j = i + 1; j < results.length; j++) {
                    const opt1 = optimizers.find(o => o.name === results[i].name);
                    const opt2 = optimizers.find(o => o.name === results[j].name);

                    if (opt1 && opt2) {
                        // results[i] beat results[j] (lower score)
                        const expected1 = 1 / (1 + Math.pow(10, (opt2.elo - opt1.elo) / 400));
                        const expected2 = 1 / (1 + Math.pow(10, (opt1.elo - opt2.elo) / 400));

                        opt1.elo += K * (1.0 - expected1);
                        opt2.elo += K * (0.0 - expected2);
                    }
                }
            }
        }

        function visualize2DSurface(canvasId, surfaceFunc, algorithmPaths = []) {
            const canvas = document.getElementById(canvasId);
            if (!canvas) {
                console.warn(`Canvas ${canvasId} not found`);
                return;
            }

            const ctx = canvas.getContext('2d');
            if (!ctx) {
                console.warn(`Could not get 2D context for canvas ${canvasId}`);
                return;
            }

            const width = canvas.width;
            const height = canvas.height;

            console.log(`Drawing surface on canvas ${canvasId} (${width}x${height})`);

            try {
                // Clear canvas
            ctx.clearRect(0, 0, width, height);
            console.log(`Cleared canvas ${canvasId} for redrawing`);

            // Generate surface data
            const resolution = 50;
            const surfaceData = [];
            let minVal = Infinity, maxVal = -Infinity;

            for (let i = 0; i < resolution; i++) {
                surfaceData[i] = [];
                for (let j = 0; j < resolution; j++) {
                    const x = i / (resolution - 1);
                    const y = j / (resolution - 1);
                    try {
                        const value = surfaceFunc([x, y]);
                        // Handle invalid values
                        if (!isFinite(value) || isNaN(value)) {
                            surfaceData[i][j] = 0;
                        } else {
                            surfaceData[i][j] = value;
                            if (isFinite(value)) {
                                minVal = Math.min(minVal, value);
                                maxVal = Math.max(maxVal, value);
                            }
                        }
                    } catch (error) {
                        console.warn('Surface evaluation error:', error);
                        surfaceData[i][j] = 0;
                    }
                }
            }

            // Ensure we have valid min/max values
            if (!isFinite(minVal) || !isFinite(maxVal) || minVal === maxVal) {
                console.warn('Invalid surface range, using defaults');
                minVal = 0;
                maxVal = 1;
            }

            // Draw surface as colored grid
            const cellWidth = width / resolution;
            const cellHeight = height / resolution;

            for (let i = 0; i < resolution; i++) {
                for (let j = 0; j < resolution; j++) {
                    const value = surfaceData[i][j];
                    let normalized = 0;

                    if (maxVal > minVal) {
                        normalized = Math.max(0, Math.min(1, (value - minVal) / (maxVal - minVal)));
                    }

                    // Color mapping: blue (low) to red (high)
                    const red = Math.floor(255 * normalized);
                    const blue = Math.floor(255 * (1 - normalized));
                    const green = Math.floor(128 * (1 - Math.abs(2 * normalized - 1)));

                    // Ensure valid RGB values
                    const safeRed = Math.max(0, Math.min(255, red));
                    const safeGreen = Math.max(0, Math.min(255, green));
                    const safeBlue = Math.max(0, Math.min(255, blue));

                    ctx.fillStyle = `rgb(${safeRed}, ${safeGreen}, ${safeBlue})`;
                    ctx.fillRect(i * cellWidth, j * cellHeight, cellWidth, cellHeight);
                }
            }

            // Draw algorithm paths
            const colors = ['#ff0000', '#00ff00', '#0000ff', '#ff8800', '#8800ff'];
            if (algorithmPaths && Array.isArray(algorithmPaths)) {
                console.log(`Drawing ${algorithmPaths.length} algorithm paths`);
                algorithmPaths.forEach((path, index) => {
                const color = colors[index % colors.length];
                ctx.strokeStyle = color;
                ctx.fillStyle = color;
                ctx.lineWidth = 2;

                if (path.points && path.points.length > 1) {
                    ctx.beginPath();
                    path.points.forEach((point, i) => {
                        const x = point[0] * width;
                        const y = point[1] * height;
                        if (i === 0) {
                            ctx.moveTo(x, y);
                        } else {
                            ctx.lineTo(x, y);
                        }
                    });
                    ctx.stroke();

                    // Mark start and end points
                    const start = path.points[0];
                    const end = path.points[path.points.length - 1];

                    // Start point (larger circle)
                    ctx.beginPath();
                    ctx.arc(start[0] * width, start[1] * height, 6, 0, 2 * Math.PI);
                    ctx.fill();

                    // End point (smaller circle with white center)
                    ctx.beginPath();
                    ctx.arc(end[0] * width, end[1] * height, 4, 0, 2 * Math.PI);
                    ctx.fill();
                    ctx.fillStyle = 'white';
                    ctx.beginPath();
                    ctx.arc(end[0] * width, end[1] * height, 2, 0, 2 * Math.PI);
                    ctx.fill();
                }
                });
            }

            // Add legend
            ctx.fillStyle = 'white';
            ctx.fillRect(5, 5, 120, 40);
            ctx.strokeStyle = 'black';
            ctx.strokeRect(5, 5, 120, 40);

            ctx.fillStyle = 'black';
            ctx.font = '10px Arial';
            ctx.fillText(`Min: ${minVal.toFixed(3)}`, 10, 20);
            ctx.fillText(`Max: ${maxVal.toFixed(3)}`, 10, 35);

            console.log(`Successfully drew surface on canvas ${canvasId}, range: ${minVal.toFixed(3)} to ${maxVal.toFixed(3)}`);

            } catch (error) {
                console.error(`Error visualizing surface on canvas ${canvasId}:`, error);
                // Draw error message on canvas
                ctx.fillStyle = '#ff6b6b';
                ctx.fillRect(0, 0, width, height);
                ctx.fillStyle = 'white';
                ctx.font = '14px Arial';
                ctx.fillText('Visualization Error', 10, height/2);
            }
        }