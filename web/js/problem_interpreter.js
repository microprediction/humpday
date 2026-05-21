/**
 * Sophisticated problem interpretation system
 * Maps natural language to optimization surface parameters using LLM + templates
 */

// Standard problem templates with detailed configurations
const ProblemTemplates = {
    // Smooth optimization problems
    smooth_optimization: {
        name: "Smooth Function Optimization",
        description: "Optimize smooth, unimodal landscapes",
        typical_dimensions: [2, 5, 10, 20],
        surface_config: {
            primary_surfaces: [
                {
                    type: 'sphere',
                    params: { center: 0.3, scale: 1.0 }
                },
                {
                    type: 'quadratic_with_correlation',
                    params: { correlation_strength: 0.2, noise_level: 0.01 }
                }
            ]
        },
        evaluation_budget_factor: 0.8, // Smooth problems need fewer evaluations
        keywords: ['smooth', 'simple', 'unimodal', 'quadratic', 'sphere']
    },

    multimodal_optimization: {
        name: "Multimodal Optimization",
        description: "Optimize landscapes with multiple local minima",
        typical_dimensions: [2, 5, 8, 15],
        surface_config: {
            primary_surfaces: [
                {
                    type: 'rastrigin',
                    params: { A: 8 }
                },
                {
                    type: 'ackley',
                    params: {}
                },
                {
                    type: 'griewank',
                    params: {}
                }
            ]
        },
        evaluation_budget_factor: 1.5, // Multimodal needs more evaluations
        keywords: ['multimodal', 'multiple', 'local', 'minima', 'rastrigin', 'ackley', 'peaks']
    },

    valley_optimization: {
        name: "Valley/Ridge Optimization",
        description: "Optimize curved valley landscapes",
        typical_dimensions: [2, 5, 10],
        surface_config: {
            primary_surfaces: [
                {
                    type: 'rosenbrock',
                    params: { a: 1.0, b: 100.0 }
                },
                {
                    type: 'constrained_rosenbrock',
                    params: { constraint_strength: 0.5 }
                }
            ]
        },
        evaluation_budget_factor: 1.2,
        keywords: ['valley', 'rosenbrock', 'banana', 'curved', 'ridge']
    },

    noisy_optimization: {
        name: "Noisy Optimization",
        description: "Optimize functions with measurement noise",
        typical_dimensions: [2, 5, 10, 15],
        surface_config: {
            primary_surfaces: [
                {
                    type: 'noisy_sphere',
                    params: { noise_level: 0.1 }
                },
                {
                    type: 'noisy_rosenbrock',
                    params: { noise_level: 0.05 }
                }
            ]
        },
        evaluation_budget_factor: 1.8, // Noisy problems need more evaluations
        keywords: ['noisy', 'stochastic', 'random', 'uncertain', 'measurement']
    },

    hyperparameter_tuning: {
        name: "ML Hyperparameter Optimization",
        description: "Tune machine learning model parameters",
        typical_dimensions: [5, 10, 15, 25],
        surface_config: {
            primary_surfaces: [
                {
                    type: 'validation_loss_landscape',
                    params: { local_minima_count: 3, smoothness: 0.7 }
                },
                {
                    type: 'noisy_rosenbrock',
                    params: { noise_level: 0.1, valley_width: 0.3 }
                }
            ]
        },
        evaluation_budget_factor: 1.5, // ML needs more evaluations
        keywords: ['neural', 'network', 'hyperparameter', 'learning', 'model', 'ml', 'ai', 'tune', 'parameters']
    },

    engineering_design: {
        name: "Engineering Design Optimization",
        description: "Optimize engineering parameters for performance",
        typical_dimensions: [8, 12, 20],
        surface_config: {
            primary_surfaces: [
                {
                    type: 'constrained_rosenbrock',
                    params: { constraint_strength: 0.8 }
                },
                {
                    type: 'multi_objective_pareto',
                    params: { objective_count: 2, conflict_level: 0.6 }
                }
            ]
        },
        evaluation_budget_factor: 1.0,
        keywords: ['design', 'engineering', 'antenna', 'engine', 'structure', 'material', 'optimize', 'parameters']
    },

    supply_chain: {
        name: "Supply Chain Optimization",
        description: "Optimize logistics and routing decisions",
        typical_dimensions: [15, 30, 50],
        surface_config: {
            primary_surfaces: [
                {
                    type: 'discrete_approximation',
                    params: { discrete_levels: 5, smoothing: 0.2 }
                },
                {
                    type: 'network_flow',
                    params: { node_count: 'auto', flow_constraints: true }
                }
            ]
        },
        evaluation_budget_factor: 1.3,
        keywords: ['supply', 'chain', 'logistics', 'routing', 'warehouse', 'distribution', 'inventory']
    }
};

// Enhanced problem interpreter with LLM integration
class SmartProblemInterpreter {
    constructor() {
        this.llm_engine = null;
        this.llm_available = false;
        this.templates = ProblemTemplates;
        // Don't auto-initialize LLM to avoid blocking - initialize on demand
    }

    async initializeLLM() {
        try {
            console.log('Initializing browser-based LLM...');

            // Show proper loading overlay with progress bar
            this.showLLMLoadingOverlay();

            // Initialize WebLLM with a small, fast model
            this.llm_engine = await window.CreateMLCEngine("Llama-3.2-1B-Instruct-q4f16_1-MLC", {
                initProgressCallback: (report) => {
                    console.log('LLM Loading:', report.text);
                    this.updateLLMProgress(report);
                }
            });

            this.llm_available = true;
            console.log('LLM initialized successfully');

            // Show success and hide overlay
            this.showLLMSuccess();

        } catch (error) {
            console.warn('Failed to initialize LLM, using template fallback:', error);
            this.llm_available = false;
            this.showLLMError(error);
        }
    }

    showLLMLoadingOverlay() {
        // Create overlay
        const overlay = document.createElement('div');
        overlay.id = 'llm-loading-overlay';
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
            font-family: Georgia, serif;
        `;

        // Create loading content
        const content = document.createElement('div');
        content.style.cssText = `
            background: white;
            padding: 40px;
            border-radius: 10px;
            text-align: center;
            max-width: 500px;
            margin: 20px;
        `;

        content.innerHTML = `
            <h2 style="margin-top: 0; color: #2c3e50;">Loading AI Language Model</h2>
            <p style="color: #666; margin-bottom: 30px;">
                Downloading and initializing the AI model for problem interpretation.<br>
                <strong>This will be much faster next time!</strong>
            </p>

            <div style="background: #f0f0f0; border-radius: 10px; padding: 3px; margin-bottom: 20px;">
                <div id="llm-progress-bar" style="
                    background: linear-gradient(90deg, #3498db, #2980b9);
                    height: 20px;
                    border-radius: 8px;
                    width: 0%;
                    transition: width 0.3s ease;
                "></div>
            </div>

            <div id="llm-status-text" style="color: #666; font-size: 14px;">
                Initializing...
            </div>

            <div style="margin-top: 20px; padding: 15px; background: #e8f4f8; border-radius: 5px; font-size: 13px; color: #2c3e50;">
                💡 <strong>First time loading:</strong> Downloads ~1GB model files<br>
                🚀 <strong>Subsequent uses:</strong> Loads instantly from cache
            </div>
        `;

        overlay.appendChild(content);
        document.body.appendChild(overlay);
    }

    updateLLMProgress(report) {
        const statusText = document.getElementById('llm-status-text');
        const progressBar = document.getElementById('llm-progress-bar');

        if (statusText) {
            statusText.textContent = report.text || 'Loading...';
        }

        if (progressBar && report.progress !== undefined) {
            const percentage = Math.round(report.progress * 100);
            progressBar.style.width = `${percentage}%`;

            // Update status with percentage if available
            if (statusText && percentage > 0) {
                statusText.textContent = `${report.text} (${percentage}%)`;
            }
        }
    }

    showLLMSuccess() {
        const overlay = document.getElementById('llm-loading-overlay');
        if (overlay) {
            const content = overlay.querySelector('div');
            content.innerHTML = `
                <h2 style="margin-top: 0; color: #27ae60;">✅ AI Model Ready!</h2>
                <p style="color: #666;">
                    Language model loaded successfully.<br>
                    You can now describe optimization problems in plain English.
                </p>
                <div style="margin-top: 20px;">
                    <button onclick="this.parentElement.parentElement.parentElement.remove()"
                            style="background: #27ae60; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-size: 16px;">
                        Continue
                    </button>
                </div>
            `;

            // Auto-close after 3 seconds
            setTimeout(() => {
                if (overlay.parentNode) {
                    overlay.remove();
                }
            }, 3000);
        }
    }

    showLLMError(error) {
        const overlay = document.getElementById('llm-loading-overlay');
        if (overlay) {
            const content = overlay.querySelector('div');
            content.innerHTML = `
                <h2 style="margin-top: 0; color: #e74c3c;">⚠️ AI Model Loading Failed</h2>
                <p style="color: #666;">
                    Could not load the AI model. Using template-based interpretation instead.<br>
                    <small style="color: #999;">Error: ${error.message}</small>
                </p>
                <div style="margin-top: 20px; padding: 15px; background: #fff3cd; border-radius: 5px;">
                    <strong>Don't worry!</strong> The system will still work using built-in problem templates.
                </div>
                <div style="margin-top: 20px;">
                    <button onclick="this.parentElement.parentElement.parentElement.remove()"
                            style="background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-size: 16px;">
                        Continue with Templates
                    </button>
                </div>
            `;
        }
    }

    async interpretProblem(description, use_llm = true) {
        // Try LLM interpretation first if available
        if (use_llm && this.llm_available && this.llm_engine) {
            try {
                return await this.llmInterpretation(description);
            } catch (error) {
                console.warn('LLM interpretation failed, falling back to templates:', error);
            }
        }

        // Fallback to template matching + rule-based interpretation
        return this.templateBasedInterpretation(description);
    }

    async llmInterpretation(description) {
        if (!this.llm_engine) {
            throw new Error('LLM not available');
        }

        const prompt = this.buildLLMPrompt(description);
        console.log('Querying LLM with prompt:', prompt);

        try {
            const response = await this.llm_engine.chat.completions.create({
                messages: [
                    {
                        role: "system",
                        content: "You are an optimization expert. Analyze problem descriptions and return JSON configurations for optimization benchmarks. Be precise and technical."
                    },
                    {
                        role: "user",
                        content: prompt
                    }
                ],
                temperature: 0.1,
                max_tokens: 512
            });

            const responseText = response.choices[0].message.content;
            console.log('LLM response:', responseText);

            // Parse JSON from response
            const jsonMatch = responseText.match(/\{[\s\S]*\}/);
            if (jsonMatch) {
                const parsed = JSON.parse(jsonMatch[0]);

                // Convert to our internal format
                const dimensions = parsed.dimensions || 10;
                return {
                    template_used: 'llm_generated',
                    dimensions: dimensions,
                    domain: 'general',
                    surface_type: 'custom_generated',
                    difficulty: dimensions <= 5 ? 'easy' : dimensions <= 15 ? 'medium' : 'hard',
                    budget: parsed.evaluation_budget || (dimensions <= 5 ? 60 : dimensions <= 15 ? 100 : 150),
                    surface_count: this.calculateSurfaceCount(dimensions),
                    surface_config: {
                        custom_surfaces: parsed.surfaces || [],
                        llm_generated: true
                    },
                    confidence: 0.9 // High confidence for LLM results
                };
            } else {
                throw new Error('Could not parse JSON from LLM response');
            }

        } catch (error) {
            console.error('LLM interpretation error:', error);
            throw error;
        }
    }

    mapLLMSurfaceType(characteristics) {
        if (!characteristics || !Array.isArray(characteristics)) return 'mixed';

        const charStr = characteristics.join(' ').toLowerCase();

        if (charStr.includes('smooth') || charStr.includes('unimodal')) return 'smooth';
        if (charStr.includes('multimodal') || charStr.includes('jagged') || charStr.includes('rough')) return 'multimodal';
        if (charStr.includes('valley') || charStr.includes('rosenbrock')) return 'valley';
        if (charStr.includes('noisy') || charStr.includes('stochastic')) return 'noisy';

        return 'mixed';
    }

    convertLLMSurfaceConfig(parsed) {
        const surfaceType = this.mapLLMSurfaceType(parsed.surface_characteristics);

        // Use our templates as base but customize based on LLM analysis
        const template = this.templates[`${surfaceType}_optimization`] || this.templates['multimodal_optimization'];

        let config = { ...template.surface_config };

        // Customize based on LLM suggestions
        if (parsed.surface_parameters && parsed.surface_parameters.length > 0) {
            config.primary_surfaces = parsed.surface_parameters.map(param => ({
                type: param.type,
                params: param.params || {}
            }));
        }

        return config;
    }

    buildLLMPrompt(description) {
        return `
You are an optimization surface generator. Create test surfaces based on this description:

Problem: "${description}"

Generate 5 different base optimization surface patterns that match the description (system will create variations). Return JSON:

{
    "dimensions": <number from description>,
    "evaluation_budget": <number from description or reasonable default>,
    "surfaces": [
        {
            "name": "<descriptive name>",
            "type": "custom_generated",
            "description": "<what this surface tests>",
            "generator_code": {
                "base_function": "<sphere|rosenbrock|rastrigin|ackley|griewank>",
                "modulations": [
                    {
                        "type": "<noise|scaling|frequency|offset|combination>",
                        "params": {<specific parameters>}
                    }
                ]
            }
        }
    ]
}

For "jagged" surfaces: use high-frequency modulations, multiple local minima
For "smooth" surfaces: use simple quadratic forms, low noise
For "valley" surfaces: use rosenbrock-like curved valleys
For "noisy" surfaces: add stochastic components

Extract dimensions and evaluation budget from numbers in the description.
Example: "12 dimensional" → 12 dimensions, "20 trials" → 20 evaluations
`;
    }

    templateBasedInterpretation(description) {
        const lower = description.toLowerCase();

        // Find best matching template
        let bestTemplate = null;
        let bestScore = 0;

        for (const [key, template] of Object.entries(this.templates)) {
            const score = this.calculateTemplateMatch(lower, template.keywords);
            if (score > bestScore) {
                bestScore = score;
                bestTemplate = { key, ...template };
            }
        }

        // Extract dimensions from text
        const dimensions = this.extractDimensions(description, bestTemplate);

        // Build configuration
        const config = {
            template_used: bestTemplate?.key || 'general',
            dimensions: dimensions,
            domain: this.inferDomain(lower, bestTemplate),
            surface_type: this.inferSurfaceType(lower, bestTemplate),
            difficulty: this.calculateDifficulty(dimensions, bestTemplate),
            budget: this.calculateBudget(dimensions, bestTemplate),
            surface_count: this.calculateSurfaceCount(dimensions),
            surface_config: bestTemplate?.surface_config || this.getDefaultSurfaceConfig(),
            confidence: bestScore
        };

        return config;
    }

    calculateSurfaceCount(dimensions) {
        // Reduced surface counts to prevent browser freezing
        // Balance between statistical significance and performance
        if (dimensions <= 2) return 5; // Good for visualization
        if (dimensions <= 5) return 4;
        if (dimensions <= 10) return 3;
        return 2; // High-dimensional problems
    }

    calculateTemplateMatch(description, keywords) {
        let matches = 0;
        let totalWeight = 0;

        keywords.forEach(keyword => {
            totalWeight += 1;
            if (description.includes(keyword)) {
                matches += 1;
                // Bonus for exact word boundaries
                if (new RegExp(`\\b${keyword}\\b`).test(description)) {
                    matches += 0.5;
                }
            }
        });

        return totalWeight > 0 ? matches / totalWeight : 0;
    }

    extractDimensions(description, template) {
        // Look for explicit numbers
        const dimPatterns = [
            /(\d+)\s*(?:dimension|parameter|variable|feature|asset|stock|factor)/i,
            /(?:with|have|using)\s*(\d+)\s*(?:param|var|dim)/i,
            /(\d+)d\s/i
        ];

        for (const pattern of dimPatterns) {
            const match = description.match(pattern);
            if (match) {
                return parseInt(match[1]);
            }
        }

        // Use template suggestions
        if (template?.typical_dimensions) {
            // Choose middle value as default
            const dims = template.typical_dimensions;
            return dims[Math.floor(dims.length / 2)];
        }

        return 10; // Reasonable default
    }

    inferDomain(description, template) {
        if (template?.key.includes('portfolio') || description.includes('financ')) return 'finance';
        if (template?.key.includes('hyperparameter') || description.includes('ml')) return 'machine_learning';
        if (template?.key.includes('engineering') || description.includes('design')) return 'engineering';
        if (template?.key.includes('supply') || description.includes('logistic')) return 'logistics';
        return 'general';
    }

    inferSurfaceType(description, template) {
        if (description.includes('smooth') || description.includes('simple')) return 'smooth';
        if (description.includes('noisy') || description.includes('stochastic')) return 'noisy';
        if (description.includes('multimodal') || description.includes('multiple')) return 'multimodal';
        if (description.includes('valley') || description.includes('rosenbrock')) return 'valley';

        // Use template default
        if (template?.surface_config?.primary_surfaces) {
            return 'template_based';
        }

        return 'mixed';
    }

    calculateDifficulty(dimensions, template) {
        if (dimensions <= 5) return 'easy';
        if (dimensions <= 15) return 'medium';
        return 'hard';
    }

    calculateBudget(dimensions, template) {
        const baseBudget = dimensions <= 5 ? 60 : dimensions <= 15 ? 100 : 150;
        const factor = template?.evaluation_budget_factor || 1.0;
        return Math.round(baseBudget * factor);
    }

    getDefaultSurfaceConfig() {
        return {
            primary_surfaces: [
                { type: 'sphere', params: {} },
                { type: 'rosenbrock', params: {} },
                { type: 'rastrigin', params: { A: 5 } }
            ]
        };
    }

    // Generate example problems for UI
    getExampleProblems() {
        return [
            {
                text: "Find minimum of smooth 2D function",
                expected: "Smooth domain, 2D, sphere-like surfaces with visualization"
            },
            {
                text: "Optimize Rosenbrock valley in 4 dimensions",
                expected: "Valley domain, 4D, curved valley landscapes"
            },
            {
                text: "Multimodal optimization with 6 variables and local minima",
                expected: "Multimodal domain, 6D, Rastrigin/Ackley surfaces"
            },
            {
                text: "Noisy optimization problem with 8 parameters",
                expected: "Noisy domain, 8D, stochastic surfaces"
            },
            {
                text: "High-dimensional sphere function with 25 variables",
                expected: "Smooth domain, 25D, high-dim sphere optimization"
            },
            {
                text: "Complex landscape with multiple peaks and valleys",
                expected: "Mixed domain, auto-detect dimensions, varied surfaces"
            }
        ];
    }
}

// Enhanced surface generator that uses template configurations
class EnhancedSurfaceGenerator {
    generateFromConfig(config) {
        if (config.surface_config?.llm_generated && config.surface_config?.custom_surfaces) {
            return this.generateCustomSurfaces(config);
        } else if (config.surface_config?.primary_surfaces) {
            return this.generateFromTemplate(config);
        } else {
            // Fallback to basic surface generation
            return this.generateBasicSuite(config);
        }
    }

    generateCustomSurfaces(config) {
        const surfaces = [];
        const customSurfaces = config.surface_config.custom_surfaces;

        for (const surfaceSpec of customSurfaces) {
            try {
                const surface = this.buildCustomSurface(surfaceSpec, config.dimensions);
                surfaces.push({
                    name: surfaceSpec.name || `custom_${surfaces.length}`,
                    func: surface,
                    spec: surfaceSpec
                });
            } catch (error) {
                console.error('Failed to build custom surface:', error);
                // Fallback to a basic surface
                surfaces.push({
                    name: `fallback_${surfaces.length}`,
                    func: TestSurfaces.sphere(),
                    spec: surfaceSpec
                });
            }
        }

        return surfaces.length > 0 ? surfaces : this.generateBasicSuite(config);
    }

    buildCustomSurface(surfaceSpec, dimensions) {
        const generator = surfaceSpec.generator_code;
        if (!generator) {
            return TestSurfaces.sphere();
        }

        // Get base function
        const baseFunc = TestSurfaces[generator.base_function] || TestSurfaces.sphere;
        let baseSurface = baseFunc();

        // Apply modulations
        if (generator.modulations && generator.modulations.length > 0) {
            return this.applyModulations(baseSurface, generator.modulations, dimensions);
        }

        return baseSurface;
    }

    applyModulations(baseFunc, modulations, dimensions) {
        return function(x) {
            let result = baseFunc(x);

            for (const mod of modulations) {
                switch (mod.type) {
                    case 'noise':
                        const noiseLevel = mod.params.level || 0.1;
                        result += noiseLevel * (Math.random() - 0.5) * 2;
                        break;

                    case 'scaling':
                        const scale = mod.params.factor || 1.0;
                        result *= scale;
                        break;

                    case 'frequency':
                        const freq = mod.params.frequency || 5.0;
                        const amplitude = mod.params.amplitude || 0.2;
                        let oscillation = 0;
                        for (let i = 0; i < x.length; i++) {
                            oscillation += amplitude * Math.sin(freq * Math.PI * x[i]);
                        }
                        result += oscillation;
                        break;

                    case 'offset':
                        const center = mod.params.center || 0.5;
                        let penalty = 0;
                        for (let i = 0; i < x.length; i++) {
                            penalty += Math.pow(x[i] - center, 2);
                        }
                        result += mod.params.strength || 1.0 * penalty;
                        break;

                    case 'combination':
                        // Add multiple peaks/valleys
                        const peaks = mod.params.peak_count || 3;
                        const strength = mod.params.strength || 0.5;
                        for (let p = 0; p < peaks; p++) {
                            const center = 0.2 + p * (0.6 / peaks);
                            let dist = 0;
                            for (let i = 0; i < x.length; i++) {
                                dist += Math.pow(x[i] - center, 2);
                            }
                            result += strength * Math.exp(-10 * dist) * Math.sin(20 * dist);
                        }
                        break;
                }
            }

            return result;
        };
    }

    generateFromTemplate(config) {
        const surfaces = [];
        const targetCount = config.surface_count || 10;
        const baseSurfaces = config.surface_config.primary_surfaces;

        // Generate multiple variations of each base surface type
        for (let i = 0; i < targetCount; i++) {
            const baseIndex = i % baseSurfaces.length;
            const surfaceSpec = baseSurfaces[baseIndex];

            // Create variations by modifying parameters
            const variation = this.createSurfaceVariation(surfaceSpec, i, config.dimensions);
            surfaces.push({
                name: `${config.template_used}_${surfaceSpec.type}_${i}`,
                func: variation,
                spec: surfaceSpec
            });
        }

        return surfaces;
    }

    createSurfaceVariation(baseSpec, variationIndex, dimensions) {
        const spec = { ...baseSpec };

        // Add systematic variations
        const variation = variationIndex % 4;

        switch (baseSpec.type) {
            case 'sphere':
                spec.params = {
                    ...spec.params,
                    center: 0.2 + (variation * 0.2),
                    scale: 0.8 + (variation * 0.3)
                };
                break;

            case 'rosenbrock':
                spec.params = {
                    ...spec.params,
                    a: 0.5 + (variation * 0.5),
                    b: 80 + (variation * 40)
                };
                break;

            case 'rastrigin':
                spec.params = {
                    ...spec.params,
                    A: 5 + (variation * 3)
                };
                break;

            case 'ackley':
                spec.params = {
                    ...spec.params,
                    a: 18 + (variation * 4),
                    b: 0.15 + (variation * 0.1)
                };
                break;
        }

        return this.createSurfaceFromSpec(spec, dimensions);
    }

    createSurfaceFromSpec(spec, dimensions) {
        switch (spec.type) {
            case 'quadratic_with_correlation':
                return this.createCorrelatedQuadratic(spec.params, dimensions);
            case 'validation_loss_landscape':
                return this.createMLLossLandscape(spec.params, dimensions);
            case 'constrained_rosenbrock':
                return this.createConstrainedRosenbrock(spec.params, dimensions);
            // Add more specialized surface types...
            default:
                return TestSurfaces[spec.type] ?
                    TestSurfaces[spec.type](spec.params) :
                    TestSurfaces.sphere();
        }
    }

    createCorrelatedQuadratic(params, dimensions) {
        const correlation = params.correlation_strength || 0.3;
        const noise = params.noise_level || 0.05;

        return function(x) {
            let sum = 0;
            // Quadratic with correlation between variables
            for (let i = 0; i < x.length; i++) {
                sum += Math.pow(x[i] - 0.5, 2);
                if (i < x.length - 1) {
                    sum += correlation * (x[i] - 0.5) * (x[i + 1] - 0.5);
                }
            }
            // Add noise
            sum += noise * (Math.random() - 0.5);
            return sum;
        };
    }

    createMLLossLandscape(params, dimensions) {
        const localMinimaCount = params.local_minima_count || 3;
        const smoothness = params.smoothness || 0.7;

        return function(x) {
            // Simulate ML loss landscape with local minima
            let loss = 0;

            // Global structure (validation loss)
            for (let i = 0; i < x.length; i++) {
                loss += Math.pow(x[i] - 0.6, 2); // Optimal around 0.6
            }

            // Local minima (overfitting regions)
            for (let m = 0; m < localMinimaCount; m++) {
                const center = 0.2 + m * 0.3;
                let localLoss = 0;
                for (let i = 0; i < x.length; i++) {
                    localLoss += Math.pow(x[i] - center, 2);
                }
                loss += 0.5 * Math.exp(-5 * localLoss);
            }

            return loss * smoothness;
        };
    }

    createConstrainedRosenbrock(params, dimensions) {
        const constraintStrength = params.constraint_strength || 0.8;

        return function(x) {
            // Standard Rosenbrock
            let sum = 0;
            for (let i = 0; i < x.length - 1; i++) {
                sum += 100 * Math.pow(x[i + 1] - x[i] * x[i], 2) + Math.pow(1 - x[i], 2);
            }

            // Add constraint penalty (engineering constraints)
            let penalty = 0;
            const constraintValue = x.reduce((a, b) => a + b, 0) - x.length * 0.5;
            if (Math.abs(constraintValue) > 0.1) {
                penalty = constraintStrength * Math.pow(constraintValue, 2);
            }

            return sum + penalty;
        };
    }

    generateBasicSuite(config) {
        const targetCount = config.surface_count || 10;

        // Use sophisticated stochastic surface generation
        const stochasticGen = new StochasticSurfaceGenerator(Date.now());
        const suite = stochasticGen.getRandomFunctionSuite(targetCount);

        console.log('🎲 Generated stochastic surface suite:', stochasticGen.getBenchmarkMetadata());

        return suite.map(item => ({
            name: item.name,
            func: item.func,
            spec: { type: item.type, stochastic: true }
        }));
    }

    generateBasicSuite_OLD(config) {
        const surfaces = [];
        const targetCount = config.surface_count || 10;

        // Generate truly diverse algorithmic surfaces (backup method)
        const surfaceGenerators = [
            () => this.createHybridSurface(['sphere', 'rastrigin'], config.dimensions),
            () => this.createCompositionSurface(['rosenbrock', 'ackley'], config.dimensions),
            () => this.createNoisySurface('griewank', config.dimensions),
            () => this.createScaledSurface('styblinski', config.dimensions),
            () => this.createMultiModalSurface(config.dimensions),
            () => this.createVallySurface(config.dimensions),
            () => this.createRoughSurface(config.dimensions),
            () => this.createSmoothSurface(config.dimensions),
            () => this.createAsymmetricSurface(config.dimensions),
            () => this.createRotatedSurface('rosenbrock', config.dimensions)
        ];

        for (let i = 0; i < targetCount; i++) {
            const generator = surfaceGenerators[i % surfaceGenerators.length];
            const surface = generator();

            surfaces.push({
                name: surface.name,
                func: surface.func,
                spec: surface.spec
            });
        }

        return surfaces;
    }

    createHybridSurface(baseTypes, dimensions) {
        const [type1, type2] = baseTypes;
        const weight = 0.3 + Math.random() * 0.4; // Random mix

        return {
            name: `hybrid_${type1}_${type2}_${Math.floor(weight*100)}`,
            func: function(x) {
                const f1 = TestSurfaces[type1]()(x);
                const f2 = TestSurfaces[type2]()(x);
                return weight * f1 + (1 - weight) * f2;
            },
            spec: { type: 'hybrid', components: [type1, type2], weight }
        };
    }

    createCompositionSurface(baseTypes, dimensions) {
        const [type1, type2] = baseTypes;
        const offset = Math.random() * 0.5 + 0.2;

        return {
            name: `composition_${type1}_${type2}_${Math.floor(offset*100)}`,
            func: function(x) {
                const shifted_x = x.map(xi => (xi + offset) % 1.0);
                const f1 = TestSurfaces[type1]()(x);
                const f2 = TestSurfaces[type2]()(shifted_x);
                return f1 * Math.exp(-f2 * 0.1);
            },
            spec: { type: 'composition', components: [type1, type2], offset }
        };
    }

    createMultiModalSurface(dimensions) {
        const peakCount = 3 + Math.floor(Math.random() * 4);
        const peaks = [];
        for (let i = 0; i < peakCount; i++) {
            peaks.push({
                center: Array(dimensions).fill(0).map(() => Math.random()),
                strength: 0.5 + Math.random() * 2.0,
                width: 0.1 + Math.random() * 0.3
            });
        }

        return {
            name: `multimodal_${peakCount}_peaks`,
            func: function(x) {
                let sum = 0;
                peaks.forEach(peak => {
                    let dist = 0;
                    for (let i = 0; i < x.length; i++) {
                        dist += Math.pow(x[i] - peak.center[i], 2);
                    }
                    sum += peak.strength * Math.exp(-dist / peak.width);
                });
                return sum + 0.1 * Math.sin(20 * x.reduce((a,b) => a+b, 0));
            },
            spec: { type: 'multimodal', peaks }
        };
    }

    createVallySurface(dimensions) {
        const direction = Array(dimensions).fill(0).map(() => Math.random() - 0.5);
        const norm = Math.sqrt(direction.reduce((sum, d) => sum + d*d, 0));
        const normalizedDir = direction.map(d => d / norm);

        return {
            name: `valley_oriented_${Math.floor(norm*100)}`,
            func: function(x) {
                // Project onto valley direction
                const projection = x.reduce((sum, xi, i) => sum + xi * normalizedDir[i], 0);
                const perpendicular = x.reduce((sum, xi, i) =>
                    sum + Math.pow(xi - projection * normalizedDir[i], 2), 0);
                return Math.pow(projection - 0.5, 2) + 10 * perpendicular;
            },
            spec: { type: 'valley', direction: normalizedDir }
        };
    }

    createRoughSurface(dimensions) {
        const frequency = 5 + Math.random() * 15;
        const amplitude = 0.5 + Math.random() * 1.0;

        return {
            name: `rough_freq${Math.floor(frequency)}_amp${Math.floor(amplitude*100)}`,
            func: function(x) {
                let base = TestSurfaces.sphere()(x);
                let roughness = 0;
                for (let i = 0; i < x.length; i++) {
                    roughness += amplitude * Math.sin(frequency * Math.PI * x[i]) *
                                 Math.cos(frequency * 1.3 * Math.PI * x[(i+1) % x.length]);
                }
                return base + roughness;
            },
            spec: { type: 'rough', frequency, amplitude }
        };
    }

    createSmoothSurface(dimensions) {
        const curvature = 1 + Math.random() * 3;
        const center = Array(dimensions).fill(0).map(() => 0.3 + Math.random() * 0.4);

        return {
            name: `smooth_curved_${Math.floor(curvature*100)}`,
            func: function(x) {
                return x.reduce((sum, xi, i) =>
                    sum + curvature * Math.pow(xi - center[i], 2), 0);
            },
            spec: { type: 'smooth', curvature, center }
        };
    }

    createAsymmetricSurface(dimensions) {
        const asymmetry = 2 + Math.random() * 4;

        return {
            name: `asymmetric_${Math.floor(asymmetry*100)}`,
            func: function(x) {
                return x.reduce((sum, xi, i) => {
                    const shift = (i + 1) * 0.1;
                    return sum + Math.pow(Math.abs(xi - 0.5 - shift), asymmetry);
                }, 0);
            },
            spec: { type: 'asymmetric', asymmetry }
        };
    }

    createRotatedSurface(baseType, dimensions) {
        const angle = Math.random() * Math.PI;

        return {
            name: `rotated_${baseType}_${Math.floor(angle*100)}`,
            func: function(x) {
                if (x.length < 2) return TestSurfaces[baseType]()(x);

                // Simple 2D rotation for first two dimensions
                const cos_a = Math.cos(angle);
                const sin_a = Math.sin(angle);
                const rotated = [...x];
                rotated[0] = cos_a * x[0] - sin_a * x[1];
                rotated[1] = sin_a * x[0] + cos_a * x[1];

                // Clip to [0,1] bounds
                for (let i = 0; i < rotated.length; i++) {
                    rotated[i] = Math.max(0, Math.min(1, rotated[i]));
                }

                return TestSurfaces[baseType]()(rotated);
            },
            spec: { type: 'rotated', baseType, angle }
        };
    }

    generateRandomParams(surfaceType, seed) {
        // Use seed for reproducible variations
        const random = () => ((seed * 9301 + 49297) % 233280) / 233280.0;

        switch (surfaceType) {
            case 'sphere':
                return {
                    center: 0.2 + random() * 0.6,
                    scale: 0.5 + random() * 1.0
                };

            case 'rosenbrock':
                return {
                    a: 0.5 + random() * 1.0,
                    b: 50 + random() * 100
                };

            case 'rastrigin':
                return {
                    A: 5 + random() * 10
                };

            case 'ackley':
                return {
                    a: 15 + random() * 10,
                    b: 0.1 + random() * 0.2
                };

            case 'griewank':
            default:
                return {};
        }
    }
}

// Export enhanced interpreter
window.SmartProblemInterpreter = SmartProblemInterpreter;
window.EnhancedSurfaceGenerator = EnhancedSurfaceGenerator;