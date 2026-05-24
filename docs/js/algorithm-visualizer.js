/**
 * 3D Algorithm Visualization Tool
 * Shows optimization algorithms navigating 3D surfaces in real-time
 */

class AlgorithmVisualizer {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        this.onReady = options.onReady || (() => {});
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.surface = null;
        this.optimizerPath = [];
        this.currentPosition = { x: 0, y: 0, z: 0 };
        this.isRunning = false;
        this.animationId = null;

        // Comprehensive test functions from optimization literature
        this.functions = {
            // Unimodal functions
            sphere: {
                name: '2D Sphere (Unimodal)',
                func: (x, y) => x*x + y*y,
                range: { min: -2, max: 2 },
                optimum: { x: 0, y: 0, z: 0 },
                category: 'Unimodal'
            },
            rosenbrock: {
                name: '2D Rosenbrock (Valley)',
                func: (x, y) => Math.pow(1 - x, 2) + 100 * Math.pow(y - x*x, 2),
                range: { min: -2, max: 2 },
                optimum: { x: 1, y: 1, z: 0 },
                category: 'Valley'
            },

            // Multimodal functions
            ackley: {
                name: '2D Ackley (Multimodal)',
                func: (x, y) => {
                    const a = 20;
                    const b = 0.2;
                    const c = 2 * Math.PI;
                    return -a * Math.exp(-b * Math.sqrt(0.5 * (x*x + y*y))) -
                           Math.exp(0.5 * (Math.cos(c*x) + Math.cos(c*y))) + Math.E + a;
                },
                range: { min: -5, max: 5 },
                optimum: { x: 0, y: 0, z: 0 },
                category: 'Multimodal'
            },
            rastrigin: {
                name: '2D Rastrigin (Highly Multimodal)',
                func: (x, y) => {
                    const A = 10;
                    const n = 2;
                    return A * n + (x*x - A * Math.cos(2 * Math.PI * x)) + (y*y - A * Math.cos(2 * Math.PI * y));
                },
                range: { min: -3, max: 3 },
                optimum: { x: 0, y: 0, z: 0 },
                category: 'Multimodal'
            },
            griewank: {
                name: '2D Griewank (Multimodal)',
                func: (x, y) => {
                    return 1 + (x*x + y*y) / 4000 - Math.cos(x) * Math.cos(y / Math.sqrt(2));
                },
                range: { min: -3, max: 3 },
                optimum: { x: 0, y: 0, z: 0 },
                category: 'Multimodal'
            },
            schwefel: {
                name: '2D Schwefel (Deceptive)',
                func: (x, y) => {
                    const term1 = x * Math.sin(Math.sqrt(Math.abs(x)));
                    const term2 = y * Math.sin(Math.sqrt(Math.abs(y)));
                    return 418.9829 * 2 - (term1 + term2);
                },
                range: { min: -250, max: 250 },
                optimum: { x: 420.9687, y: 420.9687, z: 0 },
                category: 'Deceptive'
            },

            // Special functions
            beale: {
                name: '2D Beale (Steep Ridges)',
                func: (x, y) => {
                    const a = 1.5 - x + x*y;
                    const b = 2.25 - x + x*y*y;
                    const c = 2.625 - x + x*y*y*y;
                    return a*a + b*b + c*c;
                },
                range: { min: -3, max: 3 },
                optimum: { x: 3, y: 0.5, z: 0 },
                category: 'Ridged'
            },
            himmelblau: {
                name: '2D Himmelblau (4 Global Optima)',
                func: (x, y) => {
                    return Math.pow(x*x + y - 11, 2) + Math.pow(x + y*y - 7, 2);
                },
                range: { min: -4, max: 4 },
                optimum: { x: 3, y: 2, z: 0 }, // One of four optima
                category: 'Multiple Optima'
            },
            booth: {
                name: '2D Booth (Simple Bowl)',
                func: (x, y) => {
                    return Math.pow(x + 2*y - 7, 2) + Math.pow(2*x + y - 5, 2);
                },
                range: { min: -5, max: 5 },
                optimum: { x: 1, y: 3, z: 0 },
                category: 'Simple'
            },
            matyas: {
                name: '2D Matyas (Saddle-shaped)',
                func: (x, y) => {
                    return 0.26 * (x*x + y*y) - 0.48 * x * y;
                },
                range: { min: -5, max: 5 },
                optimum: { x: 0, y: 0, z: 0 },
                category: 'Saddle'
            },
            levy: {
                name: '2D Lévy (Multimodal Waves)',
                func: (x, y) => {
                    const w1 = 1 + (x - 1) / 4;
                    const w2 = 1 + (y - 1) / 4;
                    const term1 = Math.pow(Math.sin(Math.PI * w1), 2);
                    const term2 = Math.pow(w1 - 1, 2) * (1 + 10 * Math.pow(Math.sin(Math.PI * w1 + 1), 2));
                    const term3 = Math.pow(w2 - 1, 2) * (1 + Math.pow(Math.sin(2 * Math.PI * w2), 2));
                    return term1 + term2 + term3;
                },
                range: { min: -5, max: 5 },
                optimum: { x: 1, y: 1, z: 0 },
                category: 'Waves'
            },
            easom: {
                name: '2D Easom (Needle in Haystack)',
                func: (x, y) => {
                    return -Math.cos(x) * Math.cos(y) * Math.exp(-(Math.pow(x - Math.PI, 2) + Math.pow(y - Math.PI, 2)));
                },
                range: { min: -5, max: 5 },
                optimum: { x: Math.PI, y: Math.PI, z: -1 },
                category: 'Needle'
            }
        };

        this.currentFunction = 'sphere';
        this.resolution = 50; // Grid resolution for surface
        this.viewMode = '3d'; // '3d' or 'wireframe'
        this.animationSpeed = 'medium'; // 'slow', 'medium', 'fast'

        this.init();
    }

    getScaledZ(z) {
        // Smart scaling based on function characteristics
        const funcName = this.currentFunction;

        if (['rosenbrock', 'beale', 'levy', 'booth', 'himmelblau'].includes(funcName)) {
            // Functions that can get very large - use log scaling
            return Math.log(Math.abs(z) + 1) * 0.15 * Math.sign(z);
        } else if (['schwefel'].includes(funcName)) {
            // Very large range functions - heavy scaling
            return z * 0.001;
        } else if (['ackley', 'griewank', 'rastrigin'].includes(funcName)) {
            // Moderate range multimodal functions
            return z * 0.08;
        } else if (['easom'].includes(funcName)) {
            // Functions with negative values
            return (z + 2) * 0.3; // Shift and scale
        } else if (['matyas'].includes(funcName)) {
            // Small range functions
            return z * 2.0;
        } else {
            // Default scaling for sphere and similar
            return z * 0.3;
        }
    }

    updateViewMode(mode) {
        this.viewMode = mode;
        if (this.surface) {
            if (mode === 'wireframe') {
                this.surface.material.wireframe = true;
            } else {
                this.surface.material.wireframe = false;
            }
        }
    }

    init() {
        // Create Three.js scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xfafafa);

        // Camera
        this.camera = new THREE.PerspectiveCamera(
            60,
            this.container.clientWidth / this.container.clientHeight,
            0.1,
            1000
        );
        // Set camera to match the ideal side-view orientation for optimization surfaces
        this.camera.position.set(10, 4, 2);
        this.camera.lookAt(0, 0, 0);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.container.appendChild(this.renderer.domElement);

        // Controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        // Set ideal viewing constraints for optimization surfaces
        this.controls.minPolarAngle = Math.PI * 0.1; // Prevent looking too far up
        this.controls.maxPolarAngle = Math.PI * 0.8; // Prevent looking too far down
        this.controls.minDistance = 4;
        this.controls.maxDistance = 20;

        // Lighting
        this.setupLighting();

        // Create surface
        this.createSurface();

        // Create UI controls
        this.createUI();

        // Start render loop
        this.animate();

        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());

        // Call ready callback after everything is initialized
        setTimeout(() => this.onReady(), 500);
    }

    setupLighting() {
        // Ambient light - brighter for better visibility
        const ambientLight = new THREE.AmbientLight(0x404040, 0.8);
        this.scene.add(ambientLight);

        // Main directional light
        const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
        dirLight.position.set(8, 8, 6);
        dirLight.castShadow = true;
        dirLight.shadow.mapSize.width = 2048;
        dirLight.shadow.mapSize.height = 2048;
        dirLight.shadow.camera.near = 0.5;
        dirLight.shadow.camera.far = 50;
        dirLight.shadow.camera.left = -10;
        dirLight.shadow.camera.right = 10;
        dirLight.shadow.camera.top = 10;
        dirLight.shadow.camera.bottom = -10;
        this.scene.add(dirLight);

        // Fill light from opposite side
        const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
        fillLight.position.set(-5, -5, 3);
        this.scene.add(fillLight);
    }

    createSurface() {
        if (this.surface) {
            this.scene.remove(this.surface);
        }

        const func = this.functions[this.currentFunction];
        const range = func.range;

        // Normalize all surfaces to the same visual size (8x8 units)
        const visualSize = 8;
        const geometry = new THREE.PlaneGeometry(visualSize, visualSize, this.resolution - 1, this.resolution - 1);

        const vertices = geometry.attributes.position.array;
        const colors = [];

        let minZ = Infinity, maxZ = -Infinity;
        const zValues = [];

        // First pass: calculate all Z values and find range
        // Map visual coordinates (-4 to +4) to function range
        for (let i = 0; i < vertices.length; i += 3) {
            const visualX = vertices[i];
            const visualY = vertices[i + 1];

            // Map from visual range [-4, +4] to function range
            const mathX = (visualX / visualSize) * (range.max - range.min) + (range.max + range.min) / 2;
            const mathY = (visualY / visualSize) * (range.max - range.min) + (range.max + range.min) / 2;

            const z = func.func(mathX, mathY);
            zValues.push(z);

            if (z < minZ) minZ = z;
            if (z > maxZ) maxZ = z;
        }

        // Second pass: normalize Z values and apply to geometry
        for (let i = 0, zIndex = 0; i < vertices.length; i += 3, zIndex++) {
            const z = zValues[zIndex];

            const scaledZ = this.getScaledZ(z);

            vertices[i + 2] = scaledZ;

            // Create smooth color gradient
            const normalizedZ = (z - minZ) / (maxZ - minZ);
            const color = new THREE.Color();

            // Better color scheme: blue (low) -> green -> yellow -> red (high)
            if (normalizedZ < 0.33) {
                color.setHSL(0.67 - normalizedZ * 0.67, 0.9, 0.5); // Blue to cyan
            } else if (normalizedZ < 0.67) {
                color.setHSL(0.33 - (normalizedZ - 0.33) * 0.33, 0.9, 0.5); // Cyan to green to yellow
            } else {
                color.setHSL(0.11 - (normalizedZ - 0.67) * 0.11, 0.9, 0.5); // Yellow to red
            }

            colors.push(color.r, color.g, color.b);
        }

        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        geometry.computeVertexNormals(); // Important for proper lighting!

        const material = new THREE.MeshLambertMaterial({
            vertexColors: true,
            side: THREE.DoubleSide,
            wireframe: false
        });

        this.surface = new THREE.Mesh(geometry, material);
        this.surface.receiveShadow = true;
        this.surface.castShadow = true;

        // Make the surface bigger by default for better visibility
        this.surface.scale.setScalar(1.5);

        this.scene.add(this.surface);

        // Add optimum marker
        this.addOptimumMarker();
    }

    addOptimumMarker() {
        // Remove existing optimum marker
        const existingMarker = this.scene.getObjectByName('optimumMarker');
        if (existingMarker) {
            this.scene.remove(existingMarker);
        }

        const func = this.functions[this.currentFunction];
        const optimum = func.optimum;
        const range = func.range;

        const geometry = new THREE.SphereGeometry(0.15, 16, 16);
        const material = new THREE.MeshBasicMaterial({
            color: 0x00ff00,
            transparent: false, // Make solid for better visibility
            opacity: 1.0
        });
        const marker = new THREE.Mesh(geometry, material);

        // Map mathematical coordinates to visual coordinates
        const visualSize = 8;
        const sceneX = ((optimum.x - (range.max + range.min) / 2) / (range.max - range.min)) * visualSize;
        const sceneY = ((optimum.y - (range.max + range.min) / 2) / (range.max - range.min)) * visualSize;

        // Calculate Z position using same scaling as surface
        const z = func.func(optimum.x, optimum.y);
        let scaledZ = this.getScaledZ(z);

        marker.position.set(sceneX, sceneY, scaledZ + 0.2); // Higher above surface
        marker.name = 'optimumMarker';

        console.log('Creating optimum marker at:', sceneX, sceneY, scaledZ + 0.2); // Debug log

        // Add a small pulsing animation
        marker.scale.setScalar(1.0);

        this.scene.add(marker);
    }

    createUI() {
        // Try to find a separate controls container, otherwise create overlay
        const controlsContainer = document.getElementById('visualizationControls');
        const useOverlay = !controlsContainer;

        // Clear any existing controls to prevent duplicates
        if (controlsContainer) {
            controlsContainer.innerHTML = '';
        }

        const controlPanel = document.createElement('div');

        if (useOverlay) {
            controlPanel.style.cssText = `
                position: absolute;
                top: 10px;
                left: 10px;
                background: rgba(255, 255, 255, 0.9);
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #ddd;
                font-family: system-ui, sans-serif;
                font-size: 14px;
                z-index: 100;
            `;
        } else {
            controlPanel.style.cssText = `
                font-family: system-ui, sans-serif;
                font-size: 14px;
            `;
        }

        // Function selector
        const functionSelect = document.createElement('select');
        functionSelect.style.cssText = 'width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;';
        Object.keys(this.functions).forEach(key => {
            const option = document.createElement('option');
            option.value = key;
            option.textContent = this.functions[key].name;
            functionSelect.appendChild(option);
        });
        functionSelect.value = this.currentFunction;
        functionSelect.addEventListener('change', (e) => {
            console.log('Function changed to:', e.target.value); // Debug log
            this.currentFunction = e.target.value;
            this.createSurface();
            this.resetOptimization();

            // Force immediate render to ensure surface updates are visible
            this.renderer.render(this.scene, this.camera);
        });

        // Algorithm selector
        const algorithmSelect = document.createElement('select');
        algorithmSelect.style.cssText = 'width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;';

        // Comprehensive algorithm list from test suite
        const algorithms = [
            // PRIMA algorithms
            { value: 'PRIMA_UOBYQA', text: 'PRIMA UOBYQA' },
            { value: 'PRIMA_NEWUOA', text: 'PRIMA NEWUOA' },
            { value: 'PRIMA_BOBYQA', text: 'PRIMA BOBYQA' },

            // SciPy algorithms
            { value: 'SciPy_NelderMead', text: 'Nelder-Mead (SciPy)' },
            { value: 'SciPy_Powell', text: 'Powell Method' },
            { value: 'SciPy_BFGS', text: 'BFGS (SciPy)' },
            { value: 'DifferentialEvolution', text: 'Differential Evolution' },
            { value: 'SimulatedAnnealing', text: 'Simulated Annealing' },

            // Evolutionary algorithms
            { value: 'GeneticAlgorithm', text: 'Genetic Algorithm' },
            { value: 'EvolutionStrategy', text: 'Evolution Strategy' },

            // Swarm intelligence
            { value: 'ParticleSwarm', text: 'Particle Swarm Optimization' },

            // Advanced optimization
            { value: 'CMAEvolutionStrategy', text: 'CMA Evolution Strategy' },
            { value: 'BayesianOpt', text: 'Bayesian Optimization' },

            // Search algorithms
            { value: 'RandomSearch', text: 'Random Search' },
            { value: 'AdaptiveRandomSearch', text: 'Adaptive Random Search' },
            { value: 'CoordinateDescent', text: 'Coordinate Descent' },
            { value: 'PatternSearch', text: 'Pattern Search' },
            { value: 'HillClimbing', text: 'Hill Climbing' },

            // Metaheuristics
            { value: 'TabuSearch', text: 'Tabu Search' },
            { value: 'FireflyAlgorithm', text: 'Firefly Algorithm' },
            { value: 'AntColonyOpt', text: 'Ant Colony Optimization' },
            { value: 'HarmonySearch', text: 'Harmony Search' }
        ];

        algorithms.forEach(alg => {
            const option = document.createElement('option');
            option.value = alg.value;
            option.textContent = alg.text;
            algorithmSelect.appendChild(option);
        });

        // Default to Harmony Search if we're on the harmony search page
        if (document.title.includes('Harmony Search')) {
            algorithmSelect.value = 'HarmonySearch';
        }

        // Control buttons container
        const buttonContainer = document.createElement('div');
        buttonContainer.style.cssText = 'display: flex; gap: 8px;';

        const buttonStyle = 'padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; font-weight: 500;';

        const startBtn = document.createElement('button');
        startBtn.textContent = 'Start';
        startBtn.style.cssText = buttonStyle + 'background: #007bff; color: white;';
        startBtn.addEventListener('click', () => this.startOptimization(algorithmSelect.value));

        const stopBtn = document.createElement('button');
        stopBtn.textContent = 'Stop';
        stopBtn.style.cssText = buttonStyle + 'background: #dc3545; color: white;';
        stopBtn.addEventListener('click', () => this.stopOptimization());

        const resetBtn = document.createElement('button');
        resetBtn.textContent = 'Reset';
        resetBtn.style.cssText = buttonStyle + 'background: #6c757d; color: white;';
        resetBtn.addEventListener('click', () => this.resetOptimization());

        buttonContainer.appendChild(startBtn);
        buttonContainer.appendChild(stopBtn);
        buttonContainer.appendChild(resetBtn);

        // Status display
        const statusDiv = document.createElement('div');
        statusDiv.id = 'optimizationStatus';
        statusDiv.style.cssText = 'font-size: 14px; color: #666; padding: 8px; background: #f8f9fa; border-radius: 4px; margin-top: 10px;';
        statusDiv.innerHTML = 'Ready to optimize';

        if (useOverlay) {
            // Old overlay layout
            controlPanel.appendChild(document.createTextNode('Function: '));
            controlPanel.appendChild(functionSelect);
            controlPanel.appendChild(document.createElement('br'));
            controlPanel.appendChild(document.createTextNode('Algorithm: '));
            controlPanel.appendChild(algorithmSelect);
            controlPanel.appendChild(document.createElement('br'));

            // Add speed selector for overlay layout
            controlPanel.appendChild(document.createTextNode('Speed: '));
            const speedSelect = document.createElement('select');
            speedSelect.style.cssText = 'padding: 4px; border: 1px solid #ddd; border-radius: 4px; margin-left: 5px;';
            const speedOptions = [
                { value: 'slow', text: 'Slow (3s)' },
                { value: 'medium', text: 'Medium (1.5s)' },
                { value: 'fast', text: 'Fast (0.5s)' },
                { value: 'very-fast', text: 'Very Fast (0.2s)' }
            ];
            speedOptions.forEach(opt => {
                const option = document.createElement('option');
                option.value = opt.value;
                option.textContent = opt.text;
                speedSelect.appendChild(option);
            });
            speedSelect.value = this.animationSpeed;
            speedSelect.addEventListener('change', (e) => {
                this.animationSpeed = e.target.value;
                console.log('Speed changed to:', this.animationSpeed, 'delay:', this.getAnimationDelay() + 'ms');
            });
            controlPanel.appendChild(speedSelect);
            controlPanel.appendChild(document.createElement('br'));

            controlPanel.appendChild(buttonContainer);
            controlPanel.appendChild(statusDiv);
            this.container.appendChild(controlPanel);
        } else {
            // Clean horizontal layout below visualization
            const rowDiv = document.createElement('div');
            rowDiv.style.cssText = 'display: flex; gap: 20px; align-items: start; flex-wrap: wrap; margin-bottom: 15px;';

            const functionDiv = document.createElement('div');
            functionDiv.style.cssText = 'min-width: 200px;';
            functionDiv.innerHTML = '<label style="display: block; margin-bottom: 5px; font-weight: 600;">Test Function:</label>';
            functionDiv.appendChild(functionSelect);

            const algorithmDiv = document.createElement('div');
            algorithmDiv.style.cssText = 'min-width: 200px;';
            algorithmDiv.innerHTML = '<label style="display: block; margin-bottom: 5px; font-weight: 600;">Algorithm:</label>';
            algorithmDiv.appendChild(algorithmSelect);

            // View type selector
            const viewDiv = document.createElement('div');
            viewDiv.style.cssText = 'min-width: 150px;';
            viewDiv.innerHTML = '<label style="display: block; margin-bottom: 5px; font-weight: 600;">View:</label>';

            const viewSelect = document.createElement('select');
            viewSelect.style.cssText = 'width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;';
            const viewOptions = [
                { value: '3d', text: '3D Surface' },
                { value: 'wireframe', text: '3D Wireframe' }
            ];
            viewOptions.forEach(opt => {
                const option = document.createElement('option');
                option.value = opt.value;
                option.textContent = opt.text;
                viewSelect.appendChild(option);
            });
            viewSelect.addEventListener('change', (e) => {
                this.updateViewMode(e.target.value);
            });
            viewDiv.appendChild(viewSelect);

            // Speed selector
            const speedDiv = document.createElement('div');
            speedDiv.style.cssText = 'min-width: 120px;';
            speedDiv.innerHTML = '<label style="display: block; margin-bottom: 5px; font-weight: 600;">Speed:</label>';

            const speedSelect = document.createElement('select');
            speedSelect.style.cssText = 'width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;';
            const speedOptions = [
                { value: 'slow', text: 'Slow (3s)' },
                { value: 'medium', text: 'Medium (1.5s)' },
                { value: 'fast', text: 'Fast (0.5s)' },
                { value: 'very-fast', text: 'Very Fast (0.2s)' }
            ];
            speedOptions.forEach(opt => {
                const option = document.createElement('option');
                option.value = opt.value;
                option.textContent = opt.text;
                speedSelect.appendChild(option);
            });
            speedSelect.value = this.animationSpeed;
            speedSelect.addEventListener('change', (e) => {
                this.animationSpeed = e.target.value;
                console.log('Speed changed to:', this.animationSpeed, 'delay:', this.getAnimationDelay() + 'ms');
            });
            speedDiv.appendChild(speedSelect);

            const controlDiv = document.createElement('div');
            controlDiv.innerHTML = '<label style="display: block; margin-bottom: 5px; font-weight: 600;">Controls:</label>';
            controlDiv.appendChild(buttonContainer);

            rowDiv.appendChild(functionDiv);
            rowDiv.appendChild(algorithmDiv);
            rowDiv.appendChild(viewDiv);
            rowDiv.appendChild(speedDiv);
            rowDiv.appendChild(controlDiv);

            controlPanel.appendChild(rowDiv);
            controlPanel.appendChild(statusDiv);
            controlsContainer.appendChild(controlPanel);
        }
    }

    startOptimization(algorithm) {
        if (this.isRunning) return;

        this.isRunning = true;
        this.resetPath();

        // Create the real algorithm instance
        const range = this.functions[this.currentFunction].range;
        const func = this.functions[this.currentFunction];

        // Create objective function that maps from [0,1]² to function range
        const objective = (x) => {
            const mappedX = x[0] * (range.max - range.min) + range.min;
            const mappedY = x[1] * (range.max - range.min) + range.min;
            return func.func(mappedX, mappedY);
        };

        // Create algorithm instance with path tracking
        let optimizer;
        try {
            // Use more trials for better visualization and force minimum steps
            const nTrials = 1000;
            const minVisualizationSteps = 25;

            switch (algorithm) {
                case 'HarmonySearch':
                    optimizer = new HarmonySearch(objective, nTrials, 2);
                    break;
                case 'RandomSearch':
                    optimizer = new RandomSearch(objective, nTrials, 2);
                    break;
                case 'HillClimbing':
                    optimizer = new HillClimbing(objective, nTrials, 2);
                    break;
                case 'SimulatedAnnealing':
                    optimizer = new SimulatedAnnealing(objective, nTrials, 2);
                    break;
                case 'SciPy_NelderMead':
                    optimizer = new NelderMead(objective, nTrials, 2);
                    break;
                case 'ParticleSwarm':
                    optimizer = new ParticleSwarm(objective, nTrials, 2);
                    break;
                case 'DifferentialEvolution':
                    optimizer = new DifferentialEvolution(objective, nTrials, 2);
                    break;
                case 'GeneticAlgorithm':
                    optimizer = new GeneticAlgorithm(objective, nTrials, 2);
                    break;
                case 'BayesianOpt':
                    optimizer = new BayesianOpt(objective, nTrials, 2);
                    break;
                case 'PRIMA_UOBYQA':
                    optimizer = new PRIMA_UOBYQA(objective, nTrials, 2);
                    break;
                case 'PRIMA_NEWUOA':
                    optimizer = new PRIMA_NEWUOA(objective, nTrials, 2);
                    break;
                case 'PRIMA_BOBYQA':
                    optimizer = new PRIMA_BOBYQA(objective, nTrials, 2);
                    break;
                case 'SciPy_Powell':
                    optimizer = new Powell(objective, nTrials, 2);
                    break;
                case 'SciPy_BFGS':
                    optimizer = new LBFGSB(objective, nTrials, 2);
                    break;
                case 'EvolutionStrategy':
                    optimizer = new EvolutionStrategy(objective, nTrials, 2);
                    break;
                case 'CMAEvolutionStrategy':
                    optimizer = new CMAEvolutionStrategy(objective, nTrials, 2);
                    break;
                case 'AdaptiveRandomSearch':
                    optimizer = new AdaptiveRandomSearch(objective, nTrials, 2);
                    break;
                case 'CoordinateDescent':
                    optimizer = new CoordinateDescent(objective, nTrials, 2);
                    break;
                case 'PatternSearch':
                    optimizer = new PatternSearch(objective, nTrials, 2);
                    break;
                case 'TabuSearch':
                    optimizer = new TabuSearch(objective, nTrials, 2);
                    break;
                case 'FireflyAlgorithm':
                    optimizer = new FireflyAlgorithm(objective, nTrials, 2);
                    break;
                case 'AntColonyOpt':
                    optimizer = new AntColonyOpt(objective, nTrials, 2);
                    break;
                default:
                    optimizer = new RandomSearch(objective, nTrials, 2);
                    break;
            }

            optimizer.trackPath = true;

            // Override optimization for better visualization - force minimum steps
            const originalOptimize = optimizer.optimize.bind(optimizer);
            optimizer.optimize = function() {
                // Override evaluate to track more frequently and prevent early stopping
                const originalEvaluate = this.evaluate.bind(this);
                this.evaluate = function(x) {
                    const result = originalEvaluate(x);

                    // Force path tracking every 5 evaluations for visualization
                    if (this.trackPath && (this.evaluations % 5 === 0 || this.evaluations === 1)) {
                        const clippedX = x.map(val => Math.max(0, Math.min(1, val)));
                        this.path.push([...clippedX]);
                    }

                    return result;
                };

                // Force algorithm to run for minimum evaluations by temporarily inflating best value
                const originalBestValue = this.bestValue;

                // Run original optimization
                const result = originalOptimize();

                // If we don't have enough path points, add some exploratory steps
                while (this.path.length < minVisualizationSteps && this.evaluations < nTrials) {
                    // Add some random exploration around current best
                    const noise = 0.1;
                    const x = this.bestX.map(val => {
                        const noisy = val + (Math.random() - 0.5) * noise;
                        return Math.max(0, Math.min(1, noisy));
                    });
                    this.evaluate(x);
                }

                return result;
            };

            // Run optimization and animate the path
            setTimeout(() => {
                optimizer.optimize();
                console.log(`${algorithm} completed with ${optimizer.evaluations} evaluations, ${optimizer.path.length} path points`);
                this.animateRealPath(optimizer.path, algorithm);
            }, 100);

        } catch (error) {
            console.error('Algorithm error:', algorithm, error);
            document.getElementById('optimizationStatus').innerHTML =
                `Error running ${algorithm}: ${error.message}`;
            this.isRunning = false;
        }
    }

    animateRealPath(algorithmPath, algorithmName) {
        console.log(`Starting animation for ${algorithmName}:`, algorithmPath?.length, 'path points');

        if (!algorithmPath || algorithmPath.length === 0) {
            console.error('No path data for', algorithmName);
            document.getElementById('optimizationStatus').innerHTML = 'No path data available';
            this.isRunning = false;
            return;
        }

        let pathIndex = 0;
        const totalSteps = algorithmPath.length;

        const animateStep = () => {
            if (!this.isRunning || pathIndex >= totalSteps) {
                this.isRunning = false;
                document.getElementById('optimizationStatus').innerHTML =
                    `${algorithmName} completed<br>Steps: ${totalSteps}<br>Best value: ${this.evaluateFunction(this.currentPosition.x, this.currentPosition.y).toFixed(6)}`;
                return;
            }

            // Get current point from algorithm path (in [0,1]² space)
            const pathPoint = algorithmPath[pathIndex];

            // Convert to function range
            const range = this.functions[this.currentFunction].range;
            const newPosition = {
                x: pathPoint[0] * (range.max - range.min) + range.min,
                y: pathPoint[1] * (range.max - range.min) + range.min
            };

            // Update position and add to visual path
            this.currentPosition = newPosition;
            this.addToPath(newPosition);

            // Update status
            const currentValue = this.evaluateFunction(newPosition.x, newPosition.y);
            document.getElementById('optimizationStatus').innerHTML =
                `Running ${algorithmName}...<br>Step: ${pathIndex + 1}/${totalSteps}<br>Current value: ${currentValue.toFixed(6)}`;

            pathIndex++;

            // Continue animation with selected speed
            const delay = this.getAnimationDelay();
            console.log(`Animation step ${pathIndex}/${totalSteps}, speed: ${this.animationSpeed}, delay: ${delay}ms`);
            setTimeout(animateStep, delay);
        };

        // Start animation
        animateStep();
    }


    evaluateFunction(x, y) {
        return this.functions[this.currentFunction].func(x, y);
    }

    getAnimationDelay() {
        // Map speed setting to milliseconds
        switch (this.animationSpeed) {
            case 'slow': return 3000;    // 3 seconds
            case 'fast': return 500;     // 0.5 seconds
            case 'very-fast': return 200; // 0.2 seconds
            case 'medium':
            default: return 1500;        // 1.5 seconds
        }
    }

    addToPath(position) {
        const func = this.functions[this.currentFunction];
        const range = func.range;

        // Convert mathematical coordinates to visual coordinates
        const visualSize = 8;
        const sceneX = ((position.x - (range.max + range.min) / 2) / (range.max - range.min)) * visualSize;
        const sceneY = ((position.y - (range.max + range.min) / 2) / (range.max - range.min)) * visualSize;

        // Calculate Z position using same scaling as surface
        const z = func.func(position.x, position.y);
        const scaledZ = this.getScaledZ(z);

        // Create NEW SAMPLE POINT - Large and bright red to show where we're sampling
        const geometry = new THREE.SphereGeometry(0.12, 16, 16);
        const material = new THREE.MeshBasicMaterial({
            color: 0xff0000,  // BRIGHT RED for new sample point
            transparent: false, // Make solid for better visibility
            opacity: 1.0
        });
        const marker = new THREE.Mesh(geometry, material);

        // Position marker well above surface for visibility
        marker.position.set(sceneX, sceneY, scaledZ + 0.2);
        marker.name = 'pathMarker';

        // Make NEW point extra large to show it's the current sample
        marker.scale.setScalar(2.5);  // Much bigger!

        console.log('Creating path marker at:', sceneX, sceneY, scaledZ + 0.2); // Debug log

        this.scene.add(marker);
        this.optimizerPath.push(marker);

        // Force render to show new marker immediately
        this.renderer.render(this.scene, this.camera);

        // Shrink and recolor previous markers to show progression
        this.optimizerPath.forEach((oldMarker, index) => {
            if (oldMarker !== marker) {
                const age = this.optimizerPath.length - index - 1;

                // Shrink previous markers
                oldMarker.scale.setScalar(Math.max(0.6, 1.0 - age * 0.1));

                // Fade to red and reduce opacity based on age
                const redIntensity = Math.max(0.3, 1.0 - age * 0.1);
                oldMarker.material.color.setHex(0xff0000 + Math.floor((1-redIntensity) * 0x004400)); // Red to dark red
                oldMarker.material.opacity = Math.max(0.3, 1.0 - age * 0.08);
            }
        });

        // Limit path length for performance
        if (this.optimizerPath.length > 50) { // Reduced for better visibility
            const oldMarker = this.optimizerPath.shift();
            this.scene.remove(oldMarker);
        }
    }

    resetPath() {
        this.optimizerPath.forEach(marker => this.scene.remove(marker));
        this.optimizerPath = [];
    }

    stopOptimization() {
        this.isRunning = false;
        document.getElementById('optimizationStatus').innerHTML = 'Optimization stopped';
    }

    resetOptimization() {
        this.stopOptimization();
        this.resetPath();
        document.getElementById('optimizationStatus').innerHTML = 'Ready to optimize';
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    onWindowResize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }

    destroy() {
        this.stopOptimization();
        this.container.removeChild(this.renderer.domElement);
        this.scene.clear();
    }
}

// Export for use in algorithm pages
window.AlgorithmVisualizer = AlgorithmVisualizer;