        // Initialize the visualization when page loads
        document.addEventListener('DOMContentLoaded', function() {
            // Check for WebGL support
            function webglAvailable() {
                try {
                    const canvas = document.createElement('canvas');
                    return !!(window.WebGLRenderingContext && canvas.getContext('webgl'));
                } catch (e) {
                    return false;
                }
            }

            if (webglAvailable()) {
                const visualizer = new AlgorithmVisualizer('algorithmVisualization', {
                    onReady: () => {
                        // Clear loading message once visualization is ready
                        const loadingMsg = document.getElementById('loadingMessage');
                        if (loadingMsg) {
                            loadingMsg.style.display = 'none';
                        }
                    }
                });
            } else {
                document.getElementById('loadingMessage').innerHTML =
                    '<h3>WebGL Not Supported</h3>' +
                    '<p>Your browser does not support WebGL, which is required for 3D visualization.</p>' +
                    '<p>Please try a modern browser like Chrome, Firefox, or Safari.</p>';
            }
        });