# HumpDay Browser Demo

This directory contains the browser-based interactive demonstration of HumpDay optimization package.

## Features

🏁 **Live Optimizer Racing** - Watch different scipy optimizers compete in real-time
📊 **Interactive Visualizations** - See convergence curves update as optimizers run  
🌐 **Zero Installation** - Runs entirely in your browser using Pyodide
📱 **Mobile Friendly** - Works on phones and tablets
🔗 **Shareable** - Send links to specific optimization problems

## How It Works

The demo uses [Pyodide](https://pyodide.org/) to run Python directly in the browser, including:
- NumPy for numerical computing
- SciPy optimizers (Powell, Nelder-Mead, L-BFGS-B, TNC)
- Plotly for interactive plotting

## Objective Functions

- **Sphere**: Simple quadratic function (easy to optimize)
- **Rosenbrock**: Classic "banana function" with narrow valley (medium difficulty)
- **Rastrigin**: Highly multimodal with many local minima (hard)  
- **Ackley**: Nearly flat outer region with central spike (very hard)

## Local Development

To run locally:
```bash
# Simple HTTP server (Python 3)
python -m http.server 8000

# Or with Node.js
npx serve .

# Then open http://localhost:8000
```

Note: Must use HTTP server (not file://) due to Pyodide CORS requirements.

## Future Enhancements

- [ ] Add more objective functions from humpday package
- [ ] Thurstone-based ranking system  
- [ ] Custom objective function input
- [ ] 3D visualization of optimization landscapes
- [ ] Comparison with heavier optimizers (Optuna, etc.)
- [ ] Save and share optimization results
- [ ] Educational mode with explanations
