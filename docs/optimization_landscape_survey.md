# Comprehensive Survey: Optimization Packages & Benchmarking Suites

*Last updated: 2026-05-20*

## 🎯 **Derivative-Free Optimization Packages** 

### **Premier Libraries**

#### **1. PRIMA** - Powell's Reference Implementation
- **URL**: https://github.com/libprima/prima
- **Description**: Reference implementation of Powell's methods with modernization  
- **Methods**: COBYLA, UOBYQA, NEWUOA, BOBYQA, LINCOA
- **Languages**: Fortran, C, Python, MATLAB, Julia
- **Key Features**: 
  - Bug-fixed versions of Powell's original algorithms
  - Battle-tested with 200,000+ hours of automated tests
  - Performance improvements over original Fortran 77
  - Modern, maintainable code structure
- **Status**: ✅ Active, production-ready
- **Integration**: High priority for HumpDay

#### **2. SciPy optimize**
- **URL**: https://docs.scipy.org/doc/scipy/reference/optimize.html  
- **Description**: Comprehensive optimization toolkit
- **Derivative-Free Methods**: 
  - Nelder-Mead, Powell, COBYLA (now uses PRIMA), SLSQP
  - Differential Evolution, Basin Hopping, SHGO, Dual Annealing
- **Languages**: Python (with Fortran/C backends)
- **Status**: ✅ Industry standard
- **Integration**: ✅ Already integrated in HumpDay

#### **3. NLopt**
- **URL**: https://github.com/stevengj/nlopt
- **Description**: Nonlinear optimization library
- **Derivative-Free Methods**: 
  - DIRECT, CRS2, MLSL, StoGO, ISRES, ESCH, COBYLA, BOBYQA, NEWUOA, PRAXIS, Nelder-Mead, Subplex
- **Languages**: C/C++ with bindings for Python, MATLAB, Julia, etc.
- **Key Features**: Unified interface for diverse algorithms
- **Status**: ✅ Active, widely used
- **Integration**: 🎯 Should add to HumpDay

#### **4. Nevergrad**
- **URL**: https://github.com/facebookresearch/nevergrad
- **Description**: Facebook's gradient-free optimization library
- **Methods**: 200+ optimizers including evolutionary strategies, PSO, Bayesian optimization
- **Languages**: Python
- **Key Features**: Extensive algorithm collection, modern research methods
- **Status**: ✅ Active research library
- **Integration**: ✅ Already integrated in HumpDay

#### **5. Optuna**
- **URL**: https://github.com/optuna/optuna
- **Description**: Hyperparameter optimization framework
- **Methods**: TPE, CMA-ES, Random Search, Grid Search, Bayesian optimization
- **Languages**: Python
- **Key Features**: AutoML focus, distributed optimization, pruning
- **Status**: ✅ Widely adopted in ML
- **Integration**: ✅ Already integrated in HumpDay

### **Specialized Packages**

#### **6. CMA-ES (pycma)**
- **URL**: https://github.com/CMA-ES/pycma
- **Description**: Covariance Matrix Adaptation Evolution Strategy
- **Methods**: CMA-ES variants, bipop-CMA-ES, IPOP-CMA-ES
- **Languages**: Python, MATLAB
- **Status**: ✅ Reference implementation
- **Integration**: 🎯 Should add to HumpDay

#### **7. DEAP**
- **URL**: https://github.com/DEAP/deap
- **Description**: Distributed Evolutionary Algorithms in Python
- **Methods**: GA, GP, ES, PSO, DE variants
- **Languages**: Python
- **Status**: ✅ Academic/research favorite
- **Integration**: 🎯 Consider for HumpDay

#### **8. PyGMO/pagmo2**
- **URL**: https://github.com/esa/pagmo2
- **Description**: European Space Agency's optimization library
- **Methods**: 100+ algorithms including metaheuristics, local search
- **Languages**: C++ with Python bindings
- **Key Features**: Multi-objective, parallel, archipelago model
- **Status**: ✅ Active, aerospace industry
- **Integration**: 🎯 High-quality addition for HumpDay

#### **9. Platypus**
- **URL**: https://github.com/Project-Platypus/Platypus
- **Description**: Multiobjective optimization library
- **Methods**: NSGA-II, NSGA-III, SPEA2, MOEA/D, etc.
- **Languages**: Python
- **Focus**: Multi-objective optimization
- **Status**: ✅ Active
- **Integration**: 🤔 If HumpDay adds multi-objective support

#### **10. NOMAD**
- **URL**: https://github.com/bbopt/nomad
- **Description**: Mesh Adaptive Direct Search algorithm
- **Methods**: MADS, BiMADS, PSD-MADS
- **Languages**: C++ with Python/MATLAB interfaces
- **Key Features**: Handles general constraints, blackbox optimization
- **Status**: ✅ Active research
- **Integration**: 🎯 Excellent addition for HumpDay

### **Bayesian Optimization**

#### **11. scikit-optimize**
- **URL**: https://github.com/scikit-optimize/scikit-optimize
- **Description**: Sequential model-based optimization
- **Methods**: Gaussian Processes, Random Forest, Gradient Boosted Trees
- **Languages**: Python
- **Status**: ⚠️ Maintenance mode (but still valuable)
- **Integration**: 🎯 Classic Bayesian optimization

#### **12. GPyOpt**
- **URL**: https://github.com/SheffieldML/GPyOpt
- **Description**: Gaussian Process optimization
- **Methods**: Expected Improvement, UCB, Probability of Improvement
- **Languages**: Python
- **Status**: ⚠️ Less active
- **Integration**: 🤔 Consider alternatives

#### **13. Hyperopt**
- **URL**: https://github.com/hyperopt/hyperopt
- **Description**: Distributed asynchronous hyperparameter optimization
- **Methods**: Random Search, TPE, Adaptive TPE
- **Languages**: Python
- **Status**: ✅ Mature, widely used
- **Integration**: 🎯 Popular in ML

#### **14. BoTorch**
- **URL**: https://github.com/pytorch/botorch
- **Description**: Bayesian optimization in PyTorch
- **Methods**: Multi-task, multi-objective, high-dimensional BO
- **Languages**: Python
- **Key Features**: Modern, scalable, research-oriented
- **Status**: ✅ Active development
- **Integration**: 🎯 State-of-the-art Bayesian methods

### **Metaheuristics & Swarm Intelligence**

#### **15. TPOT**
- **URL**: https://github.com/EpistasisLab/tpot
- **Description**: Tree-based Pipeline Optimization Tool
- **Methods**: Genetic Programming for AutoML
- **Languages**: Python
- **Focus**: ML pipeline optimization
- **Status**: ✅ Active
- **Integration**: 🤔 Specialized for ML

#### **16. pymoo**
- **URL**: https://github.com/anyoptimization/pymoo
- **Description**: Multi-objective optimization in Python
- **Methods**: NSGA-II, NSGA-III, R-NSGA-III, MOEA/D, etc.
- **Languages**: Python
- **Status**: ✅ Very active
- **Integration**: 🎯 If multi-objective support added

#### **17. SMAC3**
- **URL**: https://github.com/automl/SMAC3
- **Description**: Sequential Model-based Algorithm Configuration
- **Methods**: Random Forest-based optimization
- **Languages**: Python
- **Focus**: Algorithm configuration
- **Status**: ✅ Active
- **Integration**: 🎯 Good for hyperparameter optimization

## 🏆 **Benchmarking Suites & Test Problems**

### **Standard Benchmark Collections**

#### **1. CUTEst**
- **URL**: https://github.com/ralna/CUTEst
- **Description**: Constrained and Unconstrained Testing Environment  
- **Problems**: 1500+ optimization problems
- **Languages**: Fortran with interfaces
- **Status**: ✅ Gold standard
- **Integration**: 🎯 Essential for serious benchmarking

#### **2. COCO/BBOB**
- **URL**: https://github.com/numbbo/coco
- **Description**: Comparing Continuous Optimizers platform
- **Problems**: Black-box optimization benchmarking
- **Key Features**: Systematic difficulty progression, performance profiles
- **Status**: ✅ Research standard
- **Integration**: ✅ Already inspiring HumpDay's systematic approach

#### **3. benchmark-functions**
- **URL**: https://github.com/nathanrooy/benchmark-functions
- **Description**: Collection of single-objective test functions
- **Problems**: 30+ classic functions (Sphere, Rastrigin, Rosenbrock, etc.)
- **Languages**: Python
- **Status**: ✅ Simple, practical
- **Integration**: ✅ Already using in HumpDay

#### **4. Optproblems**
- **URL**: https://github.com/andim/optproblems
- **Description**: Infrastructure for optimization benchmarking
- **Problems**: CEC functions, multi-objective problems
- **Languages**: Python
- **Status**: ✅ Academic use
- **Integration**: 🎯 Consider adding

#### **5. SciPy benchmarks**
- **URL**: https://github.com/scipy/scipy/tree/main/benchmarks/benchmarks
- **Description**: SciPy's internal optimization benchmarks
- **Status**: ✅ Reference implementation
- **Integration**: 🎯 Learn from their approach

### **Competition & Challenge Datasets**

#### **6. CEC Competition Functions**
- **Description**: Congress on Evolutionary Computation benchmark functions
- **Years**: CEC 2005, 2013, 2014, 2017, 2020, 2022, etc.
- **Problems**: Shifted, rotated, hybrid, composition functions
- **Integration**: 🎯 Standard in evolutionary computation

#### **7. GECCO Competition**
- **Description**: Genetic and Evolutionary Computation Conference competitions
- **Problems**: Real-parameter optimization, multiobjective optimization
- **Status**: ✅ Annual competitions
- **Integration**: 🎯 Current research problems

#### **8. Black-Box Optimization Competition (BBCOMP)**
- **Description**: AAAI conference competition
- **Problems**: Expensive black-box optimization
- **Status**: ✅ Annual
- **Integration**: 🎯 Realistic expensive problems

### **Application-Specific Benchmarks**

#### **9. OpenAI Gym**
- **URL**: https://github.com/openai/gym
- **Description**: Reinforcement learning environments (policy optimization)
- **Status**: ✅ RL standard
- **Integration**: 🤔 If RL optimization added

#### **10. HPOBench**
- **URL**: https://github.com/automl/HPOBench
- **Description**: Hyperparameter optimization benchmarks
- **Problems**: Tabular benchmarks from real ML problems
- **Status**: ✅ Active
- **Integration**: 🎯 Realistic hyperparameter problems

#### **11. JAX MD**
- **URL**: https://github.com/jax-md/jax-md
- **Description**: Molecular dynamics (energy minimization problems)
- **Status**: ✅ Active
- **Integration**: 🤔 Physics simulations

## 🚀 **Advanced & Emerging Packages**

### **Recent Research Developments**

#### **12. Ax Platform**
- **URL**: https://github.com/facebook/Ax
- **Description**: Meta's adaptive experimentation platform
- **Methods**: Bayesian optimization, multi-objective, bandit algorithms
- **Languages**: Python
- **Status**: ✅ Industry-grade
- **Integration**: 🎯 High-quality modern methods

#### **13. Ray Tune**
- **URL**: https://github.com/ray-project/ray
- **Description**: Distributed hyperparameter tuning
- **Methods**: Population-based training, HyperBand, BOHB
- **Languages**: Python
- **Status**: ✅ Very active, scalable
- **Integration**: 🎯 Distributed optimization

#### **14. HpBandSter**
- **URL**: https://github.com/automl/HpBandSter
- **Description**: HyperBand on steroids
- **Methods**: BOHB (Bayesian Optimization + HyperBand)
- **Status**: ✅ Academic success
- **Integration**: 🎯 Multi-fidelity optimization

#### **15. DEHB**
- **URL**: https://github.com/automl/DEHB  
- **Description**: Differential Evolution + HyperBand
- **Methods**: Multi-fidelity evolutionary optimization
- **Status**: ✅ Research active
- **Integration**: 🎯 Novel approach

### **Domain-Specific Optimizers**

#### **16. DLib**
- **URL**: https://github.com/davisking/dlib
- **Description**: Machine learning toolkit with optimization
- **Methods**: LBFGS, trust region, global optimization
- **Languages**: C++ with Python bindings
- **Status**: ✅ Mature
- **Integration**: 🎯 Well-tested implementations

#### **17. OR-Tools**
- **URL**: https://github.com/google/or-tools
- **Description**: Google's operations research tools
- **Methods**: CP-SAT, linear programming, routing
- **Languages**: C++, Python, Java, C#
- **Focus**: Discrete optimization primarily
- **Status**: ✅ Industry-grade
- **Integration**: 🤔 If discrete optimization added

#### **18. casADi**
- **URL**: https://github.com/casadi/casadi
- **Description**: Optimal control and nonlinear optimization
- **Methods**: IPOPT, SQP, multiple shooting
- **Languages**: C++ with Python/MATLAB bindings
- **Focus**: Control theory, robotics
- **Status**: ✅ Active
- **Integration**: 🤔 Specialized domain

## 📊 **HumpDay Integration Priorities**

### **High Priority (Should Add Immediately)**
1. **PRIMA** - Reference Powell's methods
2. **NLopt** - Comprehensive algorithm collection  
3. **CMA-ES** - Modern evolution strategy
4. **PyGMO/pagmo2** - ESA's high-quality library
5. **NOMAD** - Mesh adaptive direct search
6. **CUTEst** - Gold standard test problems

### **Medium Priority (Research & Development)**
1. **BoTorch** - Modern Bayesian optimization
2. **Ax Platform** - Meta's optimization platform
3. **Ray Tune** - Distributed optimization
4. **scikit-optimize** - Classic Bayesian methods
5. **Hyperopt** - Popular hyperparameter optimization
6. **CEC Competition Functions** - Research benchmarks

### **Future Consideration**
1. **Multi-objective libraries** (pymoo, Platypus) - if MO support added
2. **Reinforcement learning** (OpenAI Gym) - policy optimization
3. **Physics simulations** (JAX MD) - energy minimization
4. **AutoML tools** (TPOT) - specialized applications

## 🔬 **Research Gaps & Opportunities**

### **What HumpDay Brings Uniquely**
1. **Stochastic surface generation** - addresses benchmark bias
2. **Multi-dimensional Thurstone analysis** - sophisticated rating system  
3. **Browser-first approach** - accessibility & interactivity
4. **Educational focus** - teaching optimization concepts
5. **Derivative-free specialization** - focused scope

### **Potential Research Contributions**
1. **Robust benchmark methodology** with stochastic surfaces
2. **Context-aware optimizer recommendation** system
3. **Interactive optimization education** platform
4. **Multi-dimensional performance analysis** beyond simple rankings
5. **Integration with "embarrassingly" robust techniques**

## 📈 **Implementation Roadmap**

### **Phase 1: Core Expansion**
- Add PRIMA integration
- Include NLopt methods  
- Expand CMA-ES variants
- Integrate PyGMO algorithms

### **Phase 2: Benchmarking Enhancement** 
- Add CUTEst problem interface
- Implement CEC competition functions
- Develop systematic BBOB-inspired progression
- Integrate "embarrassingly" robust techniques

### **Phase 3: Advanced Methods**
- Bayesian optimization (BoTorch, Ax)
- Multi-fidelity methods (BOHB, DEHB)
- Distributed optimization (Ray Tune)
- Multi-objective support (pymoo)

### **Phase 4: Research Platform**
- Real-world problem datasets (HPOBench)
- Physics/engineering benchmarks
- AutoML integration
- Industry collaboration

---

*This survey represents the current state of derivative-free optimization as of 2026. The landscape continues to evolve rapidly with new research and applications.*