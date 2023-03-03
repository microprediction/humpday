### Install
Pick one of:

    pip install humpday
    pip install humpday[full]

The full option will try to install a slew of optim packages. You may prefer to do that piecemeal. See below. 

Bleeding edge:

    pip install git+https://github.com/microprediction/humpday
  
File an issue if you have problems. See [this thread](https://stackoverflow.com/questions/65745683/how-to-install-scipy-on-apple-silicon-arm-m1) if you have 
issues on mac silicon M1. 

#### This might help some of you sometimes

    pip install cython pybind11
    brew install openblas
    export OPENBLAS=/opt/homebrew/opt/openblas/lib/

#### Installing one optimizer at a time 

    pip install scikit-optimize
    pip install optuna
    pip install platypus-opt
    pip install poap
    pip install pysot
    
Some of these are really good, but not 100% stable on all platforms we've used. 

    pip install cmake
    pip install ultraopt
    pip install dlib 
    pip install ax-platform
    pip install py-bobyqa
    pip install hebo
    pip install nlopt
    pip install freelunch
    
Broken pending [issue](https://github.com/fmfn/BayesianOptimization/issues/300):
    
    pip install bayesian-optimization
    pip install nevergrad
 

