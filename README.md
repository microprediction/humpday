# humpday

- Assigns [Elo ratings](https://github.com/microprediction/optimizer-elo-ratings/tree/main/results/leaderboards/overall) to global derivative-free Python global optimization "strategies", and 
- Presents [many different popular optimizers](https://github.com/microprediction/humpday/tree/main/humpday/optimizers) in a common calling syntax. 
- Thereby, takes the pain out of choosing a Python global optimizer for your bespoke problem, assuming a continous domain.   

# Included

At time of writing, there are 52 strategies compared. These are drawn from [Ax-Platform](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/axcube.py), [bayesian-optimization](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/bayesoptcube.py), [DLib](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/dlibcube.py), [HyperOpt](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/hyperoptcube.py), [NeverGrad](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/nevergradcube.py), [Optuna](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/optunacube.py), [Platypus](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/platypuscube.py), [PyMoo](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/pymoocube.py), [PySOT](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/pysotcube.py), Scipy [classic](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/scipycube.py) and [shgo](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/shgocube.py), [Skopt](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/skoptcube.py), and [UltraOpt](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/ultraoptcube.py). 
 


![](https://i.imgur.com/FCiSrMQ.png)

### Explanation

See:
- [HumpDay: A Package to Take the Pain Out of Choosing a Python Optimizer](https://www.microprediction.com/blog/humpday). This is the follow-on article to: 
- [Comparing Python Global Optimizers](https://www.microprediction.com/blog/optimize).


### Install notes

Possibly 

    pip install humpday
    
will simply work, or for the bleeding edge

    pip install git+https://github.com/microprediction/humpday
    
If you get a CMake error, 

    pip install cmake
    pip install humpday 


#### Optional packages

A few packages are not included in setup as their install isn't quite stable on some operating systems I have tried. Please add manually if you wish them to be included. 

    pip install cmake
    pip install ultraopt
    

