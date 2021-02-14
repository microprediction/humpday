# humpday

- Assigns [Elo ratings](https://github.com/microprediction/optimizer-elo-ratings/tree/main/results/leaderboards/overall) to global derivative-free Python global optimization "strategies", and 
- Presents [many different popular optimizers](https://github.com/microprediction/humpday/tree/main/humpday/optimizers) in a common calling syntax. 
- Supposedly makes it dead easy to choose a Python global optimizer for your bespoke problem. 

### 


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

A few packages are not included in setup as their install isn't quite stable on some operating systems I have tried. So install themm manually:

    pip install ultraopt
    

