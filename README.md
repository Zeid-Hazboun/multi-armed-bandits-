# multi-armed-bandits-
This is the code of an assignment for the course "reinforcement learning".  We were tasked with the creation of $k$-armed bandits 
that follow different strategies and rewards sampled from different distributions.

### The strategies implemented
* $\epsilon$-Greedy
* Optimistic Initial Values
* Upper Confidence Bound
* Action Preferences

### The sampling distributions:
* Guaussian sampling distribution
* Bernoulli sampling distribution

For each sampling distribution each algorithm was used, therefore the resulting graph has 8 plots on it. The hyper-parameters that can easily be 
manipulated from the terminal are the number of arms, the number of epochs and the $\epsilon$ (exploration probability)
