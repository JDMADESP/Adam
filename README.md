## SGD vs Adam: Optimization Experiment

We compare SGD and Adam on the quadratic loss:

f(x, y) = 100x² + y²

Both optimizers start at (2, 2), using learning rates
of 0.005 for SGD and 0.1 for Adam.

### Optimization Trajectories

![Optimization Trajectories](plots/optimization_trajectories.png)

### Convergence

![SGD vs Adam Convergence](plots/convergence.png)