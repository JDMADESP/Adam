
import numpy as np
import matplotlib.pyplot as plt

from adam import Adam
from sgd import SGD
from pathlib import Path

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

def loss(theta):
    x, y = theta
    return 100 * x**2 + y**2

def gradient(theta):
    x, y = theta
    return np.array([200 * x, 2 * y])

def optimize(optimizer, steps=100):
    trajectory = [optimizer.theta.copy()]
    losses = [loss(optimizer.theta)]

    for _ in range(steps):
        g = gradient(optimizer.theta)
        theta = optimizer.step(g)

        trajectory.append(theta.copy())
        losses.append(loss(theta))

    return np.array(trajectory), np.array(losses)


theta0 = np.array([2.0, 2.0])

sgd = SGD(theta0, lr=0.005)
adam = Adam(theta0, lr=0.1)

sgd_path, sgd_losses = optimize(sgd, steps=100)
adam_path, adam_losses = optimize(adam, steps=100)


x = np.linspace(-2.5, 2.5, 200)
y = np.linspace(-2.5, 2.5, 200)

X, Y = np.meshgrid(x, y)
Z = 100 * X**2 + Y**2

plt.figure(figsize=(9, 6))

plt.contour(
    X, Y, Z,
    levels=[1, 5, 20, 50, 100, 200, 400, 600],
    alpha=0.6
)

plt.plot(
    sgd_path[:, 0],
    sgd_path[:, 1],
    label="SGD",
    marker=".",
    markersize=3
)

plt.plot(
    adam_path[:, 0],
    adam_path[:, 1],
    label="Adam",
    marker=".",
    markersize=3
)

plt.scatter(0, 0, marker="x", s=100, label="Minimum")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Optimization Trajectories: SGD vs Adam")
plt.legend()
plt.grid(alpha=0.2)
plt.savefig(
    PLOTS_DIR / "optimization_trajectories.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


plt.figure(figsize=(9, 5))

plt.plot(sgd_losses, label="SGD")
plt.plot(adam_losses, label="Adam")

plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("SGD vs Adam: Convergence")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(
    PLOTS_DIR / "convergence.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

