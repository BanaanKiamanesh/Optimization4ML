import numpy as np
import matplotlib.pyplot as plt
from Func import *
from Optimizers import *
plt.style.use("dark_background")

# Initial Point
start = (0.0, 0.0)

#--------------------->  Run Optimizers
traj_gd       = gradient_descent(grad_f, start,  lr=0.01, steps=500)
traj_rmsprop  =          rmsprop(grad_f, start,  lr=0.01, steps=500, beta=0.9, epsilon=1e-8)
traj_adam     =             adam(grad_f, start,  lr=0.01, steps=500, beta1=0.9, beta2=0.999, epsilon=1e-8)
traj_momentum =      momentum_gd(grad_f, start, lr=0.001, steps=500, momentum=0.5)

#--------------------->  Plotting
fig = plt.figure(figsize=(9, 7), dpi=130)
ax = fig.add_subplot(111, projection="3d")

# Optimzier Trajectories
def plot_trajectory(traj, label, color):
    ax.plot(traj[:, 0], traj[:, 1], f(traj[:, 0], traj[:, 1]), color=color, label=label, alpha=0.7)

plot_trajectory(      traj_gd, "Gradient Descent",    "cyan")
plot_trajectory( traj_rmsprop,          "RMSProp", "magenta")
plot_trajectory(    traj_adam,             "Adam",  "yellow")
plot_trajectory(traj_momentum,      "Momentum GD",  "orange")

# Plot Start Point
ax.plot([start[0]], [start[1]], [f(start[0], start[1])], marker="o", color="g", markersize=8, label="Start Point")

ax.legend()

# 3D Contour Plot
# Create a Grid of Points
w1 = np.linspace(-6, 6, 400)
w2 = np.linspace(-6, 6, 400)
W1, W2 = np.meshgrid(w1, w2)
Y = f(W1, W2)

ax.contour3D(W1, W2, Y, 60)
ax.set_xlabel("w1")
ax.set_ylabel("w2")
ax.set_zlabel("y")
ax.set_title("3D Contour of y = f(w1, w2)")
ax.view_init(elev=60, azim=-60)

plt.tight_layout()
plt.show()
