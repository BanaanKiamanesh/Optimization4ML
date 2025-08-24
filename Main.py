import numpy as np
import matplotlib.pyplot as plt
from Func import *
plt.style.use("dark_background")

# Plotting
# Create a Grid of Points
w1 = np.linspace(-6, 6, 400)
w2 = np.linspace(-6, 6, 400)
W1, W2 = np.meshgrid(w1, w2)
Y = f(W1, W2)

# 3D Contour Plot
fig = plt.figure(figsize=(9, 7), dpi=130)
ax = fig.add_subplot(111, projection="3d")

ax.contour3D(W1, W2, Y, 60)
ax.set_xlabel("w1")
ax.set_ylabel("w2")
ax.set_zlabel("y")
ax.set_title("3D Contour of y = f(w1, w2)")
ax.view_init(elev=30, azim=-45)

plt.tight_layout()
plt.show()
