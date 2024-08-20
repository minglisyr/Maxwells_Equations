import numpy as np
import matplotlib.pyplot as plt

def electric_field(x, y, q, x0, y0):
    k = 8.99e9  # Coulomb's constant
    dx = x - x0
    dy = y - y0
    r = np.sqrt(dx**2 + dy**2)
    E_x = k * q * dx / r**3
    E_y = k * q * dy / r**3
    return E_x, E_y

x = np.linspace(-5, 5, 20)
y = np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x, y)

plt.figure(figsize=(10, 8))
plt.streamplot(X, Y, Ex, Ey, density=1, linewidth=1, arrowsize=1.5, arrowstyle='->')
plt.title("Electric Field of a Point Charge")
plt.xlabel("x")
plt.ylabel("y")
plt.axhline(y=0, color='k', linestyle='--')
plt.axvline(x=0, color='k', linestyle='--')
plt.grid(True)
plt.show()