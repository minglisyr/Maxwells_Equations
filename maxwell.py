import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
k = 8.99e9  # Coulomb's constant
mu0 = 4 * np.pi * 1e-7  # Permeability of free space

# Charge properties
q = 1e-9  # Charge in Coulombs
v = np.array([1, 0, 0])  # Velocity vector (m/s)
x0, y0, z0 = 0, 0, 0  # Position of the charge

# Vector density parameter (adjust this to change the number of vectors)
density = 10  # Number of vectors in each dimension

# Create grid
x = np.linspace(-5, 5, density)
y = np.linspace(-5, 5, density)
z = np.linspace(-5, 5, density)
X, Y, Z = np.meshgrid(x, y, z)

# Electric field function
def electric_field(x, y, z, q, x0, y0, z0):
    dx, dy, dz = x - x0, y - y0, z - z0
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    r[r == 0] = np.inf  # Avoid division by zero
    E = k * q / r**3
    return E * dx, E * dy, E * dz

# Magnetic field function
def magnetic_field(x, y, z, q, v, x0, y0, z0):
    dx, dy, dz = x - x0, y - y0, z - z0
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    r[r == 0] = np.inf  # Avoid division by zero
    r_vec = np.stack([dx, dy, dz], axis=-1)
    v_vec = np.array(v)
    B = (mu0 / (4 * np.pi)) * q * np.cross(v_vec, r_vec) / r[:,:,:,np.newaxis]**3
    return B[:,:,:,0], B[:,:,:,1], B[:,:,:,2]

# Calculate fields
Ex, Ey, Ez = electric_field(X, Y, Z, q, x0, y0, z0)
Bx, By, Bz = magnetic_field(X, Y, Z, q, v, x0, y0, z0)

# Create figure
fig = plt.figure(figsize=(10, 16))

# Electric field subplot
ax1 = fig.add_subplot(211, projection='3d')
ax1.quiver(X, Y, Z, Ex, Ey, Ez, length=0.5, normalize=True)
ax1.set_title(f"Electric Field (Density: {density}x{density}x{density})")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")

# Magnetic field subplot
ax2 = fig.add_subplot(212, projection='3d')
ax2.quiver(X, Y, Z, Bx, By, Bz, length=0.5, normalize=True)
ax2.set_title(f"Magnetic Field (Density: {density}x{density}x{density})")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")

# Plot charge position and velocity
for ax in [ax1, ax2]:
    ax.plot([x0], [y0], [z0], 'ro', markersize=10, label='Charge')
    ax.quiver(x0, y0, z0, v[0], v[1], v[2], color='r', length=2, label='Velocity')
    ax.legend()

plt.tight_layout()
plt.show()