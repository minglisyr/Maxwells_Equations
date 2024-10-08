# Maxwell's Equations Visualization

This project visualizes the electric and magnetic fields based on Maxwell's equations.

## Maxwell's Equations

Maxwell's equations are a set of four fundamental equations that describe the behavior of electric and magnetic fields. Here are the equations in their differential form:

1. **Gauss's Law for Electricity:**

   $$\nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0}$$

   This equation relates the electric field to the charge density.

2. **Gauss's Law for Magnetism:**

   $$\nabla \cdot \mathbf{B} = 0$$

   This equation states that magnetic monopoles do not exist.

3. **Faraday's Law of Induction:**

   $$\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}$$

   This equation describes how a changing magnetic field induces an electric field.

4. **Ampère's Law (with Maxwell's correction):**

   $$\nabla \times \mathbf{B} = \mu_0\left(\mathbf{J} + \varepsilon_0\frac{\partial \mathbf{E}}{\partial t}\right)$$

   This equation relates the magnetic field to the current density and the changing electric field.

Where:
- $\mathbf{E}$ is the electric field
- $\mathbf{B}$ is the magnetic field
- $\rho$ is the charge density
- $\mathbf{J}$ is the current density
- $\varepsilon_0$ is the permittivity of free space
- $\mu_0$ is the permeability of free space
- $\nabla \cdot$ is the divergence operator
- $\nabla \times$ is the curl operator

## Dependencies

[numpy, matplotlib]

## License

[GPL 3.0]
