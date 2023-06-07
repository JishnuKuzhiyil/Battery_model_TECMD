# Equivalent Circuit Model (ECM) with Diffusion and Thermal Dynamics for Lithium-ion Batteries

This repository presents a Julia implementation of a first-order Equivalent Circuit Model (ECM) with diffusion dynamics and thermal dynamics. The diffusion model is based on a third-order implementation of Fick's law, as proposed in the groundbreaking work by Xiong-2022 (Applied Energy, https://doi.org/10.1016/j.apenergy.2022.118915). To enhance accuracy, a zero-dimensional lumped thermal equation has been incorporated to capture temperature variations of the system parameters.

The model is implemented as a 7th order differential-algebraic system, consisting of six ordinary differential equations (ODEs) and one algebraic constraint, specifically designed to simulate the voltage hold phase. 

