# Gauss-Seidel & SOR Linear System Solver

## Overview
A comprehensive numerical analysis tool built in Python to solve systems of linear equations using the Gauss-Seidel and Successive Over-Relaxation (SOR) iterative methods. This project goes beyond simple solving by providing deep mathematical insights into the matrix operations and convergence conditions.

## Mathematical Features
* **Matrix Decomposition:** Automatically breaks down the coefficient matrix $A$ into its Diagonal ($D$), Lower Triangular ($L$), and Upper Triangular ($U$) components[cite: 15, 17].
* **Iteration Matrix Calculation:** Computes the iteration matrix $B$ and vector $c$ using the formulas:
  $$B = (D+L)^{-1}(-U)$$
  $$c = (D+L)^{-1}b$$
* **Advanced Convergence Checking:** Before iterating, the system evaluates convergence viability by calculating[cite: 15, 17]:
  * L1, L2, and Infinity Matrix Norms
  * Spectral Radius ($\rho(B)$)
  * Strict Diagonal Dominance
* **Precision Tracking:** Tracks the error margin per iteration and calculates the final mathematical residual[cite: 15, 17]:
  $$Residual = \|Ax - b\|$$
* **Report Generation:** Allows users to export the full iteration history, matrix states, and convergence reports directly to a text file[cite: 15, 17].

## Technical Stack
* **Language:** Python 3.8+[cite: 18]
* **Libraries:** NumPy (for high-performance matrix inversions and eigenvalue calculations)[cite: 18]

## Setup & Execution

1. **Install Dependencies:**
```bash
   pip install numpy
```
2. **Run the Solver:**
```Bash

python gauss_seidel.py
```
3. **Sample Input:**
When prompted, you can test the system with the following 2x2 matrix data:

* n: 2

* Matrix A:

    * Row 1: 4 2

    * Row 2: 1 4

* Vector b: 6 5

* Omega (ω): 1 (for standard Gauss-Seidel)

## Credits

Developed by Niloofar Asoubar & Saina Pourjafari.