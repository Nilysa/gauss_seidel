# Gauss-Seidel & SOR Linear System Solver

## Overview
A comprehensive numerical analysis tool built in Python to solve systems of linear equations using the Gauss-Seidel and Successive Over-Relaxation (SOR) iterative methods. This project goes beyond simple solving by providing deep mathematical insights into the matrix operations and convergence conditions.

## Mathematical Features
* **Matrix Decomposition:** Automatically breaks down the coefficient matrix $A$ into its Diagonal ($D$), Lower Triangular ($L$), and Upper Triangular ($U$) components.
* **Iteration Matrix Calculation:** Computes the iteration matrix $B$ and vector $c$ using the formulas:
  $$B = (D+L)^{-1}(-U)$$
  $$c = (D+L)^{-1}b$$
* **Advanced Convergence Checking:** Before iterating, the system evaluates convergence viability by calculating:
  * L1, L2, and Infinity Matrix Norms
  * Spectral Radius ($
ho(B)$)
  * Strict Diagonal Dominance
* **Precision Tracking:** Tracks the error margin per iteration and calculates the final mathematical residual:
  $$\text{Residual} = \|Ax - b\|$$
* **Report Generation:** Allows users to export the full iteration history, matrix states, and convergence reports directly to a text file.

## Technical Stack
* **Language:** Python 3.8+
* **Libraries:** NumPy (for high-performance matrix inversions and eigenvalue calculations)

## Setup & Execution

1. **Install Dependencies:**
```bash
pip install numpy
```

2. **Run the Solver:**
```bash
python gauss_seidel.py
```

3. **Sample Input:**
When prompted, you can test the system with the following 2x2 matrix data:
* **n:** 2
* **Matrix A:**
  * Row 1: `4 2`
  * Row 2: `1 4`
* **Vector b:** `6 5`
* **Omega ($\omega$):** `1` (for standard Gauss-Seidel)

## Credits

Developed by Niloofar Asoubar & Saina Pourjafari.
