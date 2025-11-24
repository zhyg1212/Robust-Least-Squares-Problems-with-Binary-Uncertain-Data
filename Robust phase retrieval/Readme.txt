This folder implements and compares the robustness of three least squares methods (Standard Least Squares, LASSO, and Robust Least Squares) on nonlinear phase retrieval problems with structured noise.


Main Scripts

1. main_phase_retrieval.m
   • Compares robustness across multiple noise levels (δ_base = 10⁻¹, 10⁻², 10⁻³)
   • Uses obtuse noise matrix for structured noise
   • Runs 10 trials per configuration
   • Generates comparative plots for Δ_ls and Δ_lasso across different δ values

Core Functions

3. worst_case_error_nonlinear.m
   • Computes worst-case error E_{λ,Ĉ}(x) using vertex enumeration
   • Handles the inner maximization problem for robust optimization
   • Supports hypercube vertex enumeration for noise vectors

4. phi.m (Helper function)
   • Solves the inner maximization problem 
   • Calculates objective values for all possible noise vector configurations

5. F.m (Nonlinear mapping)
   • Defines the nonlinear function F(x) = (A·x)² - b
   • Can be modified for different nonlinearities (e.g., sin, absolute value)

Dependencies

• MATLAB Optimization Toolbox (for fminunc)

• Standard MATLAB libraries
