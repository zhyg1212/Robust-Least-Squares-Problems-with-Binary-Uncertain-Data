Project Overview

This project compares the performance of four different classification models on health monitoring data, with a special focus on model robustness to label noise. The project uses real health monitoring and insole sensor data, evaluating model robustness by introducing varying levels of label noise.

File Structure

Main Code Files

1. final_accuracy_avgdata.m - Main execution script
   • Loads and processes multi-user health data

   • Implements training and evaluation of four classification models

   • Generates accuracy comparison charts

2. load_and_merge_user_data.m - Data loading function
   • Reads and merges health data with insole sensor data

   • Performs data alignment based on timestamps

   • Includes error handling and data validation

Data Files

• data/ - Data folder (needs to be created in the same directory)

Required Toolboxes

• MATLAB Statistics and Machine Learning Toolbox

• CVX Optimization Toolbox 