# Automated-ECG-Report
This is an Automated ECG report generator. It is a Python based tool that loads, preprocesses, and analyzes ECG signals. Once analysis is complete, it generates a report that provides patient metadata and the potential heart conditions that patient may have. The ML model is a Convolutional Neural Network thats been trained on dataset created by Wagner et al. (2022). The model currently sits at an accuracy of 73.6%

## Data Format
The tool accepts the following data formats:
- .dat and .hea pairs (PhysioNet compatible)
- .csv

## Output Details
Each report consists of:

F1 Score of model

Patient Info (if provided):
- Age
- Height (cm)
- Weight (kg)
- Recording Date

ECG Diagnostic Report:
- SCP code (NORM, ASMI, etc.) and description of code (ex. NORM: normal ECG)
## Example Output
F1 Score: 0.7364605946201038

Patient Info:
 - Age: 74.0
 - Height: nan
 - Weight: nan
 - Recording Date: 1986-03-05 09:04:12

ECG Diagnostic Report:
 - LAFB: left anterior fascicular block
## Citations
Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. (2022). PTB-XL, a large publicly available electrocardiography dataset (version 1.0.3). PhysioNet. https://doi.org/10.13026/kfzx-aw45.
