# Automated-ECG-Report

A Python tool that loads 12-lead ECG recordings, classifies them with a 1D convolutional neural network, and generates a readable diagnostic report with patient metadata. The model is trained on the PTB-XL dataset (Wagner et al., 2022).

## Model

- **Architecture:** 3-block 1D CNN (Conv1d → BatchNorm → ReLU → pooling), ~47k parameters
- **Input:** 12-lead, 10-second ECG sampled at 500 Hz (shape `12 × 5000`)
- **Output:** multi-label prediction over 71 SCP-ECG statements (diagnostic, form, and rhythm), thresholded at 0.5
- **Training:** `BCEWithLogitsLoss`, Adam (lr = 1e-3), batch size 64, 50 epochs

## Data Format

The tool reads PTB-XL records in WFDB format (`.dat` / `.hea` pairs, PhysioNet-compatible).

## Usage

1. Download PTB-XL (v1.0.3) from PhysioNet.
2. Set `base_path` in `new_main.py` to the dataset folder.
3. Run:

```bash
python new_main.py
```

## Output Details

Each run prints:

- Micro-F1 score of the model
- Patient info (if available): age, height (cm), weight (kg), recording date
- ECG diagnostic report: each predicted SCP code with its description (e.g., `NORM: normal ECG`). Sinus rhythm (`SR`) is excluded from the report.

## Example Output

```
F1 Score: 0.7364605946201038

Patient Info:
 - Age: 74.0
 - Height: nan
 - Weight: nan
 - Recording Date: 1986-03-05 09:04:12

ECG Diagnostic Report:
 - LAFB: left anterior fascicular block
```

## Evaluation and Known Limitations

- The reported micro-F1 (0.736) is computed on the **full dataset, including training records**, so it overstates performance on unseen data.
- Micro-F1 is weighted toward frequent labels (e.g., `SR`, `NORM`), so it does not reflect performance on rare conditions.

## Planned Improvements

- Retrain using PTB-XL's recommended stratified folds (folds 1–8 train, 9 validation, 10 test) and report held-out test micro-F1 and macro-AUROC
- Per-lead signal normalization
- Class-weighted loss to address label imbalance

## Citation

Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. (2022). PTB-XL, a large publicly available electrocardiography dataset (version 1.0.3). PhysioNet. https://doi.org/10.13026/kfzx-aw45