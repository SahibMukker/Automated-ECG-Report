import os
import ast
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
import wfdb
from ecg_model_architecture import ECGClassifier

#---------Load and preprocess metadata---------------
print('\n--- Loading and Preprocessing Metadata ---\n')
base_path = 'C:/Users/Sahib Mukker/Documents/Coding/Projects/7. Automated ECG Report/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
df = pd.read_csv(os.path.join(base_path, 'ptbxl_database.csv'))

# Convert label strings to dictionaries
df['scp_codes'] = df['scp_codes'].apply(lambda x: ast.literal_eval(x))

# Extract diagnostic labels
df['diagnostic_codes'] = df['scp_codes'].apply(lambda x: [k for k in x.keys()])

# Multi-label binarization
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(df['diagnostic_codes'])

#---------Load ECG signals---------------
print('\n--- Loading ECG Signals ---\n')
def load_ecg_signal_and_check_fs(record_path, base_path, desired_fs=500):
    '''
    Load an ECG signal and check if its sampling frequency matches the desired sampling frequency.
    '''
    full_path = os.path.join(base_path, record_path)
    record = wfdb.rdrecord(full_path)
    if record.fs == desired_fs:
        return record.p_signal.T  # Shape: (12, 5000)
    else:
        return None

signals = []
valid_labels = []

for idx, row in df.iterrows():
    path = row['filename_hr']
    try:
        ecg = load_ecg_signal_and_check_fs(path, base_path)
        if ecg is not None:
            signals.append(ecg)
            valid_labels.append(labels[idx])
    except Exception as e:
        print(f"Failed to load {path}: {e}")

signals = np.array(signals)
labels = np.array(valid_labels)

#---------ECG Dataset---------------
print('\n--- Creating ECG Dataset ---\n')
class ECGDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = torch.tensor(signals, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]

#---------Load Model---------------
print('\n--- Loading Model ---\n')
model = ECGClassifier(num_classes=labels.shape[1])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('ecg_classifier.pth', map_location=device))
model.to(device)

#---------Evaluate---------------
print('\n--- Evaluating Model ---\n')
full_dataset = ECGDataset(signals, labels)
full_dataloader = DataLoader(full_dataset, batch_size=64, shuffle=False)

model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for inputs, targets in full_dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = (outputs.cpu() > 0.5).to(torch.int)
        all_preds.append(preds)
        all_targets.extend(targets.numpy())

all_preds = np.vstack(all_preds)
all_targets = np.vstack(all_targets)

print('F1 Score:', f1_score(all_targets, all_preds, average='micro'))

#---------Generate Report for an Example Prediction---------------
print('\n--- Generating Report for an Example Prediction ---\n')
# Example: Choose the first predicted label row
example_index = 7
example_prediction = all_preds[example_index]
predicted_labels = [mlb.classes_[i] for i in range(len(example_prediction)) if example_prediction[i] == 1]

# Example patient info from the DataFrame
patient_info = {
    'Age': df.iloc[example_index]['age'],
    'Sex': df.iloc[example_index]['sex'],
    'Height': df.iloc[example_index]['height'],
    'Weight': df.iloc[example_index]['weight'],
    'Recording Date': df.iloc[example_index]['recording_date'],
}
def loading_diagnostic_map(csv_path='scp_statements.csv'):
    '''
    Load diagnostic labels from SCP Statements

    Returns: 
        Dictionary: contains the code (ex. 'NORM') and the associated disease description
    '''
    df = pd.read_csv(csv_path)
    df = df[df['diagnostic'] == 1]
    diag_map = pd.Series(df['description'].values, index=df['scp_code']).to_dict()
    return diag_map

def generating_ecg_report(predicted_labels, patient_info=None, diag_map=None):
    '''
    Creates a report based on model predictions and patient info if available

    Args:
        predicted_labels (list): List of predicted diagnostic codes (e.g., ['NORM'], ['MI']) 
        patient_info (dictionary): Dictionary of patient metadata
        diag_map (dictionary): Dictionary map of label -> description

    Returns:
        str: Formatted textual report
    '''
    if diag_map is None:
        diag_map = loading_diagnostic_map()

    report = ''

    if patient_info:
        report += 'Patient Info:\n'
        for k, v in patient_info.items():
            report += f' - {k}: {v}\n'
        report += '\n'

    report += 'ECG Diagnostic Report:\n'

    if not predicted_labels:
        report += ' - No diagnostic abnormalities detected.\n'
    else:
        for label in predicted_labels:
            description = diag_map.get(label, 'No description available.')
            report += f' - {label}: {description}\n'

    return report

# Load diagnostic map
diag_map = loading_diagnostic_map(csv_path=os.path.join(base_path, 'scp_statements.csv'))

# Generate report
report = generating_ecg_report(predicted_labels, patient_info=patient_info, diag_map=diag_map)

print('\n--- Generated ECG Report ---\n')
print(report)