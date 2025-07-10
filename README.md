# Predictive Maintenance for Nuclear Plant Auxiliary Systems

## Overview
This ML model predicts equipment failures in non-critical nuclear power plant systems (e.g., cooling pumps) using sensor data (vibration, temperature, pressure). It uses a Random Forest Classifier to estimate failure probability, supporting predictive maintenance strategies.

## Installation
1. Clone the repository: `git clone <repo_url>`
2. Navigate to the directory: `cd predictive_maintenance`
3. Install dependencies: `pip install -r requirements.txt`

## Usage
1. Run the script: `python predictive_maintenance.py`
2. The script trains the model on simulated data, evaluates performance, and saves the model as `pump_failure_model.pkl`.
3. Modify input data in the script for real sensor data integration.

## Data
- Simulated dataset with 1000 samples, including vibration, temperature, pressure, and binary failure labels.
- Adaptable to real nuclear plant sensor data (non-critical systems only).

## Notes
- Intended for non-critical systems to comply with nuclear regulatory constraints.
- Developed by EGK (Isaac Khor Eng Gian), leveraging extensive ML expertise.

## License
MIT License
