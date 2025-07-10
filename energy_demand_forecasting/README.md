# Energy Demand Forecasting for Nuclear-Powered Data Centers

## Overview
This ML model forecasts energy demand for AI data centers powered by nuclear energy, using an LSTM neural network to predict hourly consumption. It supports grid stability and Malaysia’s NETR goals.

## Installation
1. Clone the repository: `git clone <repo_url>`
2. Navigate to the directory: `cd energy_demand_forecasting`
3. Install dependencies: `pip install -r requirements.txt`

## Usage
1. Run the script: `python energy_demand_forecasting.py`
2. The script trains the LSTM model on simulated data, evaluates performance, and saves the model as `energy_demand_model.h5`.
3. Outputs a plot (`demand_forecast.png`) comparing actual vs. predicted demand.

## Data
- Simulated hourly energy demand for one year with seasonal patterns.
- Adaptable to real data center consumption data.

## Notes
- Designed for non-critical grid integration applications.
- Developed by EGK (Isaac Khor Eng Gian), leveraging ML and xAI expertise.

## License
MIT License
