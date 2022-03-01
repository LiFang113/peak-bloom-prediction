# Cherry Blossom Peak Bloom Prediction
* Submission to George Mason’s Department of Statistics cherry blossom peak bloom prediction competition
* Authors: Julia Jeng, Julia Hsu, Fang Li
* Date: Feb. 28th, 2022

## Introduction
In this analysis, we demonstrate four methods of predicting the peak bloom data in the coming decade for all four locations required by the competition. The cherry trees' blossom development is dependent on weather conditions in winter or spring and species of growing degree. Especially, the trees will bloom highly affects by their growing degree days(GDD), which is a measurement based on the temperature degrees of the area where it is located and the certain threshold base temperature.  Therefore, we studied the GDD calculations of the different cherry trees species listed in each location. 

## Data
1. [Peak Bloom Date dataset](https://github.com/GMU-CherryBlossomCompetition/peak-bloom-prediction/tree/main/data) provided by George Mason’s Department of Statistics cherry blossom peak bloom prediction competition.
2. [Historical temperature and rainfall observations](https://psl.noaa.gov/data/gridded/data.cpc.globalprecip.html) are extracted from CPC Global Unified Gauge-Based Analysis.

| File | Description |
| ---- | ----------- |
| [Timeseeries_weather_*](https://github.com/JuliaHsu/peak-bloom-prediction/tree/main/data/processed_data/) | Temperature data of 4 locations |
| other | Visit [here](https://github.com/GMU-CherryBlossomCompetition/peak-bloom-prediction/tree/main/data) for more information about the dataset provided by the competetion |

## Folder Structure
| Folder | Description |
| ------ | ----------- |
| **src** |  The src folder contains source code for the project. |
| **data**|The data folder contains orginal dataset and processed data used for the project. |
| [data/processed_data/](https://github.com/JuliaHsu/peak-bloom-prediction/tree/main/data/processed_data/) | Historical timeseries temperature dataset and processed bloom date dataset|
| [data/processed_data/features/](https://github.com/JuliaHsu/peak-bloom-prediction/tree/main/data/processed_data/features) | Extracted features for ML model training |

## Source Code
Source code are located in the /src folder. The key files are:

### Hierarchical Linear Regressions

| File | Description |
| ---- | ----------- |

### Machine Learning Models

| File | Description |
| ---- | ----------- |
| [ML/feature_extraction.ipynb](https://github.com/JuliaHsu/peak-bloom-prediction/blob/main/src/ML/feature_extraction.ipynb) | Feature extraction codes for ML model training |
| [ML/ML_pred.ipynb](https://github.com/JuliaHsu/peak-bloom-prediction/blob/main/src/ML/ML_pred.ipynb) | Codes for ML model training and PBD prediction |

### Deep Learning - LSTM 
| File | Description |
| ---- | ----------- |
| [deep_learning/LSTM_predict10years.py](https://github.com/JuliaHsu/peak-bloom-prediction/blob/main/src/deep_learning/LSTM_predict10years.py)| LSTM model for PBD predictions |

## Evaluation
|                                       | Mean absolute error | Mean squared error | 
| --------------------------------------| --------------------| -------------------|
| Hierarchical Linear Regressions       | 0.002               | 0.000004           |
| Support Vector Regression             | 3.628               | 24.094             |
| LSTM                                  | 10.739              | 16.416             |

## Predictions
| Year | Kyoto | Washington DC | Liestal | Vancouver |
| -----| ----- | ------------- | ------- | ----------|
| 2022 |
| 2023 |
| 2024 |
| 2025 |
| 2026 |
| 2027 |
| 2028 |
| 2029 |
| 2030 |
| 2031 |








