# Cherry Blossom Peak Bloom Prediction
* Submission to George Mason’s Department of Statistics cherry blossom peak bloom prediction competition
* Authors: Julia Jeng, Fang Li, Julia Hsu
* Date: Feb. 28th, 2022

## Introduction
In this analysis, we demonstrate four methods of predicting the peak bloom data in the coming decade for all four locations required by the competition. The cherry trees' blossom development is dependent on weather conditions in winter or spring and species of growing degree (citation). Especially, the trees will bloom highly affects by their growing degree days(GDD), which is a measurement based on the temperature degrees of the area where it is located and the certain threshold base temperature.  Therefore, we studied the GDD calculations of the different cherry trees species listed in each location. 

## Dataset
1. [Peak Bloom Date dataset](https://github.com/GMU-CherryBlossomCompetition/peak-bloom-prediction/tree/main/data) provided by George Mason’s Department of Statistics cherry blossom peak bloom prediction competition.
2. [Temperature Data between 1979 and 2022]()

| File | Description |
| ---- | ----------- |
| Timeseeries_weather_* | Temperature data of 4 locations |
| other | Visit [here](https://github.com/GMU-CherryBlossomCompetition/peak-bloom-prediction/tree/main/data) for more information about the dataset provided by the competetion |

## Folder Structure
| Folder | Description |
| ------ | ----------- |
| **src** |  The src folder contains source code for the project. |
| **data**|The data folder contains orginal dataset and processed data used for the project. |
| [data/processed_data/features/](https://github.com/JuliaHsu/peak-bloom-prediction/tree/main/data/processed_data/features) | Extracted features for ML model training |

## Source Code
---

Source code are located in the /src folder. The key files are:

### Multiple Regression

| File | Description |
| ---- | ----------- |

### Machine Learning Models

| File | Description |
| ---- | ----------- |
| [ML/feature_extraction.ipynb](https://github.com/JuliaHsu/peak-bloom-prediction/blob/main/src/ML/feature_extraction.ipynb) | Feature extraction codes for ML model training |
| [ML/ML_pred.ipynb](https://github.com/JuliaHsu/peak-bloom-prediction/blob/main/src/ML/ML_pred.ipynb) | Codes for ML model training and prediction |

### Evaluation
|                           | Mean absolute error | Mean squared error | 
| --------------------------| --------------------| -------------------|
| Multiple Regression       |
| Support Vector Regression |

### Predictions
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








