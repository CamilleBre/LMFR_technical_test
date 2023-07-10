# LMFR_technical_test

This repository contains the technical test for LMFR Senior Data Scientist position.

## Install 
- Install python 3.10
- Clone the repository
- Create a virtual environment and make the requirements installation
  ```
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```

## EDA 
For exploratory data analysis, notebooks are stored in the notebooks section of the repo. 

## Model training
To train a model run the main.py script
Example:
```
python3 main.py --model_name 'LR' --metric 'recall' --scaler 'standard'
```
This code will train a logistic regression model, based on recall metric performance and including a standard scaling preprocessing step. 

## Results 
The performance of the trained models are stored as .csv files in the results section.

## Models
Models with best performances are stored in .pkl format in the models section.

## Report 
The file LMFR_report.pdf is a synthesis and presentation of the work carried out and the results obtained for this technical test. 
