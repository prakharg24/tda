*Note : The repo is not maintained actively. However, we encourage you to contact us if you find any part of the code outdated or any other errors. Additionally, we recommend to train the models on your own machine and avoid using pre-trained models if possible.*

# Requirements
Please check [requirements.txt](requirements.txt) for all major library requirements.

# Preparing Dataset
## Downloading the datset
1. Download the dataset file from Google Drive [here](https://drive.google.com/file/d/11OYtPBZoj8naV7snSTgvbuSG2VimzQYw/view?usp=sharing)
2. Place the zip file inside the main folder and extract.

## Dataset format
1. Two different sets of dataset, for PPCS and AGC systems.
2. 4 different dataset files for each system, namely `Trainset_classification`, `Trainset_regression`, `Testset_classification` and `Testset_regression`.

# Architecture Definition
## File Structure
1. `dataloader.py` contains code to read csv and convert data to required format. Also contains data augmentation for training.
2. `hlstm_model.py` contains the model definition
3. `utils.py` contains various utility function, eg. implementation of various output interpretation strategies
4. `train.py` contains the code to perform training in two phases. Final complete model is saved.
5. `evaluate.py` contains the code to perform the final evaluation.

# Usage
## Model Training

```
usage: train.py [-h] [--reg_csv REG_CSV] [--cls_csv CLS_CSV]
                [--model_prefix MODEL_PREFIX] [--lower_step LOWER_STEP]
                [--sensor_channels SENSOR_CHANNELS]
                [--window_length WINDOW_LENGTH]
                [--start_overhead START_OVERHEAD]
                [--sliding_step SLIDING_STEP] [--upper_depth UPPER_DEPTH]
                [--lower_depth LOWER_DEPTH]
                [--dense_hidden_units DENSE_HIDDEN_UNITS]
                [--upper_lstm_units UPPER_LSTM_UNITS]
                [--lower_lstm_units LOWER_LSTM_UNITS] [--dropout DROPOUT]
                [--epoch_regression EPOCH_REGRESSION]
                [--epoch_classification EPOCH_CLASSIFICATION]
                [--batch_size BATCH_SIZE]
```

# Testing Existing Models
## Downloading pre-trained models
1. Download the model files for both PPCS and AGC from Google Drive [here](https://drive.google.com/drive/folders/1mNZFCNzrFWOhWykfz5pOi52HE7bjlHrg?usp=sharing)

## Performance Evaluation

```
usage: evaluate.py [-h] [--reg_csv REG_CSV] [--cls_csv CLS_CSV]
                   [--model_prefix MODEL_PREFIX] [--lower_step LOWER_STEP]
                   [--sensor_channels SENSOR_CHANNELS]
                   [--window_length WINDOW_LENGTH]
                   [--start_overhead START_OVERHEAD]
                   [--cls_strategy CLS_STRATEGY] [--reg_strategy REG_STRATEGY]
                   [--reg_strategy_param REG_STRATEGY_PARAM]
```
