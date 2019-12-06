# delay_attack_protection
Code for our implementation of a learning model for protection against time delay attacks

## Dataset format
1. 4 different datasets `train_classification`, `train_regression`, `test_classification` and `test_regression`
2. Separate csv files for each dataset with the following columns -

| ind | delay | delay_st | gasflow.200 | gasflow.202 | ... | gasflow.1500 | pressure.200 | ... |
| --- | ----- | -------- | ----------- | ----------- | --- | ------------ | ------------ | --- |

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
