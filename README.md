# delay_attack_protection
Code for our implementation of a learning model for protection against time delay attacks

## Dataset format
1. 4 different datasets `train_classification`, `train_regression`, `test_classification` and `test_regression`
2. Separate csv files for each dataset with the following columns -

| ind | delay | delay_st | gasflow.200 | gasflow.202 | ... | gasflow.1500 | pressure.200 | ... |
| --- | ----- | -------- | ----------- | ----------- | --- | ------------ | ------------ | --- |
