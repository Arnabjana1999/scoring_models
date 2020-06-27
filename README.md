
## This file contains instructions to generate data and train the scoring model.
Created by: Arnab Jana
June 25, 2020


The training data can be found in data.csv

However, if you want to regenerate the training data, you can change
the parameters in config.py and metadata.py to change the size and
distribution of data and then run the following command:

python generateData.py

Note: The data is overwritten into the file 'data.csv'

### For feature selection step, run the following command:

python feature_selectors.py <option>

where option can be (without quotes):
1. 'iv' - For Information Value based feature selection
2. 'anova' - For ANOVA based feature selection
3. 'mi' - For Mutual Information based feature selection

### For training the scoring model, run the following command:

python train.py <option>

where option can be (without quotes):
1. 'woe' - For Weight of Evidence model
2. 'linreg' - For regularized Linear Regression model
3. 'nn' - For Neural Network model

The output is displayed as 2 line graphs, the blue one corresponding to scores in the test-set and red one corresponding to model-predicted scores.