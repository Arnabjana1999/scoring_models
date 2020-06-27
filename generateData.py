import random
import numpy as np

from metadata import noOfFeatures, features
from config import DATA_SIZE, FILENAME
from config import TRAINING_REAL_RATIO, TRAINING_OUTLIER_RATIO
from config import VALIDATION_REAL_RATIO, VALIDATION_OUTLIER_RATIO
from config import TESTING_REAL_RATIO, TESTING_OUTLIER_RATIO

def generate_row_real():
    row='\n'
    score=0
    for i in range(noOfFeatures):
        selection=random.randint(0,4)
        row+=str(features[i][selection][0])+','
        score+=(features[i][selection][1]+np.random.normal(0,0.5,1)[0])
    row+=str(score)
    return row

def generate_row_outlier():
    row='\n'
    score=0
    for i in range(noOfFeatures):
        selection=random.randint(0,4)
        row+=str(features[i][selection][0])+','
        score+=(features[i][selection][1]+np.random.normal(random.randint(-50,50),10,1)[0])
    row+=str(score)
    return row

def main():

    with open(FILENAME,'w') as f:
        f.write('academics,sports,fitness,social,extra-currics,sleeping,time-on-social-media,time-on-outdoor,height')

        for _ in range(int(TRAINING_REAL_RATIO*DATA_SIZE)):
            f.write(generate_row_real())
        for _ in range(int(TRAINING_OUTLIER_RATIO*DATA_SIZE)):
            f.write(generate_row_outlier())
        for _ in range(int(VALIDATION_REAL_RATIO*DATA_SIZE)):
            f.write(generate_row_real())
        for _ in range(int(VALIDATION_OUTLIER_RATIO*DATA_SIZE)):
            f.write(generate_row_outlier())
        for _ in range(int(TESTING_REAL_RATIO*DATA_SIZE)):
            f.write(generate_row_real())
        for _ in range(int(TESTING_OUTLIER_RATIO*DATA_SIZE)):
            f.write(generate_row_outlier())

    print('Successfully generated data')



if __name__=='__main__':
    main()