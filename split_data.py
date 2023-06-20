# imports
import pandas as pd

# this function splits the data into a training set and test set.
def split_data():
    data_all = pd.read_csv('single_qna_clean_data.csv')
    data_train = data_all.sample(frac=0.9,random_state=0)
    data_test = data_all.drop(data_train.index)
    data_train = data_train.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)
    data_train.to_csv('train.csv', index=False)
    data_test.to_csv('test.csv', index=False)

if __name__ == "__main__":
    split_data()
