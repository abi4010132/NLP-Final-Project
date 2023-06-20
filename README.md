# NLP-Final-Project

This is the final project of group 7, a chatbot trained on Amazon customer support question-answer data.
#### Group 7: Jakub Ondrejka, Matthijs Jongbloed, Michal Tesnar & Abi Raveenthiran

## Installation 
Download and unzip the main branch of this repository.\
Then, we recommend to use a virtual environment to install the required modules.\
If you don't want to use a virtual environment skip to [No virtual environment](#no-virtual-environment).

### Virtual environment
If you don't have pipenv installed yet, you can install it by running the following line in your terminal.
```python
pip install --user pipenv
```
Once you've successfully installed pipenv,  in the terminal, navigate to the folder you just unzipped.\
Now you can create a virtual environment by entering the following line in the terminal.
```python
pipenv shell
```
If you closed your terminal or virtual environment, navigate back to the folder again and start up the virtual environment using:
```python
pipenv shell
```

Now to install the required packages, enter the following line in the terminal:
```python
pipenv install -r requirements.txt
```

### No virtual environment
In the terminal, navigate to the folder you just created and enter the following line:
```python
pip install -r requirements.txt
```
### Manual installation
If installing from the requirements.txt did not work for you, you can also manually install the modules by entering the following line in your terminal. 
```python
pip install pandas numpy gensim keras keras_tuner contractions nltk
```
## Usage
### Downloading data
We have provided a small sample (100) of the cleaned dataset for you to train on, as the whole dataset is too big to upload on GitHub.\
You can download the whole dataset from [here](https://www.kaggle.com/datasets/praneshmukhopadhyay/amazon-questionanswer-dataset?select=single_qna.csv), if you would like to reproduce the data cleaning process.\
Make sure you store it in the data folder.
### Run whole process
To reproduce our results, you can run the 'main.py' file in your terminal as follows:
```python
python main.py
```
This script will perform the whole process from data cleaning to model evaluation.\
If this does not work for you, the process is also reproducable by following the individual steps listed below.
### Individual steps
#### Data cleaning
The data cleaning can be done by running the 'clean_data.py' file in your terminal as follows:
```python
python clean_data.py
```
**NOTE**: Don't forget to download the dataset from kaggle if you want to reproduce the cleaning process.   
#### Baseline model
The Baseline model can be run by running the 'baseline_model.py' file in your terminal as follows:
```
python baseline_model.py
```
Some short explanation of model / code implementation maybe?
#### Attention model
The Attention model can be run by running the 'baseline_model.py' file in your terminal as follows:
```
python attention_model.py
```
Some short explanation of model / code implementation maybe?
#### Evaluation
To evaluate the results of the model run the 'evaluation.py' file in your terminal as follows:
```
python evaluation.py
```
This will print the scores of the three evaluation metrics that we use: BLEU, NIST and METEOR.
## Improvements

