# NLP-Final-Project

This is the final project of group 7, a customer support chatbot trained on Amazon customer support question-answer data.
#### Group 7: Jakub Ondrejka, Matthijs Jongbloed, Michal Tesnar & Abi Raveenthiran

## Installation 
Download and unzip the main branch of this repository.\
Then, we recommend to use a virtual environment to install the required packages.\
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
## Downloading data
We have provided a small sample (100) of the dataset for you to train on.\
If you would like to train on the whole dataset, you can download it from [here](https://www.kaggle.com/datasets/praneshmukhopadhyay/amazon-questionanswer-dataset?select=single_qna.csv).\

## Usage
To reproduce our results, you can run the 'main.py' file as follows:
```python
python main.py
```
This script runs several functions defined in the other scripts of the repository.
