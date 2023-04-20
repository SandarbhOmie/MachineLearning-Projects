import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # first we will divide data into training data and testing data
from sklearn.feature_extraction.text import TfidfVectorizer  # 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
raw_mail_data = pd.read_csv(r'D:\PYTHON\projects\Machine learning\Spam-Emails Detector\spam.csv',encoding="ISO-8859-1")
print(raw_mail_data)