import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv('data/dataset.csv')

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test,
