import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report, roc_auc_score, roc_curve, \
    confusion_matrix, auc
from imblearn.pipeline import Pipeline
import multiprocessing as mp
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint
from model import logistic, xgb, mlp
warnings.filterwarnings("ignore")

# load data
X_train = np.load("data/train.npy")
y_train = np.load("data/train_labels.npy")
X_test = np.load("data/test.npy")
y_test = np.load("data/test_labels.npy")

# print data shape
print("Training data shape: {}".format(X_train.shape))
print("Training labels shape: {}".format(y_train.shape))
print("Testing data shape: {}".format(X_test.shape))
print("Testing labels shape: {}".format(y_test.shape))


yhat, probs = mlp(X_train, y_train, X_test)

# accuracy

# confidence interval

# classification report

# confusion matrix

# roc_auc

# fpr, tpr