from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.metrics import make_scorer
from numpy import mean
import os
import pandas as pd

def read_and_shuffle(f):
    df = pd.read_csv(f, header=None, skiprows=1)
    df = shuffle(df)
    x = df.iloc[:,2:]
    y = df[1]

    return x.iloc[:-200], y.iloc[:-200], x.iloc[-200:], y.iloc[-200:]

def read_and_shuffle_cv(f):
    df = pd.read_csv(f, header=None, skiprows=1)
    df = shuffle(df)
    x = df.iloc[:,2:]
    y = df[1]

    return x, y

def pearson(ty, py):
    return pearsonr(ty, py)[0]

DIR = '/home/s3094723/SEMEVAL/affecthor/data/EI-reg/En/train/'
files = [DIR+f for f in os.listdir(DIR) if "combined.csv" in f]

for f in files:
    print(f)
    train_x, train_y = read_and_shuffle_cv(f)
    regr = GradientBoostingRegressor(max_depth=3, max_leaf_nodes=30000)
    forest = RandomForestRegressor(max_depth=3)
    scorer = make_scorer(pearson)
    regr_cv = cross_val_score(regr, train_x, train_y, cv=10, scoring=scorer)
    forest_cv = cross_val_score(forest, train_x, train_y, cv=10, scoring=scorer)
    print("GRADIENT BOOSTING:\t", regr_cv)
    print("GRADIENT BOOSTING:\t", mean(regr_cv))
    print("RANDOM FOREST:\t", forest_cv)
    print("RANDOM FOREST:\t", mean(forest_cv))
