from sklearn.svm import LinearSVR
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import pearsonr
import pandas as pd

def read_files(train,test):
    train = pd.read_csv(train, header=None)
    test = pd.read_csv(test, header=None)

    train_y = train[1]
    train_x = train.iloc[:,2:]
    test_y = test[1]
    test_x = test.iloc[:,2:]

    return train_x, train_y, test_x, test_y

train_file = "EI-reg-En-anger-train.vectors.without.random.train.csv"
test_file = "EI-reg-En-anger-train.vectors.without.random.test.csv"

train_x, train_y, test_x, test_y = read_files(train_file, test_file)

regr = LinearSVR(random_state=0)

regr.fit(train_x,train_y)
preds = regr.predict(test_x)

print(pearsonr(preds, test_y))
