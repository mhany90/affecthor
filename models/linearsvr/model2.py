from sklearn.svm import LinearSVR
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr
import pandas as pd

def read_files(train,test):
    train = pd.read_table(train, header=None)
    test = pd.read_table(test, header=None)

    train_y = train[3]
    train_x = train[1]
    test_y = test[3]
    test_x = test[1]

    return train_x, train_y, test_x, test_y

train_file = "../../data/EI-reg-English-Train/EI-reg-en_fear_train.txt"
test_file = "../../data/dev/EI-reg-En-fear-dev.txt"

train_x, train_y, test_x, test_y = read_files(train_file, test_file)
print(len(train_x))
print(len(train_x.unique()))

regr = Pipeline([('vect', TfidfVectorizer(analyzer="word", ngram_range=(1,2), binary=True)),
                 ('clf', LinearSVR())
                 ])
#regr = LinearSVR(random_state=0)

regr.fit(train_x,train_y)
preds = regr.predict(test_x)

print(pearsonr(preds, test_y))
