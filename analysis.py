from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import multiprocessing as mp
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
import warnings

warnings.filterwarnings("ignore")

X = np.load("data/train.npy")
y = np.load("data/train_labels.npy")

print(X.shape)
print(y.shape)


def find_number_of_uniques(df, feature):
    print("Number of unique {} values: {}".format(feature, df[feature].nunique()))


def SMOTE_Analysis(k, o, u):
    try:
        print("k: {}, over: {}, under: {}".format(k, o, u))
        model = DecisionTreeClassifier()
        over = SMOTE(sampling_strategy=o, k_neighbors=k, random_state=2)
        under = RandomUnderSampler(sampling_strategy=u)
        steps = [('over', over), ('under', under)]
        pipeline = Pipeline(steps=steps)
        Xn, yn = pipeline.fit_resample(X, y.ravel())
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        scores = cross_val_score(model, Xn, yn, scoring='roc_auc', cv=cv, n_jobs=-1)
        score = np.mean(scores)
        print("k={}, over={}, under={}, Mean ROC AUC: {:.3f}".format(k, o, u, score))
        return [k, o, u]
    except Exception as e:
        print(e)
        return None


def count_unique(d, columns):
    for column in columns:
        print("Number of Unique values in column {} is {}".format(column, str(len(d[column].unique()))))


if __name__ == "__main__":
    k_values = [1, 2, 3, 4, 5, 6, 7]
    over_fact = [0.2, 0.4, 0.5]
    under_fact = [0.5, 0.6, 0.7]
    vals = [[k, o, u] for k in k_values for o in over_fact for u in under_fact]

    with mp.Pool(processes=10) as pool:
        results = pool.starmap(SMOTE_Analysis, vals)

    print(results)
