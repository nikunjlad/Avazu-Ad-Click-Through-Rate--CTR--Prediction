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
y = np.load("data/labels.npy")

print(X.shape)
print(y.shape)


def SMOTE_Analysis(k, o, u):
    try:
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
        return [k,o,u]
    except Exception as e:
        return ""


if __name__ == "__main__":
    k_values = [1, 2, 3, 4, 5, 6, 7]
    over_fact = [0.1, 0.3, 0.4]
    under_fact = [0.5, 0.6, 0.7, 0.8]
    vals = [[k, o, u] for k in k_values for o in over_fact for u in under_fact]

    with mp.Pool(processes=50) as pool:
        results = pool.starmap(SMOTE_Analysis, vals)

    print(results)
