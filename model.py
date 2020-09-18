from sklearn.linear_model import LogisticRegressionCV
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD


def optimize_hyperparameters(X_train_data, X_test_data, y_train_data,
                             model, param_grid, n_jobs, cv=10, scoring_fit='neg_mean_squared_error'):
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        n_jobs=n_jobs,
        scoring=scoring_fit,
        verbose=2
    )

    model = gs.fit(X_train_data, y_train_data)
    preds = model.predict(X_test_data)

    return model, preds


def logistic(X_train, y_train, X_test):
    logit_model = sm.Logit(y_train, X_train)
    result = logit_model.fit()
    print("Logistic regression model summary: \n{}".format(result.summary2()))

    model = LogisticRegressionCV(cv=5)
    model.fit(X_train, y_train)
    yhat = model.predict(X_test)
    probs = model.predict_proba(X_test)

    return yhat, probs


def xgb(X_train, y_train, X_test):
    model = XGBClassifier()
    model.fit(X_train, y_train)
    yhat = model.predict(X_test)
    probs = model.predict_proba(X_test)
    return yhat, probs


def mlp(X_train, y_train, X_test):
    def create_model():
        # create model
        model = Sequential()
        model.add(Dense(10, input_dim=15, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        # Compile model
        optimizer = SGD(learning_rate=0.1, momentum=0.9)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', ''])
        return model

    model = KerasClassifier(build_fn=create_model, verbose=0)

    params_grid = {
        "batch_size": [64, 128, 256],
        "epochs": [10, 50, 100],
        "dropout_rate": [0.1, 0.2]
    }

    model, preds, probs = optimize_hyperparameters(X_train, X_test, y_train, model, params_grid, n_jobs=-1, cv=5,
                                                   scoring_fit='neg_log_loss')

    print("Best: %f using %s" % (model.best_score_, model.best_params_))

    return model, preds, probs
