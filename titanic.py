from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import itertools
import pickle


def pre_processing_data():
    train_df = pd.read_csv("./train.csv")
    test_df = pd.read_csv("./test.csv")

    train_x = train_df.drop(["Survived"], axis=1)
    train_y = train_df["Survived"]
    test_x = test_df.copy()

    train_x = train_x.drop(["PassengerId"], axis=1)
    test_x = test_x.drop(["PassengerId"], axis=1)

    train_x = train_x.drop(["Name", "Ticket", "Cabin"], axis=1)
    test_x = test_x.drop(["Name", "Ticket", "Cabin"], axis=1)

    for c in ["Sex", "Embarked"]:
        le = LabelEncoder()
        le.fit(train_x[c].fillna("NA"))

        train_x[c] = le.transform(train_x[c].fillna("NA"))
        test_x[c] = le.transform(test_x[c].fillna("NA"))

    return train_x, train_y, test_x


def train(train_x: pd.DataFrame, train_y: pd.DataFrame):
    param_space = {
        "max_depth": [1, 3, 5, 7],
        "min_child_weight": [1.0, 2.0, 4.0, 8.0],
    }

    param_combinations = itertools.product(param_space["max_depth"], param_space["min_child_weight"])

    params = []
    scores = []

    for max_depth, min_child_weight in param_combinations:
        score_logloss = []
        score_accuracy = []

        kf = KFold(4, shuffle=True, random_state=123456)
        model = XGBClassifier(n_estimators=20, random_state=71, max_depth=max_depth, min_child_weight=min_child_weight)
        for tr_i, vr_i in kf.split(train_x):
            tr_x, vr_x = train_x.iloc[tr_i], train_x.iloc[vr_i]
            tr_y, vr_y = train_y.iloc[tr_i], train_y.iloc[vr_i]

            model.fit(tr_x, tr_y)

            vr_pred = model.predict_proba(vr_x)[:, 1]

            logloss = log_loss(vr_y, vr_pred)
            accuracy = accuracy_score(vr_y, vr_pred > 0.5)
            score_logloss.append(logloss)
            score_accuracy.append(accuracy)

        with open(f"./model/xgbclassfiler_{max_depth}_{min_child_weight}.pickle", mode="wb") as f:
            pickle.dump(model, f)

        score_mean = np.mean(score_logloss)
        accuracy_mean = np.mean(score_accuracy)
        params.append((max_depth, min_child_weight))
        scores.append(score_mean)
        print(f"{max_depth=}, {min_child_weight=}, {score_mean=}, {accuracy_mean=}")

    # logloss は値が低いほどよい
    best_i = np.argsort(scores)[0]
    best_param = params[best_i]
    print(f"max_depth: {best_param[0]}, min_child_weight: {best_param[1]}, score: {scores[best_i]}")


def test(test_x: pd.DataFrame):
    best_max_depth = 3
    best_min_chidl_weight = 4.0

    with open(f"./model/xgbclassfiler_{best_max_depth}_{best_min_chidl_weight}.pickle", mode="rb") as f:
        model = pickle.load(f)
        Y = model.predict(test_x)

    predict_result = pd.Series(Y, name="Survived")

    test_df = pd.read_csv("./test.csv")
    test_passenger_id = test_df["PassengerId"]
    result_df = pd.concat([test_passenger_id, predict_result], axis=1)
    result_df.to_csv("./result_submission.csv", index=False)
    return


if __name__ == "__main__":
    train_x, train_y, test_x = pre_processing_data()
    train(train_x, train_y)
    # test(test_x)
