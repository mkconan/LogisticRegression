import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from pprint import pprint
import pandas as pd
from itertools import combinations
import os
import pickle

features = ["PassengerId", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Ticket"]
object_value_name = "Survived"


def load_data(path: str, is_train: bool = True):
    df = pd.read_csv(path)

    # filter
    if is_train:
        df = df[[*features, object_value_name]]
    else:  # for eval
        df = df[[*features]]

    # 欠損データに対する処理
    # df = df.dropna()
    df["Age"] = df["Age"].fillna(df["Age"].mean())
    df["Fare"] = df["Fare"].fillna(df["Fare"].mean())
    df["Embarked"] = df["Embarked"].fillna("S")

    # mapping
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"C": 0, "Q": 1, "S": 2})
    return df


def pre_process_data(df: pd.DataFrame):
    # 料金0は除外
    df = df[df["Fare"] != 0].reset_index(drop=True)

    # ticket
    ticket_values = df["Ticket"].value_counts()
    ticket_values = ticket_values[ticket_values > 1]
    ticket_values = pd.Series(ticket_values.index, name="Ticket")
    categories = set(ticket_values.tolist())
    df["Ticket"] = pd.Categorical(df["Ticket"], categories=categories)
    df = pd.get_dummies(df, columns=["Ticket"])

    # SibSp/Parch
    df = pd.get_dummies(df, columns=["SibSp"])
    df = pd.get_dummies(df, columns=["Parch"])

    return df


def pair_plot(df: pd.DataFrame):
    graph_folder = "graph"
    os.makedirs(graph_folder, exist_ok=True)
    pair_plot = sns.pairplot(
        df,
        hue="Survived",
        plot_kws={"alpha": 0.2},
    )
    # plt.show()
    # ペアプロット
    pair_plot.savefig(f"{graph_folder}/pairplot.png")
    plt.close()

    # 2要素を散布図を書く
    feture_combinations = list(combinations(df.columns, 2))
    for combo in feture_combinations:
        print(combo)
        joint_plot = sns.jointplot(data=df, x=combo[0], y=combo[1], hue=object_value_name)
        joint_plot.savefig(f"{graph_folder}/{combo[0]}_{combo[1]}.png")
        plt.close()


def train():
    df = load_data("./train.csv")
    df = pre_process_data(df)
    # 使えそうな特徴量をさがす
    # pair_plot(df)

    # 質的データはダミー変数に変換
    # df = pd.get_dummies(df, drop_first=True)

    X = df.drop(object_value_name, axis=1)
    Y = df[object_value_name]

    X_train, X_eval, Y_train, Y_eval = train_test_split(X, Y, test_size=0.2, random_state=0)

    lr = LogisticRegression(penalty="l1", solver="liblinear")
    result = lr.fit(X_train, Y_train)  # ロジスティック回帰モデルの重みを学習
    pprint(result.get_params())

    Y_pred = lr.predict(X_eval)

    print("confusion matrix = \n", confusion_matrix(y_true=Y_eval, y_pred=Y_pred))
    print("accuracy = ", accuracy_score(y_true=Y_eval, y_pred=Y_pred))
    print("precision = ", precision_score(y_true=Y_eval, y_pred=Y_pred))
    print("recall = ", recall_score(y_true=Y_eval, y_pred=Y_pred))
    print("f1 score = ", f1_score(y_true=Y_eval, y_pred=Y_pred))

    Y_score = lr.predict_proba(X_eval)[:, 1]  # 検証データがクラス1に属する確率
    print("auc = ", roc_auc_score(y_true=Y_eval, y_score=Y_score))

    with open("model.pickle", mode="wb") as f:
        pickle.dump(lr, f)


def test():
    X = load_data("./test.csv", is_train=False)
    print(X["PassengerId"])
    with open("./model.pickle", mode="rb") as f:
        model = pickle.load(f)
        Y = model.predict(X)

    predict_result = pd.Series(Y, name=object_value_name)
    result_df = pd.concat([X["PassengerId"], predict_result], axis=1)
    result_df.to_csv("./result_submission.csv", index=False)


if __name__ == "__main__":
    train()
    # test()
