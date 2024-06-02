import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from pprint import pprint
import pandas as pd
from itertools import combinations

features = ["PassengerId", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
object_value_name = "Survived"


def load_data(path: str):
    df = pd.read_csv(path)
    # filter
    df = df[[*features, object_value_name]]
    # drop if exists na data
    df = df.dropna()
    # mapping
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"C": 0, "Q": 1, "S": 2})
    return df


def pair_plot(df: pd.DataFrame):
    pair_plot = sns.pairplot(
        df,
        hue="Survived",
        plot_kws={"alpha": 0.2},
    )
    # plt.show()
    # ペアプロット
    pair_plot.savefig("pairplot.png")

    # 2要素を散布図を書く
    feture_combinations = list(combinations(df.columns, 2))
    for combo in feture_combinations:
        print(combo)
        joint_plot = sns.jointplot(data=df, x=combo[0], y=combo[1], hue=object_value_name)
        joint_plot.savefig(f"{combo[0]}_{combo[1]}.png")


def main():
    df = load_data("./train.csv")
    # 使えそうな特徴量をさがす
    pair_plot(df)

    X = df[features]
    Y = df[object_value_name]
    print(X.info())
    print(Y.info())

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    lr = LogisticRegression()
    result = lr.fit(X_train, Y_train)  # ロジスティック回帰モデルの重みを学習
    pprint(result.get_params())

    Y_pred = lr.predict(X_test)

    print("confusion matrix = \n", confusion_matrix(y_true=Y_test, y_pred=Y_pred))
    print("accuracy = ", accuracy_score(y_true=Y_test, y_pred=Y_pred))
    print("precision = ", precision_score(y_true=Y_test, y_pred=Y_pred))
    print("recall = ", recall_score(y_true=Y_test, y_pred=Y_pred))
    print("f1 score = ", f1_score(y_true=Y_test, y_pred=Y_pred))

    Y_score = lr.predict_proba(X_test)[:, 1]  # 検証データがクラス1に属する確率
    print("auc = ", roc_auc_score(y_true=Y_test, y_score=Y_score))


if __name__ == "__main__":
    main()
