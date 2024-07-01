import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from pprint import pprint
import pandas as pd
from itertools import combinations
import os
import pickle

features = ["PassengerId", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Ticket"]
object_value_name = "Survived"


def _pre_process_cmn_data(df: pd.DataFrame) -> pd.DataFrame:
    """学習、テストデータに共通する前処理

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    # 欠損データに対する処理
    # df = df.dropna()
    df["Age"] = df["Age"].fillna(df["Age"].mean()).copy()
    df["Fare"] = df["Fare"].fillna(df["Fare"].mean()).copy()
    df["Embarked"] = df["Embarked"].fillna("S").copy()

    # mapping
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"C": 0, "Q": 1, "S": 2})

    # 料金0は除外
    df = df[df["Fare"] != 0].reset_index(drop=True)

    return df


def convert_categories(
    train_df: pd.DataFrame, test_df: pd.DataFrame, column_name: str
) -> list[pd.DataFrame, pd.DataFrame]:
    """カテゴリ変数を変換（学習データとテストデータで同じように変換する）

    Args:
        train_df (pd.DataFrame): _description_
        test_df (pd.DataFrame): _description_
        column_name (str): _description_

    Returns:
        list[pd.DataFrame, pd.DataFrame]: _description_
    """
    categories = train_df[column_name].unique()
    train_df[column_name] = pd.Categorical(train_df[column_name], categories)
    test_df[column_name] = pd.Categorical(test_df[column_name], categories)
    train_df = pd.get_dummies(train_df, columns=[column_name])
    test_df = pd.get_dummies(test_df, columns=[column_name])
    return train_df, test_df


def pre_process_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[pd.DataFrame, pd.DataFrame]:
    """データの前処理

    Args:
        train_df (pd.DataFrame): _description_
        test_df (pd.DataFrame): _description_

    Returns:
        list[pd.DataFrame, pd.DataFrame]: _description_
    """

    train_df = train_df[[*features, object_value_name]].copy()
    test_df = test_df[[*features]].copy()

    # 料金をクリッピング
    p01 = train_df["Fare"].quantile(0.01)
    p99 = train_df["Fare"].quantile(0.99)
    train_df["Fare"] = train_df["Fare"].clip(p01, p99)
    test_df["Fare"] = test_df["Fare"].clip(p01, p99)

    # 標準化
    scaler = StandardScaler()
    scaler.fit(train_df[["Age", "Fare"]])
    train_df[["Age", "Fare"]] = scaler.transform(train_df[["Age", "Fare"]]).copy()
    test_df[["Age", "Fare"]] = scaler.transform(test_df[["Age", "Fare"]]).copy()

    # ticket
    ticket_values = train_df["Ticket"].value_counts()
    ticket_values = ticket_values[ticket_values > 1]
    ticket_values = pd.Series(ticket_values.index, name="Ticket")
    categories = set(ticket_values.tolist())
    train_df["Ticket"] = pd.Categorical(train_df["Ticket"], categories=categories)
    test_df["Ticket"] = pd.Categorical(test_df["Ticket"], categories=categories)
    train_df = pd.get_dummies(train_df, columns=["Ticket"])
    test_df = pd.get_dummies(test_df, columns=["Ticket"])

    # SibSp/Parch
    train_df, test_df = convert_categories(train_df, test_df, "SibSp")
    train_df, test_df = convert_categories(train_df, test_df, "Parch")

    # 学習、テストデータの共通処理
    train_df = _pre_process_cmn_data(train_df)
    test_df = _pre_process_cmn_data(test_df)

    return train_df, test_df


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


def train(train_df: pd.DataFrame):
    # 使えそうな特徴量をさがす
    # pair_plot(df)

    # 質的データはダミー変数に変換
    # df = pd.get_dummies(df, drop_first=True)

    X = train_df.drop(object_value_name, axis=1)
    Y = train_df[object_value_name]

    # cross validation
    kf = KFold(n_splits=4, shuffle=True, random_state=57)

    for i, (tr_i, vr_i) in enumerate(kf.split(X)):
        X_train, X_valid = X.iloc[tr_i], X.iloc[vr_i]
        Y_train, Y_valid = Y.iloc[tr_i], Y.iloc[vr_i]

        lr = LogisticRegression(penalty="l1", solver="liblinear")
        result = lr.fit(X_train, Y_train)  # ロジスティック回帰モデルの重みを学習
        pprint(result.get_params())

        Y_pred = lr.predict(X_valid)

        print("confusion matrix = \n", confusion_matrix(y_true=Y_valid, y_pred=Y_pred))
        print("accuracy = ", accuracy_score(y_true=Y_valid, y_pred=Y_pred))
        print("precision = ", precision_score(y_true=Y_valid, y_pred=Y_pred))
        print("recall = ", recall_score(y_true=Y_valid, y_pred=Y_pred))
        print("f1 score = ", f1_score(y_true=Y_valid, y_pred=Y_pred))

        Y_score = lr.predict_proba(X_valid)[:, 1]  # 検証データがクラス1に属する確率
        print("auc = ", roc_auc_score(y_true=Y_valid, y_score=Y_score))

        with open(f"./model/model_{i}.pickle", mode="wb") as f:
            pickle.dump(lr, f)


def test(X: pd.DataFrame):
    with open("./model/model.pickle", mode="rb") as f:
        model = pickle.load(f)
        Y = model.predict(X)

    predict_result = pd.Series(Y, name=object_value_name)
    result_df = pd.concat([X["PassengerId"], predict_result], axis=1)
    result_df.to_csv("./result_submission.csv", index=False)


if __name__ == "__main__":
    train_df = pd.read_csv("./train.csv")
    test_df = pd.read_csv("./test.csv")
    post_process_train_df, post_process_test_df = pre_process_data(train_df, test_df)
    train(post_process_train_df)
    test(post_process_test_df)
