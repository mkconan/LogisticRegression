from judge_is_sruvived_in_titanic import load_data, object_value_name
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


def main():
    # df = load_data("./train.csv")
    df = pd.read_csv("./train.csv")
    # 使えそうな特徴量をさがす
    # pair_plot(df)

    # 質的データはダミー変数に変換
    # df = pd.get_dummies(df, drop_first=True)
    """
    X = df.drop(object_value_name, axis=1)
    for column in X.columns:
        print(column)
        plt.figure()
        df.plot.scatter(X)
        plt.savefig(f"{column}_scatter.png")
    """

    # sex
    encoder_sex = LabelEncoder()
    df["Sex"] = encoder_sex.fit_transform(df["Sex"].values)

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
    print(df)

    exit()
    # print(pd.factorize(df[analysis_data_name], sort=True)[1])

    # df[analysis_data_name] = pd.factorize(df[analysis_data_name], sort=True)[0]

    cross_age = pd.crosstab(df["Survived"], round(df[analysis_data_name], -1))
    cross_age.T.plot(kind="bar", stacked=False, width=0.8)
    plt.show()


if __name__ == "__main__":
    main()
