import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    log_loss,
)
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
from optuna.pruners import MedianPruner
from optuna.trial import TrialState
import pickle


object_value_name = "target"


def load_data():
    train_df = pd.read_csv("./train.csv")
    test_df = pd.read_csv("./test.csv")

    train_df = train_df.drop(["ID_code"], axis=1)
    return train_df, test_df


def pre_process_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    # 標準化
    scaler = StandardScaler()
    # 標準化したいパラメータを限定
    standard_scale_col = train_df.columns.tolist()
    standard_scale_col.remove(object_value_name)
    # 学習データに合わせてスケーリングパラメータを決定
    scaler.fit(train_df[standard_scale_col])

    standard_train = scaler.transform(train_df[standard_scale_col])
    standard_test = scaler.transform(test_df[standard_scale_col])

    # 主成分分析
    pca_col_num = 5
    pca = PCA(n_components=pca_col_num)
    # 学習データによる主成分分析を定義
    pca_model = pca.fit(standard_train)
    with open("pca.pkl", mode="wb") as f:
        pickle.dump(pca_model, f)

    # 学習、テストデータに主成分分析を適用
    trans_train_df = pd.DataFrame(pca.transform(standard_train), columns=[f"pca{i+i}" for i in range(pca_col_num)])
    trans_test_df = pd.DataFrame(pca.transform(standard_test), columns=[f"pca{i+i}" for i in range(pca_col_num)])
    print("PCA fit completed.")

    trans_train_df[object_value_name] = train_df[object_value_name]
    # pca_graph_1 = sns.jointplot(data=trans_train_df, x="pca1", y="pca2", kind="hex")
    # pca_graph_1.savefig("./pca_result.png")
    # pca_graph_2 = sns.jointplot(data=trans_train_df, x="pca1", y="pca2", hue=train_df[object_value_name])
    # pca_graph_2.savefig("./pca_category.png")
    # plt.close()

    # return train_df, test_df
    return trans_train_df, trans_test_df


def train(trial: optuna.Trial, train_df: pd.DataFrame):
    x = train_df.drop([object_value_name], axis=1)
    y = train_df[object_value_name]

    # 学習、検証データを分割する
    # x_train, x_eval, y_train, y_eval = train_test_split(x, y, test_size=0.2, random_state=0)

    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"])
    C = trial.suggest_float("C", 0.0001, 10, log=True)
    max_iter = trial.suggest_int("max_iter", 100, 100000, log=True)

    # 無効な組み合わせをチェック
    if solver == "liblinear" and penalty == "elasticnet":
        raise optuna.exceptions.TrialPruned()

    if penalty == "l1" and solver not in ["liblinear", "saga"]:
        raise optuna.exceptions.TrialPruned()

    """
    try:
        model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=max_iter)
    except ValueError:
        raise optuna.exceptions.TrialPruned()
    cross_entropy_mean = cross_val_score(model, x, y, cv=5, scoring="accuracy").mean()
    """
    # Cross validation
    auc_scores = []
    cross_entropies = []
    kf = KFold(n_splits=4, shuffle=True, random_state=71)
    for train_i, eval_i in kf.split(x):
        # 学習、検証データを分割する
        x_train, x_eval = x.iloc[train_i], x.iloc[eval_i]
        y_train, y_eval = y.iloc[train_i], y.iloc[eval_i]
        # 学習
        try:
            lr = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=max_iter)
        # ハイパーパラメータの組み合わせが不適当な場合はスキップ
        except Exception as e:
            raise optuna.exceptions.TrialPruned()
        lr.fit(x_train, y_train)

        # 検証データを予測
        y_pred = lr.predict(x_eval)

        # print("confusion matrix = \n", confusion_matrix(y_true=y_eval, y_pred=y_pred))
        # print("accuracy = ", accuracy_score(y_true=y_eval, y_pred=y_pred))
        # print("precision = ", precision_score(y_true=y_eval, y_pred=y_pred))
        # print("recall = ", recall_score(y_true=y_eval, y_pred=y_pred))
        # print("f1 score = ", f1_score(y_true=y_eval, y_pred=y_pred))

        y_score = lr.predict_proba(x_eval)[:, 1]  # 検証データがクラス1に属する確率

        # save ROC plot
        fpr, tpr, _ = roc_curve(y_true=y_eval, y_score=y_score, drop_intermediate=False)
        plt.plot(fpr, tpr)
        plt.savefig(f"roc_trial_{trial.number:02d}.png")
        # plt.show()
        plt.close()

        auc = roc_auc_score(y_true=y_eval, y_score=y_score)
        auc_scores.append(auc)

        cross_entropy = log_loss(y_true=y_eval, y_pred=y_pred)
        cross_entropies.append(cross_entropy)

    auc_score_mean = sum(auc_scores) / len(auc_scores)
    cross_entropy_mean = sum(cross_entropies) / len(cross_entropies)
    # print(f"auc score mean: {auc_score_mean}")

    return cross_entropy_mean


def train_all_data(train_df: pd.DataFrame, best_params: dict):
    x = train_df.drop([object_value_name], axis=1)
    y = train_df[object_value_name]

    lr = LogisticRegression(**best_params)
    model = lr.fit(x, y)
    with open("best_model.pkl", mode="wb") as f:
        pickle.dump(model, f)


def main():
    train_df, test_df = load_data()
    train_df, test_df = pre_process_data(train_df, test_df)

    study = optuna.create_study(
        direction="minimize", pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1)
    )
    study.optimize(lambda trial: train(trial, train_df), n_trials=500, show_progress_bar=True)

    # トライアルのステータスを表示
    pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

    print("Number of finished trials: ", len(study.trials))
    print("Number of pruned trials: ", len(pruned_trials))
    print("Number of complete trials: ", len(complete_trials))

    # train_all_data(train_df, study.best_params)


if __name__ == "__main__":
    main()
