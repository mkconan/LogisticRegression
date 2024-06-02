import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from pprint import pprint

iris_df = sns.load_dataset("iris")  # データセットの読み込み
iris_df = iris_df[(iris_df["species"] == "versicolor") | (iris_df["species"] == "virginica")]  # 簡単のため、2品種に絞る
print(iris_df.info())

# sns.pairplot(iris_df, hue="species")
# plt.show()

X = iris_df[["petal_length", "sepal_length", "sepal_width", "petal_width"]]  # 説明変数
Y = iris_df["species"].map({"versicolor": 0, "virginica": 1})  # versicolorをクラス0, virginicaをクラス1とする
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0
)  # 80%のデータを学習データに、20%を検証データにする

lr = LogisticRegression()  # ロジスティック回帰モデルのインスタンスを作成
result = lr.fit(X_train, Y_train)  # ロジスティック回帰モデルの重みを学習
pprint(result.get_params())

print("coefficient = ", lr.coef_)
print("intercept = ", lr.intercept_)

Y_pred = lr.predict(X_test)
print(Y_pred)

print("confusion matrix = \n", confusion_matrix(y_true=Y_test, y_pred=Y_pred))
print("accuracy = ", accuracy_score(y_true=Y_test, y_pred=Y_pred))
print("precision = ", precision_score(y_true=Y_test, y_pred=Y_pred))
print("recall = ", recall_score(y_true=Y_test, y_pred=Y_pred))
print("f1 score = ", f1_score(y_true=Y_test, y_pred=Y_pred))

Y_score = lr.predict_proba(X_test)[:, 1]  # 検証データがクラス1に属する確率
print("auc = ", roc_auc_score(y_true=Y_test, y_score=Y_score))
