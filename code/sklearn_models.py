import time

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

import utils

def load_data(standard=True, pca=False):
    train_data = utils.MNISTDataset()
    train_x, train_y = train_data.data, train_data.label
    train_x = train_x.reshape(-1, 28 * 28)
    train_data = utils.MNISTDataset(type="test")
    test_x2, test_y2 = train_data.data, train_data.label
    test_x2 = test_x2.reshape(-1, 28 * 28)

    test_x, test_y = utils.read_test_data()
    test_x = test_x.reshape(-1, 28 * 28)

    if standard:
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)
        test_x2 = scaler.transform(test_x2)

    if pca:
        pca = PCA(n_components=16)
        train_x = pca.fit_transform(train_x)
        test_x = pca.transform(test_x)
        test_x2 = pca.transform(test_x2)

    return train_x, train_y, test_x, test_y, test_x2, test_y2


def test_model(model, model_name, data):
    train_x, train_y, test_x, test_y, test_x2, test_y2 = data
    begin = time.time()
    model.fit(train_x, train_y)
    score1 = model.score(test_x2, test_y2)
    score2 = model.score(test_x, test_y)
    end = time.time()
    print(f"{model_name}，给定数据准确率：{100 * score1:.2f}%，额外数据准确率：{100 * score2: .0f}%，训练时间：{end - begin:.2f}s")


def main():
    data = load_data(standard=False, pca=True)
    bayes = GaussianNB()
    test_model(bayes, "朴素贝叶斯", data)

    data = load_data(standard=False, pca=False)
    forest = RandomForestClassifier(max_depth=10)
    test_model(forest, "随机森林", data)

    svc = SVC()
    test_model(svc, "支持向量机", data)

    knn = KNeighborsClassifier(3)
    test_model(knn, "KNN", data)

    lda = LinearDiscriminantAnalysis()
    test_model(lda, "线性判别分析", data)

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 64),
        solver="sgd",
        learning_rate_init=0.01,
        batch_size=64,
        alpha=1e-4,
    )
    test_model(mlp, "多层感知机", data)


if __name__ == "__main__":
    main()
