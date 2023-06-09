import utils
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier


def load_data(standard=True, pca=False):
    train_data = utils.MNISTDataset(train=True)
    train_x, train_y = train_data.data, train_data.label
    train_x = train_x.reshape(-1, 28 * 28)
    train_data = utils.MNISTDataset(train=False)
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


def main():
    train_x, train_y, test_x, test_y, test_x2, test_y2 = load_data(
        standard=False, pca=True
    )
    bayes = GaussianNB()
    bayes.fit(train_x, train_y)
    print(bayes.score(test_x2, test_y2))
    print(bayes.score(test_x, test_y))

    train_x, train_y, test_x, test_y, test_x2, test_y2 = load_data(
        standard=False, pca=False
    )
    forest = RandomForestClassifier()
    forest.fit(train_x, train_y)
    print(forest.score(test_x2, test_y2))
    print(forest.score(test_x, test_y))

    svc = SVC()
    svc.fit(train_x, train_y)
    print(svc.score(test_x2, test_y2))
    print(svc.score(test_x, test_y))

    knn = KNeighborsClassifier()
    knn.fit(train_x, train_y)
    print(knn.score(test_x2, test_y2))
    print(knn.score(test_x, test_y))

    lda = LinearDiscriminantAnalysis()
    lda.fit(train_x, train_y)
    print(lda.score(test_x2, test_y2))
    print(lda.score(test_x, test_y))

    mlp = MLPClassifier(hidden_layer_sizes=(64, 64))
    mlp.fit(train_x, train_y)
    print(mlp.score(test_x2, test_y2))
    print(mlp.score(test_x, test_y))


if __name__ == "__main__":
    main()
