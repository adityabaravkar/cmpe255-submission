import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline
from sklearn import model_selection

class ImagePrediction:

    def __init__(self):
        self.faces = fetch_lfw_people(min_faces_per_person=60)
        print('data loaded')

    def prepare_X_y(self):
        X = self.faces.data
        y = self.faces.target
        return X, y

    def get_shape_data(self):
        return self.faces.images.shape

    def get_names(self):
        return self.faces.target_names

    def plot_gallery(self, images, titles, names_actual, h, w, fig_title):
        n_row=4
        n_col=6
        fig = plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
        for i in range(n_row * n_col):
            ax = fig.add_subplot(n_row, n_col, i + 1)
            ax.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
            fontColor = 'black'
            if titles[i]!=names_actual[i]:
                fontColor = 'red'
            title = "Predicted: "+titles[i]+"\nActual: "+names_actual[i]
            ax.set_title(titles[i], size=12,color=fontColor)
            plt.xticks(())
            plt.yticks(())
        if fig_title: 
            fig.suptitle(fig_title+'\n', fontsize=20)

        plt.show(block=True)

    def heatmap(self, cm):
        sns.heatmap(cm, annot=True)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.show(block=True)        

def test() -> None:
    imagePrediction = ImagePrediction()
    X, y = imagePrediction.prepare_X_y()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    pca = RandomizedPCA(n_components=150, svd_solver='randomized', whiten=True, random_state=42).fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    svc = SVC(kernel='rbf', class_weight='balanced')
    model = make_pipeline(pca, svc)

    C_range = np.logspace(-1, 5, 4)
    gamma_range = np.logspace(-3, 0, 4)
    param_grid = dict(svc__gamma=gamma_range, svc__C=C_range)
    gsv = model_selection.GridSearchCV(model, param_grid)

    gsv = gsv.fit(X_train_pca, y_train)

    y_pred = gsv.predict(X_test_pca)

    names = imagePrediction.get_names()
    print(metrics.classification_report(y_test, y_pred, target_names=names))

    n_samples, h, w = imagePrediction.get_shape_data()
    imagePrediction.plot_gallery(X_test, names[y_pred], names[y_test], h, w, "Predictions")

    cm = metrics.confusion_matrix( names[y_pred], names[y_test],labels=names)
    imagePrediction.heatmap(cm)
    
if __name__ == "__main__":
    test()