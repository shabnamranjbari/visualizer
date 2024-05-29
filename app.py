from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.tree import DecisionTreeClassifier, plot_tree
import io
import base64
import pydotplus
import graphviz
from flask import Flask

app = Flask(__name__)
app.static_folder = 'static'


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/svm')
def svm():
    # Load dataset
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = iris.target

    # Create SVM model
    C = float(request.args.get('C', 1.0))
    svc = SVC(kernel='linear', C=C).fit(X, y)

    # Create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Plot decision boundary
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('SVM Decision Boundary with C = {}'.format(C))

    # Save plot to a string in base64 format
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return render_template('svm.html', plot_url=plot_url)


# KNN route
@app.route('/knn')
def knn():
    # Load dataset
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = iris.target

    # Create KNN model
    n_neighbors = int(request.args.get('n_neighbors', 5))
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X, y)

    # Create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Plot decision boundary for KNN
    Z_knn = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_knn = Z_knn.reshape(xx.shape)
    
    plt.figure()
    plt.contourf(xx, yy, Z_knn, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('KNN Decision Boundary with n_neighbors = {}'.format(n_neighbors))

    # Save plot to a string in base64 format
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return render_template('knn.html', plot_url=plot_url)


@app.route('/kmeans')
def kmeans():
    # Load dataset
    iris = datasets.load_iris()
    X = iris.data[:, :2]

    # Get number of clusters from the request
    n_clusters = int(request.args.get('n_clusters', 3))

    # Create and fit the KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)

    # Plot the clusters
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

    # Plot the centroids
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('KMeans Clustering with n_clusters = {}'.format(n_clusters))

    # Save plot to a string in base64 format
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return render_template('kmeans.html', plot_url=plot_url)

@app.route('/decision_tree')
def decision_tree():
    # Load dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Create and fit the Decision Tree model
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X, y)

    # Plot the decision tree
    plt.figure(figsize=(20,10))
    plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
    plt.title("Decision Tree for Iris Dataset")

    # Save plot to a string in base64 format
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return render_template('decision_tree.html', plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
