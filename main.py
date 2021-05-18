import sklearn as sk
from sklearn.datasets import fetch_lfw_people
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import PCA
from sklearn.svm import SVC

np.random.seed(42)

dataset = fetch_lfw_people(min_faces_per_person=70, resize=0.4)  # labeled faces in the wild
n_samples, h, w = dataset.images.shape
X = dataset.data

print("Image shape : ", X.shape, "\nheight : ", h, "\nwidth : ", w)
print(f"There are {X.shape[1]} features ")

Y = dataset.target
names = dataset.target_names

n_classes = len(names)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

print(X_train,
      X_test.shape,
      y_train.shape,
      y_test.shape)

plt.imshow(X_train[0].reshape(h, w), cmap='gray')
plt.xlabel(names[y_train[0]])
plt.title("Example Image")
plt.show()

# PCA
reduced_features = 50

pca = PCA(n_components=reduced_features, whiten=True)
pca.fit(X_train)
eigenfaces = pca.components_.reshape((reduced_features, h, w))

X_train_reduced = pca.transform(X_train)
X_test_reduced = pca.transform(X_test)

print(X_train_reduced.shape, X_test_reduced.shape)

params = {
    "C": [1e3, 5e3, 1e4, 5e4, 1e5],
    "gamma": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]

}

clf = GridSearchCV(SVC(kernel="rbf", class_weight="balanced"), params)
clf = clf.fit(X_train_reduced, y_train)

print(clf.best_estimator_)

# best_clf = SVC(C=1000.0, class_weight='balanced', gamma=0.01)
# best_clf.fit(X_train_reduced, y_train)

test = clf.predict(X_test_reduced)
report = classification_report(y_test, test, target_names=names)
print("Classification Report : ")
print(report)

print("Confusion Matrix : ")
print(confusion_matrix(y_test, test))

print("Accuracy : ")
print(accuracy_score(y_test, test))
