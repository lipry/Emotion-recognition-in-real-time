from sklearn import svm
from sklearn import multiclass
from sklearn import cross_validation
import pickle, datasets


with open(datasets.EMOTION_FEATURES_DATASET, "r") as infile:
    features, labels = pickle.load(infile)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
                        features,labels, test_size=0.4, random_state=0)

clf = multiclass.OneVsRestClassifier(svm.SVC(kernel='rbf' , C = 100, gamma=0.400))
clf.fit(X_train, y_train)

print "Accuracy: ", clf.score(X_test, y_test)
