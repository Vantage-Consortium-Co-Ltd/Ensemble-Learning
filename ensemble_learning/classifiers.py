"""Base classifiers (from scikit-learn) used inside the ensembles."""
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def knn():   return KNeighborsClassifier(n_neighbors=5)
def nb():    return GaussianNB()
def stump(): return DecisionTreeClassifier(max_depth=1, random_state=0)
def tree():  return DecisionTreeClassifier(max_depth=3, random_state=0)


BASES = {"KNN": knn, "NaiveBayes": nb, "Stump": stump, "Tree(d=3)": tree}
