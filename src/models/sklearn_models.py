from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

models = {
    "MultinomialNB": MultinomialNB(),
    "SGDClassifier": SGDClassifier(loss='log'),
}