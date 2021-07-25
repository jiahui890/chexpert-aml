import pickle
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


models = {
    "MultinomialNB": MultinomialNB(),
    "SGDClassifier": SGDClassifier(loss='log'),
    "SGDClassifier_Elastic": SGDClassifier(loss='huber', penalty='elasticnet', n_jobs=-1, class_weight='balanced')
}