from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import RocCurveDisplay

from numpy import unique

import matplotlib.pyplot as plt


def eval(clf, real, pred, ax, bx):
	accuracy = (pred == real).mean()
	confusion = confusion_matrix(pred, real)
	try:
		roc = roc_auc_score(real, pred)
	except:
		roc = None
	precision = precision_score(real, pred, average="macro")
	recall = recall_score(real, pred, average="macro")
	if len(unique(real))==2:
		RocCurveDisplay.from_predictions(real, pred).plot(ax=ax, name=clf)
		PrecisionRecallDisplay.from_predictions(real, pred).plot(ax=bx, name=clf)
	else:
		pass
	return accuracy, confusion, roc, precision, recall

def plot(sklearn, pyspark, metric):
	models = ["LinearSVC", "Logistic Regression", "Decision Tree", "Random Forest", "Naive Bayes", "Gradient Boosting", "MLP Classifier", "One vs Rest", "Keras Network"]
	plt.figure(figsize=(20,8))
	plt.bar(models, sklearn, align="edge", width=-0.3)
	plt.bar(models, pyspark, align="edge", width=0.3)
	plt.legend(["Sklearn "+metric, "Pyspark "+metric], loc="lower right")
	plt.title(metric, fontsize=16)
	plt.grid()
