import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.


def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies, train_f1, valid_f1, train_auc, valid_auc):
	# TODO: Make plots for loss curves and accuracy curves.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.

	plt.plot(range(len(train_losses)), train_losses, label = 'train_losses')
	plt.plot(range(len(valid_losses)), valid_losses, label = 'valid_looses')
	plt.legend()
	plt.xlabel('Epoch')
	plt.ylabel('Loss')

	plt.savefig('./imgs/losses.png')

	plt.figure()
	plt.plot(range(len(train_accuracies)), train_accuracies, label='train_accuracies')
	plt.plot(range(len(valid_accuracies)), valid_accuracies, label='valid_accuracies')
	plt.legend()
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy in percentage')

	plt.savefig('./imgs/accuracies.png')

	plt.figure()
	plt.plot(range(len(train_f1)), train_f1, label='train_f1')
	plt.plot(range(len(valid_f1)), valid_f1, label='valid_f1')
	plt.legend()
	plt.xlabel('Epoch')
	plt.ylabel('f1')

	plt.savefig('./imgs/f1.png')

	plt.figure()
	plt.plot(range(len(train_auc)), train_auc, label='train_auc')
	plt.plot(range(len(valid_auc)), valid_auc, label='valid_auc')
	plt.legend()
	plt.xlabel('Epoch')
	plt.ylabel('auc')

	plt.savefig('./imgs/auc.png')

	zx = 1

def plot_confusion_matrix(results, class_names):
	# TODO: Make a confusion matrix plot.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	title = None,
	cmap = plt.cm.Blues

	y_true = []
	y_pred = []

	for item in results:
		y_true.append(item[0])
		y_pred.append(item[1])

	# Compute confusion matrix
	cm = confusion_matrix(y_true, y_pred)
	# Only use the labels that appear in the data
	classes = class_names

	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	ax.figure.colorbar(im, ax=ax)
	# We want to show all ticks...
	ax.set(xticks=np.arange(cm.shape[1]),
		   yticks=np.arange(cm.shape[0]),
		   # ... and label them with the respective list entries
		   xticklabels=classes, yticklabels=classes,
		   title=title,
		   ylabel='True label',
		   xlabel='Predicted label')

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			 rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	fmt = '.2f'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], fmt),
					ha="center", va="center",
					color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()

	plt.savefig('../imgs/confusionMatrix.png')

