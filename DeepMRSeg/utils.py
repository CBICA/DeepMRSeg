
################################################ DECLARATIONS ################################################
__author__ 	= 'Jimit Doshi'
__EXEC_NAME__ 	= "losses"

import tensorflow as _tf
import matplotlib.pyplot as _plt
import numpy as _np
import io as _io
import itertools as _itertools
import csv as _csv

################################################ FUNCTIONS ################################################

### Taken from: https://www.tensorflow.org/tensorboard/image_summaries
#DEF
def plot_confusion_matrix(cm, class_names):
	"""
	Returns a matplotlib figure containing the plotted confusion matrix.

	Args:
		cm (array, shape = [n, n]): a confusion matrix of integer classes
		class_names (array, shape = [n]): String names of the integer classes
	"""
	# Normalize confusion matrix
	cm_r = cm / cm.sum(axis=1)[:, _np.newaxis]

	reca = _np.diag(cm_r).mean()

	# Prepare figure
	figure = _plt.figure(figsize=(8,8))
	_plt.imshow( cm_r, interpolation='nearest', cmap=_plt.cm.Blues)
	_plt.title( "Normalized confusion matrix ($Recall_{mean}=%.4f$)" % reca )
	_plt.colorbar()
	tick_marks = _np.arange(len(class_names))
	#IF
	if len( class_names ) < 50:
		_plt.xticks(tick_marks, class_names, rotation=45)
		_plt.yticks(tick_marks, class_names)
	else:
		# Disable ticks if there are too many classes
		_plt.xticks( [] )
		_plt.yticks( [] )
	#ENDIF

	# Compute the labels from the normalized confusion matrix.
	labels = _np.around( cm_r, decimals=2 )

	# Use white text if squares are dark; otherwise black.
	if len( class_names ) < 50:
		threshold = 0.5 #cm.max() / 2.
		for i, j in _itertools.product(range(cm_r.shape[0]), range(cm_r.shape[1])):
			color = "white" if cm_r[i, j] > threshold else "black"
			_plt.text( j, i, labels[i, j], horizontalalignment="center", color=color )

	_plt.tight_layout()
	_plt.ylabel('True label')
	_plt.xlabel('Predicted label')

	return figure
#ENDDEF

### Taken from: 
#DEF
def plot_to_image(figure):
	"""Converts the matplotlib plot specified by 'figure' to a PNG image and returns it. The supplied figure is closed and inaccessible after this call."""
	# Save the plot to a PNG in memory.
	buf = _io.BytesIO()
	_plt.savefig(buf, format='png')

	# Closing the figure prevents it from being displayed directly inside
	# the notebook.
	_plt.close(figure)
	buf.seek(0)

	# Convert PNG buffer to TF image
	image = _tf.image.decode_png(buf.getvalue(), channels=4)

	# Add the batch dimension
	image = _tf.expand_dims(image, 0)

	return image
#ENDDEF

#DEF
def get_roi_indices( roicsv=None, num_classes=2 ):
	### Encode indices to ROIs if provided
	roi_indices = []
	#IF
	if roicsv:
		#WITH
		with open(roicsv) as roicsvfile:
			roi_reader = _csv.DictReader( roicsvfile )

			#FOR
			for roi_row in roi_reader:
				roi_indices.extend( [ [int(roi_row['Index']), int(roi_row['ROI'])] ] )
			#ENDFOR
		#ENDWITH
	else:
		for i in range( num_classes ):
			roi_indices.extend( [ [i,i] ] )
	#ENDIF

	return roi_indices
#ENDDEF
