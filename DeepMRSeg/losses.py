"""
This module contains various loss functions and helper functions

Functions:
	get_tp_fp_fn( y_true,y_pred )
	get_iou( y_true,y_pred )
	get_dice( y_true,y_pred )
	soft_iou_loss( y_true, y_pred )
	focal_iou_loss( y_true, y_pred, gamma=1 )
	soft_dice_loss( y_true, y_pred )
	focal_dice_loss( y_true, y_pred, gamma=1 )
	mae_loss( y_true, y_pred, gamma=1 )
	cce_loss( y_true, y_pred, gamma=1 )
	combo_loss( oh,probs,gamma,alpha )
	get_combo_loss( oh_d1,probs_d1,probs_d2,probs_d4,gamma,ds,xy,alpha )
"""

################################################ DECLARATIONS ################################################
__author__ 	= 'Jimit Doshi'
__EXEC_NAME__ 	= "losses"

### Import modules
import tensorflow as _tf

EPS = _tf.constant( 1e-7 )

################################################ FUNCTIONS ################################################

### y_true dimensions [b,x,y,c]
### y_pred dimensions [b,x,y,c]

#DEF
#@_tf.function
def get_tp_fp_fn( y_true,y_pred ):
	"""For a given pair of y_true and y_pred, calculate TP, FP and FN.

	Args:
		y_true: ground truth one_hot encodings of shape (b,x,y,c)
		y_pred: predicted probabilities of shape (b,x,y,c)
	Returns:
		tp: true positives of shape (b,c)
		fp: false positives of shape (b,c)
		fn: false negatives of shape (b,c)
	"""
	tp = _tf.math.reduce_sum( y_true * y_pred, axis=[1,2] )
	fp = _tf.math.reduce_sum( (1-y_true) * y_pred, axis=[1,2] )
	fn = _tf.math.reduce_sum( y_true * (1-y_pred), axis=[1,2] )
	
	# Shape: [b,c]
	return tp, fp, fn
#ENDDEF

#DEF
#@_tf.function
def get_iou( y_true,y_pred ):
	"""For a given pair of y_true and y_pred, calculate IOU.

	Args:
		y_true: ground truth one_hot encodings of shape (b,x,y,c)
		y_pred: predicted probabilities of shape (b,x,y,c)
	Returns:
		masked_iou: a ragged Tensor containing IOU of shape (b,None)
	"""
	tp, fp, fn = get_tp_fp_fn( y_true, y_pred )

	# Shape: [b,c]
	iou = ( tp ) / ( tp + fp + fn )

	# Mask out NaNs i.e. mask out classes that are not present in y_true as well as in y_pred
	return _tf.ragged.boolean_mask( iou, _tf.logical_not( _tf.math.is_nan(iou) ) )
#ENDDEF

#DEF
#@_tf.function
def get_dice( y_true,y_pred ):
	"""For a given pair of y_true and y_pred, calculate Dice overlap.

	Args:
		y_true: ground truth one_hot encodings of shape (b,x,y,c)
		y_pred: predicted probabilities of shape (b,x,y,c)
	Returns:
		masked_dice: a ragged Tensor containing Dice of shape (b,None)
	"""
	tp, fp, fn = get_tp_fp_fn( y_true, y_pred )

	# Shape: [b,c]
	dice = ( 2*tp ) / ( 2*tp + fp + fn )

	# Mask out NaNs i.e. mask out classes that are not present in y_true as well as in y_pred
	return _tf.ragged.boolean_mask( dice, _tf.logical_not( _tf.math.is_nan(dice) ) )
#ENDDEF

#DEF
#@_tf.function
def soft_iou_loss( y_true, y_pred ):
	"""For a given pair of y_true and y_pred, calculate IOU loss.

	Args:
		y_true: ground truth one_hot encodings of shape (b,x,y,c)
		y_pred: predicted probabilities of shape (b,x,y,c)
	Returns:
		soft_iou_loss: IOU loss of shape (b)
	"""
	# Shape: [b]
	return _tf.math.reduce_mean( 1.0 - get_iou( y_true,y_pred ), axis=1 )
#ENDDEF

#DEF
#@_tf.function
def soft_dice_loss( y_true, y_pred ):
	"""For a given pair of y_true and y_pred, calculate Dice loss.

	Args:
		y_true: ground truth one_hot encodings of shape (b,x,y,c)
		y_pred: predicted probabilities of shape (b,x,y,c)
	Returns:
		soft_dice_loss: Dice loss of shape (b)
	"""
	# Shape: [b]
	return _tf.math.reduce_mean( 1.0 - get_dice( y_true,y_pred ), axis=1 )
#ENDDEF

#DEF
#@_tf.function
def focal_iou_loss( y_true, y_pred, gamma=1 ):
	"""For a given pair of y_true and y_pred, calculate focal IOU loss.

	Args:
		y_true: ground truth one_hot encodings of shape (b,x,y,c)
		y_pred: predicted probabilities of shape (b,x,y,c)
		gamma: exponent value (default: 1)
	Returns:
		focal_iou_loss: focal IOU loss of shape (b)
				( 1-iou ) ** gamma
	"""
	# Shape: [b]
	return _tf.math.reduce_mean( _tf.math.pow( 1.0-get_iou( y_true,y_pred ),gamma ), axis=1 )
#ENDDEF

#DEF
#@_tf.function
def focal_dice_loss( y_true, y_pred, gamma=1 ):
	"""For a given pair of y_true and y_pred, calculate focal IOU loss.

	Args:
		y_true: ground truth one_hot encodings of shape (b,x,y,c)
		y_pred: predicted probabilities of shape (b,x,y,c)
		gamma: exponent value (default: 1)
	Returns:
		focal_iou_loss: focal Dice loss of shape (b)
				( 1-dice ) ** gamma
	"""
	# Shape: [b]
	return _tf.math.reduce_mean( _tf.math.pow( 1.0-get_dice( y_true,y_pred ),gamma ), axis=1 )
#ENDDEF

#DEF
#@_tf.function
def mae_loss( y_true, y_pred, gamma=1 ):
	"""For a given pair of y_true and y_pred, calculate Mean Absolute Error.

	Args:
		y_true: ground truth one_hot encodings of shape (b,x,y,c)
		y_pred: predicted probabilities of shape (b,x,y,c)
		gamma: exponent value (default: 1)
	Returns:
		mae_loss: MAE loss of shape (b)
	"""
	# Shape: [b]
	return _tf.math.reduce_mean( _tf.math.pow( _tf.math.abs( y_true-y_pred ),gamma ), axis=[1,2,3] )
#ENDDEF

#DEF
#@_tf.function
def cce_loss( y_true, y_pred, gamma=1 ):
	"""For a given pair of y_true and y_pred, calculate Categorical Cross Entropy.

	Args:
		y_true: ground truth one_hot encodings of shape (b,x,y,c)
		y_pred: predicted probabilities of shape (b,x,y,c)
		gamma: exponent value (default: 1)
	Returns:
		cce_loss: CCE loss of shape (b)
	"""
	cce = -1.0 * ( y_true*_tf.math.log( EPS + y_pred ) \
		+ (1.0-y_true)*_tf.math.log( EPS + 1.0-y_pred ) )

	# Shape: [b]
	return _tf.math.reduce_mean( _tf.math.pow( cce,gamma ),axis=[1,2,3] )
#ENDDEF

#DEF
#@_tf.function
def combo_loss( y_true,y_pred,gamma=1,alpha=50 ):
	"""For a given pair of y_true and y_pred, calculate Combo Loss.

	Args:
		y_true: ground truth one_hot encodings of shape (b,x,y,c)
		y_pred: predicted probabilities of shape (b,x,y,c)
		gamma: exponent value (default: 1)
		alpha: loss weight for IOU loss (default: 50)
	Returns:
		total_loss: weighted sum of losses of shape (b)
		iou_loss: focal IOU loss of shape (b)
		mae_loss: focal MAE loss of shape (b)
		cce_loss: focal CCE loss of shape (b)
	"""
	iou = focal_iou_loss( y_true,y_pred,gamma )
	mae = mae_loss( y_true,y_pred,gamma )
	cce = cce_loss( y_true,y_pred,gamma )

	total_loss = alpha * iou + (100-alpha) * (mae + cce)

	return total_loss, iou, mae, cce
#ENDDEF

#DEF
#@_tf.function
def get_combo_loss( oh_d1,probs_d1,probs_d2,probs_d4,gamma=1,ds=False,xy=256,alpha=50 ):
	"""For the input one_hot encodings and predicted probabilities at the 3 levels, return total losses.

	Args:
		oh_d1: ground truth one_hot encodings of shape (b,x,y,c)
		probs_d1: predicted probabilities of shape (b,x,y,c)
		probs_d2: predicted probabilities of shape (b,x/2,y/2,c)
		probs_d4: predicted probabilities of shape (b,x/4,y/4,c)
		gamma: exponent value (default: 1)
		ds: whether to use deep supervision (default: False)
		xy: xy_width (default: 256)
		alpha: loss weight for IOU loss (default: 50)
	Returns:
		total_loss: weighted sum of combo losses of shape (b)
				total_loss_d1 + 0.5*total_loss_d2 + 0.25*total_loss_d4
		total_loss_d1: weighted sum of combo losses of shape (b)
		iou_loss: focal IOU loss of shape (b)
		mae_loss: focal MAE loss of shape (b)
		cce_loss: focal CCE loss of shape (b)
	"""
	oh_d2 = _tf.image.resize( oh_d1,[int(xy/2),int(xy/2)] )
	oh_d4 = _tf.image.resize( oh_d1,[int(xy/4),int(xy/4)] )

	total_loss_d1, iou_d1, mae_d1, cce_d1 = combo_loss( oh_d1,probs_d1,gamma,alpha )

	# IF
	if ds:
		total_loss_d2, _, _, _ = combo_loss( oh_d2,probs_d2,gamma,alpha )
		total_loss_d4, _, _, _ = combo_loss( oh_d4,probs_d4,gamma,alpha )

		total_loss = total_loss_d1 + 0.5*total_loss_d2 + 0.25*total_loss_d4
	else:
		total_loss = total_loss_d1
	# ENDIF

	return total_loss, total_loss_d1, iou_d1, mae_d1, cce_d1
#ENDDEF

