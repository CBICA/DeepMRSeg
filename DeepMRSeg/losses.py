
################################################ DECLARATIONS ################################################
__author__ 	= 'Jimit Doshi'
__EXEC_NAME__ 	= "losses"

import tensorflow as _tf

EPS = _tf.constant( 1e-7 )

################################################ FUNCTIONS ################################################

### y_true dimensions [b,x,y,c]
### y_pred dimensions [b,x,y,c]

#DEF
#@_tf.function
def get_tp_fp_fn( y_true,y_pred ):
	
	tp = _tf.math.reduce_sum( y_true * y_pred, axis=[1,2] )
	fp = _tf.math.reduce_sum( (1-y_true) * y_pred, axis=[1,2] )
	fn = _tf.math.reduce_sum( y_true * (1-y_pred), axis=[1,2] )
	
	# Shape: [b,c]
	return tp, fp, fn
#ENDDEF

#DEF
#@_tf.function
def get_iou( y_true,y_pred ):
	
	tp, fp, fn = get_tp_fp_fn( y_true, y_pred )

	# Shape: [b,c]
	iou = ( tp ) / ( tp + fp + fn )

	# Mask out NaNs i.e. mask out classes that are not present in y_true as well as in y_pred
	masked_iou = _tf.ragged.boolean_mask( iou, _tf.logical_not( _tf.math.is_nan(iou) ) )

	# Shape: [b,c]
	return masked_iou
#ENDDEF

#DEF
#@_tf.function
def get_dice( y_true,y_pred ):
	
	tp, fp, fn = get_tp_fp_fn( y_true, y_pred )

	# Shape: [b,c]
	dice = ( 2*tp ) / ( 2*tp + fp + fn )

	# Mask out NaNs i.e. mask out classes that are not present in y_true as well as in y_pred
	masked_dice = _tf.ragged.boolean_mask( dice, _tf.logical_not( _tf.math.is_nan(dice) ) )

	# Shape: [b,c]
	return masked_dice
#ENDDEF

#DEF
#@_tf.function
def SoftIOULoss( y_true, y_pred ):
	# Shape: [b]
	return _tf.math.reduce_mean( 1.0 - get_iou( y_true,y_pred ), axis=1 )
#ENDDEF

#DEF
#@_tf.function
def FocalIOULoss( y_true, y_pred, gamma=1 ):
	# Shape: [b]
	return _tf.math.reduce_mean( _tf.math.pow( 1.0-get_iou( y_true,y_pred ),gamma ), axis=1 )
#ENDDEF

#DEF
#@_tf.function
def SoftDiceLoss( y_true, y_pred ):
	# Shape: [b]
	return _tf.math.reduce_mean( 1.0 - get_dice( y_true,y_pred ), axis=1 )
#ENDDEF

#DEF
#@_tf.function
def FocalDiceLoss( y_true, y_pred, gamma=1 ):
	# Shape: [b]
	return _tf.math.reduce_mean( _tf.math.pow( 1.0-get_dice( y_true,y_pred ),gamma ), axis=1 )
#ENDDEF

#DEF
#@_tf.function
def MAE( y_true, y_pred, gamma=1 ):
	# Shape: [b]
	return _tf.math.reduce_mean( _tf.math.pow( _tf.math.abs( y_true-y_pred ),gamma ), axis=[1,2,3] )
#ENDDEF

#DEF
#@_tf.function
def BCE( y_true, y_pred, gamma=1 ):

	bce = -1.0 * ( y_true*_tf.math.log( EPS + y_pred ) \
		+ (1.0-y_true)*_tf.math.log( EPS + 1.0-y_pred ) )

	# Shape: [b]
	return _tf.math.reduce_mean( _tf.math.pow( bce,gamma ),axis=[1,2,3] )
#ENDDEF

#DEF
#@_tf.function
def CombinedLoss( oh,probs,gamma,alpha ):

	iou = FocalIOULoss( oh,probs,gamma )
	mae = MAE( oh,probs,gamma )
	bce = BCE( oh,probs,gamma )

	total_loss = alpha * iou + (100-alpha) * (mae + bce)

	return total_loss, iou, mae, bce
#ENDDEF

#DEF
#@_tf.function
def getCombinedLoss( oh_d1,probs_d1,probs_d2,probs_d4,gamma,ds,xy,alpha ):

	oh_d2 = _tf.image.resize( oh_d1,[int(xy/2),int(xy/2)] )
	oh_d4 = _tf.image.resize( oh_d1,[int(xy/4),int(xy/4)] )

	total_loss_d1, iou_d1, mae_d1, bce_d1 = CombinedLoss( oh_d1,probs_d1,gamma,alpha )

	# IF
	if ds:
		total_loss_d2, _, _, _ = CombinedLoss( oh_d2,probs_d2,gamma,alpha )
		total_loss_d4, _, _, _ = CombinedLoss( oh_d4,probs_d4,gamma,alpha )

		total_loss = total_loss_d1 + 0.5*total_loss_d2 + 0.25*total_loss_d4
	else:
		total_loss = total_loss_d1
	# ENDIF

	return total_loss, total_loss_d1, iou_d1, mae_d1, bce_d1
#ENDDEF

