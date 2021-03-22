
################################################ DECLARATIONS ################################################
__author__ 	= 'Jimit Doshi'
__EXEC_NAME__ 	= "layers"

import tensorflow as _tf
import tensorflow_addons as _tfa

# Initializer
#INITIALIZER = _tf.keras.initializers.GlorotNormal( seed=None )
#INITIALIZER = _tf.keras.initializers.he_normal( seed=None )

################################################ FUNCTIONS ################################################

#DEF
def getEMA( var ):
	ema_var = _tf.train.ExponentialMovingAverage( 0.99 )
	var_ = ema_var.apply( [var] )
	var_av = ema_var.average( var )

	return var_, var_av
#ENDDEF

#DEF
def conv_layer( inp, f=4, k=1, s=1, upsample=False, activation=None, use_bias=False ):

	# IF UPSAMPLING IS NEEDED
	if upsample:
		# Transposed convolution layer for upsampling
		return _tf.keras.layers.Conv2DTranspose( filters=f, \
					kernel_size=[k,k], \
					strides=[s,s], \
					padding="same", \
					use_bias=use_bias, \
					activation=activation )( inp )
	else:
		# Normal convolution layer
		return _tf.keras.layers.Conv2D( filters=f, \
					kernel_size=[k,k], \
					strides=[s,s], \
					padding="same", \
					use_bias=use_bias, \
					activation=activation )( inp )
	# ENDIF UPSAMPLING
#ENDDEF

#DEF
def batch_norm( inp ):

	return _tf.keras.layers.BatchNormalization( center=True, \
						scale=True, \
						axis=-1, \
						momentum=0.99, \
						epsilon=1e-5, \
						fused=True )( inp )
#ENDDEF

#DEF
def instance_norm( inp ):

	return _tfa.layers.InstanceNormalization( center=True, \
						scale=True, \
						axis=-1, \
						beta_initializer="random_uniform", \
						gamma_initializer="random_uniform" )( inp )
#ENDDEF

#DEF
def norm_layer( inp, norm='batch' ):
	if norm == 'batch':
		return batch_norm( inp )
	elif norm == 'instance':
		return instance_norm( inp )
#ENDDEF

#DEF
def maxpool_layer( inp, pool, stride ):
	return _tf.keras.layers.MaxPool2D( pool_size=pool, \
					strides=stride, \
					padding='same', \
					data_format='channels_last' )( inp )

#ENDDEF

#DEF
def get_onehot( y,ls,xy,c ):
	if ls>0:
		oh = _tf.one_hot( \
			indices=_tf.reshape( y,[-1,xy,xy] ), \
			 depth=c )
		rls  = _tf.math.maximum( 0.0, \
				_tf.random.truncated_normal( [], \
							mean=0, \
							stddev=ls/2, \
							dtype=_tf.float32 ) \
					)
		
		return ( oh * (1 - rls) + 0.5 * rls )
	
	else:
		return _tf.one_hot( \
			indices=_tf.reshape( y,[-1,xy,xy] ), \
			 depth=c )
#ENDDEF

###################################################################
########################## VERSION 1 ##############################
###################################################################



#DEF
def conv_layer_resample_v1( inp, filters, ksize, stride, upsample=False, norm='batch' ):

	# convolution layer
	x = conv_layer( inp, f=filters, k=ksize, s=stride, upsample=upsample )
	
	# normalization layer
	x = norm_layer( x,norm )

	# ReLU activation
	x = _tf.nn.leaky_relu( x )

	return x

#ENDDEF

#DEF
def UNetBlock_v1( inp_layer, filters=64, ksize=3, norm='batch' ):

	conv1 = conv_layer_resample_v1( inp=inp_layer, \
						filters=filters, \
						ksize=ksize, \
						stride=1, \
						upsample=False, \
						norm=norm )

	conv2 = conv_layer_resample_v1( inp=conv1, \
						filters=filters, \
						ksize=ksize, \
						stride=1, \
						upsample=False, \
						norm=norm )

		
	### Return final resUnit
	return conv2

#ENDDEF

#DEF
def ResUnit_v1( inp_layer, filters=64, ksize=3, norm='batch' ):

	# convolution layer with f/4 filters of size 1x1
	conv1 = conv_layer( inp_layer, f=filters, k=ksize, s=1, upsample=False )

	# normalization layer
	bn1 = norm_layer( conv1,norm )

	# ReLU activation
	relu1 = _tf.nn.leaky_relu( bn1 )

	# convolution layer with f/4 filters of size kxk
	conv2 = conv_layer( relu1, f=filters, k=ksize, s=1, upsample=False )

	# normalization layer
	bn2 = norm_layer( conv2,norm )

	### SHORTCUT RESIDUAL LAYER
	res = bn2 + inp_layer

	# ReLU activation
	relu2 = _tf.nn.leaky_relu( res )

	### Return final resUnit
	return relu2

#ENDDEF

#DEF
def ResNetUnit_v1( inp_layer, filters=64, ksize=3, norm='batch' ):

	# convolution layer with f/2 filters of size 1x1
	conv1 = conv_layer( inp_layer, f=filters/2, k=1, s=1, upsample=False )

	# normalization layer
	bn1 = norm_layer( conv1,norm )

	# ReLU activation
	relu1 = _tf.nn.leaky_relu( bn1 )

	# convolution layer with f/2 filters of size kxk
	conv2 = conv_layer( relu1, f=filters/2, k=ksize, s=1, upsample=False )

	# normalization layer
	bn2 = norm_layer( conv2,norm )
	
	# ReLU activation
	relu2 = _tf.nn.leaky_relu( bn2 )

	# convolution layer with f filters of size 1x1
	conv3 = conv_layer( relu2, f=filters, k=1, s=1, upsample=False )

	# normalization layer
	bn3 = norm_layer( conv3,norm )

	### SHORTCUT RESIDUAL LAYER
	res = bn3 + inp_layer
	
	# ReLU activation
	relu3 = _tf.nn.leaky_relu( res )
		
	### Return final layer
	return relu3

#ENDDEF

#DEF
def ResInc_v1( inp_layer, filters=64, ksize=3, norm='batch' ):
	
	### Branch 1
	conv1 = conv_layer_resample_v1( inp=inp_layer, filters=filters/4, ksize=1, stride=1, \
						upsample=False, norm=norm )
						
	### Branch 2
	conv2 = conv_layer_resample_v1( inp=inp_layer, filters=filters/4, ksize=1, stride=1, \
						upsample=False, norm=norm )
	conv2 = conv_layer_resample_v1( inp=conv2, filters=filters/4, ksize=ksize, stride=1, \
						upsample=False, norm=norm )
	
	### Branch 3
	conv3 = conv_layer_resample_v1( inp=inp_layer, filters=filters/4, ksize=1, stride=1, \
						upsample=False, norm=norm )
	conv3 = conv_layer_resample_v1( inp=conv3, filters=filters/4, ksize=ksize, stride=1, \
						upsample=False, norm=norm )
	conv3 = conv_layer_resample_v1( inp=conv3, filters=filters/4, ksize=ksize, stride=1, \
						upsample=False, norm=norm )

	### Branch 4
	conv4 = conv_layer_resample_v1( inp=inp_layer, filters=filters/4, ksize=1, stride=1, \
						upsample=False, norm=norm )
	conv4 = conv_layer_resample_v1( inp=conv4, filters=filters/4, ksize=ksize, stride=1, \
						upsample=False, norm=norm )
	conv4 = conv_layer_resample_v1( inp=conv4, filters=filters/4, ksize=ksize, stride=1, \
						upsample=False, norm=norm )
	conv4 = conv_layer_resample_v1( inp=conv4, filters=filters/4, ksize=ksize, stride=1, \
						upsample=False, norm=norm )

	### Concatenate branches
	concat = _tf.concat( [ conv1,conv2,conv3,conv4 ], axis=-1 )
	
	### Convolution
	# conv
	conv = conv_layer( inp=concat, f=filters, k=1, s=1, upsample=False, activation=None )

	# normalization layer
	bn = norm_layer( conv,norm )

	### SHORTCUT RESIDUAL LAYER
	res = bn + inp_layer

	# ReLU activation
	relu = _tf.nn.leaky_relu( res )

	### Return final resunit
	return relu
#ENDDEF


#DEF
def ResNetInc_v1( inp_layer, filters=64, ksize=3, norm='batch' ):
	
	### First 3x3 convolution layer
	conv = conv_layer_resample_v1( inp=inp_layer, filters=filters, ksize=ksize, stride=1, \
						upsample=False, norm=norm )
	
	### Branch 1
	conv1 = conv_layer_resample_v1( inp=conv, filters=filters/4, ksize=1, stride=1, \
						upsample=False, norm=norm )
						
	### Branch 2
	conv2 = conv_layer_resample_v1( inp=conv, filters=filters/4, ksize=1, stride=1, \
						upsample=False, norm=norm )
	conv2 = conv_layer_resample_v1( inp=conv2, filters=filters/4, ksize=ksize, stride=1, \
						upsample=False, norm=norm )
	
	### Branch 3
	conv3 = conv_layer_resample_v1( inp=conv, filters=filters/4, ksize=1, stride=1, \
						upsample=False, norm=norm )
	conv3 = conv_layer_resample_v1( inp=conv3, filters=filters/4, ksize=ksize, stride=1, \
						upsample=False, norm=norm )
	conv3 = conv_layer_resample_v1( inp=conv3, filters=filters/4, ksize=ksize, stride=1, \
						upsample=False, norm=norm )

	### Branch 4
	conv4 = conv_layer_resample_v1( inp=conv, filters=filters/4, ksize=1, stride=1, \
						upsample=False, norm=norm )
	conv4 = conv_layer_resample_v1( inp=conv4, filters=filters/4, ksize=ksize, stride=1, \
						upsample=False, norm=norm )
	conv4 = conv_layer_resample_v1( inp=conv4, filters=filters/4, ksize=ksize, stride=1, \
						upsample=False, norm=norm )
	conv4 = conv_layer_resample_v1( inp=conv4, filters=filters/4, ksize=ksize, stride=1, \
						upsample=False, norm=norm )

	### Concatenate branches
	concat = _tf.concat( [ conv1,conv2,conv3,conv4 ], axis=-1 )
	
	### Convolution
	# conv
	final_conv = conv_layer( inp=concat, f=filters, k=1, s=1, upsample=False, activation=None )

	# normalization layer
	bn = norm_layer( final_conv,norm )

	### SHORTCUT RESIDUAL LAYER
	res = bn + inp_layer

	# ReLU activation
	relu = _tf.nn.leaky_relu( res )

	### Return final resunit
	return relu
#ENDDEF

#DEF
def ResInc_f2_v1( inp_layer, filters=64, ksize=3, norm='batch' ):
	
	### Branch 1
	conv1 = conv_layer_resample_v1( inp=inp_layer, filters=filters/2, ksize=1, stride=1, \
						upsample=False, norm=norm )
						
	### Branch 2
	conv2 = conv_layer_resample_v1( inp=inp_layer, filters=filters/2, ksize=1, stride=1, \
						upsample=False, norm=norm )
	conv2 = conv_layer_resample_v1( inp=conv2, filters=filters/2, ksize=ksize, stride=1, \
						upsample=False, norm=norm )
	
	### Branch 3
	conv3 = conv_layer_resample_v1( inp=inp_layer, filters=filters/2, ksize=1, stride=1, \
						upsample=False, norm=norm )
	conv3 = conv_layer_resample_v1( inp=conv3, filters=filters/2, ksize=ksize, stride=1, \
						upsample=False, norm=norm )
	conv3 = conv_layer_resample_v1( inp=conv3, filters=filters/2, ksize=ksize, stride=1, \
						upsample=False, norm=norm )

	### Branch 4
	conv4 = conv_layer_resample_v1( inp=inp_layer, filters=filters/2, ksize=1, stride=1, \
						upsample=False, norm=norm )
	conv4 = conv_layer_resample_v1( inp=conv4, filters=filters/2, ksize=ksize, stride=1, \
						upsample=False, norm=norm )
	conv4 = conv_layer_resample_v1( inp=conv4, filters=filters/2, ksize=ksize, stride=1, \
						upsample=False, norm=norm )
	conv4 = conv_layer_resample_v1( inp=conv4, filters=filters/2, ksize=ksize, stride=1, \
						upsample=False, norm=norm )

	### Concatenate branches
	concat = _tf.concat( [ conv1,conv2,conv3,conv4 ], axis=-1 )
	
	### Convolution
	# conv
	conv = conv_layer( inp=concat, f=filters, k=1, s=1, upsample=False, activation=None )

	# normalization layer
	bn = norm_layer( conv,norm )

	### SHORTCUT RESIDUAL LAYER
	res = bn + inp_layer

	# ReLU activation
	relu = _tf.nn.leaky_relu( res )

	### Return final resunit
	return relu
#ENDDEF

#DEF
def ResInc_f2x3_v1( inp_layer, filters=64, ksize=3, norm='batch' ):
	
	### Branch 1
	conv1 = conv_layer_resample_v1( inp=inp_layer, filters=filters/2, ksize=1, stride=1, \
						upsample=False, norm=norm )
						
	### Branch 2
	conv2 = conv_layer_resample_v1( inp=inp_layer, filters=filters/2, ksize=1, stride=1, \
						upsample=False, norm=norm )
	conv2 = conv_layer_resample_v1( inp=conv2, filters=filters/2, ksize=ksize, stride=1, \
						upsample=False, norm=norm )
	
	### Branch 3
	conv3 = conv_layer_resample_v1( inp=inp_layer, filters=filters/2, ksize=1, stride=1, \
						upsample=False, norm=norm )
	conv3 = conv_layer_resample_v1( inp=conv3, filters=filters/2, ksize=ksize, stride=1, \
						upsample=False, norm=norm )
	conv3 = conv_layer_resample_v1( inp=conv3, filters=filters/2, ksize=ksize, stride=1, \
						upsample=False, norm=norm )

	### Concatenate branches
	concat = _tf.concat( [ conv1,conv2,conv3 ], axis=-1 )
	
	### Convolution
	# conv
	conv = conv_layer( inp=concat, f=filters, k=1, s=1, upsample=False, activation=None )

	# normalization layer
	bn = norm_layer( conv,norm )

	### SHORTCUT RESIDUAL LAYER
	res = bn + inp_layer

	# ReLU activation
	relu = _tf.nn.leaky_relu( res )

	### Return final resunit
	return relu
#ENDDEF


#DEF
def ResInc_f4x4_v1( inp_layer, filters=64, ksize=3, norm='batch' ):
	
	### Branch 1
	conv1 = conv_layer_resample_v1( inp=inp_layer, filters=filters/2, ksize=1, stride=1, \
						upsample=False, norm=norm )
						
	### Branch 2
	conv2 = conv_layer_resample_v1( inp=inp_layer, filters=filters/4, ksize=1, stride=1, \
						upsample=False, norm=norm )
	conv2 = conv_layer_resample_v1( inp=conv2, filters=filters/2, ksize=ksize, stride=1, \
						upsample=False, norm=norm )
	
	### Branch 3
	conv3 = conv_layer_resample_v1( inp=inp_layer, filters=filters/4, ksize=1, stride=1, \
						upsample=False, norm=norm )
	conv3 = conv_layer_resample_v1( inp=conv3, filters=filters/4, ksize=ksize, stride=1, \
						upsample=False, norm=norm )
	conv3 = conv_layer_resample_v1( inp=conv3, filters=filters/2, ksize=ksize, stride=1, \
						upsample=False, norm=norm )

	### Branch 4
	conv4 = conv_layer_resample_v1( inp=inp_layer, filters=filters/4, ksize=1, stride=1, \
						upsample=False, norm=norm )
	conv4 = conv_layer_resample_v1( inp=conv4, filters=filters/4, ksize=ksize, stride=1, \
						upsample=False, norm=norm )
	conv4 = conv_layer_resample_v1( inp=conv4, filters=filters/2, ksize=ksize, stride=1, \
						upsample=False, norm=norm )
	conv4 = conv_layer_resample_v1( inp=conv4, filters=filters/2, ksize=ksize, stride=1, \
						upsample=False, norm=norm )

	### Concatenate branches
	concat = _tf.concat( [ conv1,conv2,conv3,conv4 ], axis=-1 )
	
	### Convolution
	# conv
	conv = conv_layer( inp=concat, f=filters, k=1, s=1, upsample=False, activation=None )

	# normalization layer
	bn = norm_layer( conv,norm )

	### SHORTCUT RESIDUAL LAYER
	res = bn + inp_layer

	# ReLU activation
	relu = _tf.nn.leaky_relu( res )

	### Return final resunit
	return relu
#ENDDEF


#DEF
def ResInc_x3_v1( inp_layer, filters=64, ksize=3, norm='batch' ):
	
	### Branch 2
	conv2 = conv_layer_resample_v1( inp=inp_layer, filters=filters/4, ksize=1, stride=1, \
						upsample=False, norm=norm )
	conv2 = conv_layer_resample_v1( inp=conv2, filters=filters/4, ksize=ksize, stride=1, \
						upsample=False, norm=norm )
	
	### Branch 3
	conv3 = conv_layer_resample_v1( inp=inp_layer, filters=filters/2, ksize=1, stride=1, \
						upsample=False, norm=norm )
	conv3 = conv_layer_resample_v1( inp=conv3, filters=filters/2, ksize=ksize, stride=1, \
						upsample=False, norm=norm )
	conv3 = conv_layer_resample_v1( inp=conv3, filters=filters/2, ksize=ksize, stride=1, \
						upsample=False, norm=norm )

	### Branch 4
	conv4 = conv_layer_resample_v1( inp=inp_layer, filters=filters/4, ksize=1, stride=1, \
						upsample=False, norm=norm )
	conv4 = conv_layer_resample_v1( inp=conv4, filters=filters/4, ksize=ksize, stride=1, \
						upsample=False, norm=norm )
	conv4 = conv_layer_resample_v1( inp=conv4, filters=filters/4, ksize=ksize, stride=1, \
						upsample=False, norm=norm )
	conv4 = conv_layer_resample_v1( inp=conv4, filters=filters/4, ksize=ksize, stride=1, \
						upsample=False, norm=norm )

	### Branch 5
	
	### Concatenate branches
	concat = _tf.concat( [ conv2,conv3,conv4 ], axis=-1 )
	
	### Convolution
	# conv
	conv = conv_layer( inp=concat, f=filters, k=1, s=1, upsample=False, activation=None )

	# normalization layer
	bn = norm_layer( conv,norm )

	### SHORTCUT RESIDUAL LAYER
	res = bn + inp_layer

	# ReLU activation
	relu = _tf.nn.leaky_relu( res )

	### Return final resunit
	return relu
#ENDDEF


#DEF
def Inc_v1( inp_layer, filters=64, ksize=3, norm='batch' ):
	
	### Branch 1
	conv1 = conv_layer_resample_v1( inp=inp_layer, filters=filters/4, ksize=1, stride=1, \
						upsample=False, norm=norm )
						
	### Branch 2
	conv2 = conv_layer_resample_v1( inp=inp_layer, filters=filters/4, ksize=1, stride=1, \
						upsample=False, norm=norm )
	conv2 = conv_layer_resample_v1( inp=conv2, filters=filters/4, ksize=ksize, stride=1, \
						upsample=False, norm=norm )
	
	### Branch 3
	conv3 = conv_layer_resample_v1( inp=inp_layer, filters=filters/4, ksize=1, stride=1, \
						upsample=False, norm=norm )
	conv3 = conv_layer_resample_v1( inp=conv3, filters=filters/4, ksize=ksize, stride=1, \
						upsample=False, norm=norm )
	conv3 = conv_layer_resample_v1( inp=conv3, filters=filters/4, ksize=ksize, stride=1, \
						upsample=False, norm=norm )

	### Branch 4
	conv4 = conv_layer_resample_v1( inp=inp_layer, filters=filters/4, ksize=1, stride=1, \
						upsample=False, norm=norm )
	conv4 = conv_layer_resample_v1( inp=conv4, filters=filters/4, ksize=ksize, stride=1, \
						upsample=False, norm=norm )
	conv4 = conv_layer_resample_v1( inp=conv4, filters=filters/4, ksize=ksize, stride=1, \
						upsample=False, norm=norm )
	conv4 = conv_layer_resample_v1( inp=conv4, filters=filters/4, ksize=ksize, stride=1, \
						upsample=False, norm=norm )

	### Concatenate branches
	concat = _tf.concat( [ conv1,conv2,conv3,conv4 ], axis=-1 )
	
	### Convolution
	# conv
	conv = conv_layer( inp=concat, f=filters, k=1, s=1, upsample=False, activation=None )

	# normalization layer
	bn = norm_layer( conv,norm )

	# ReLU activation
	relu = _tf.nn.leaky_relu( bn )

	### Return final resunit
	return relu
#ENDDEF

#DEF
def Inc_f2_v1( inp_layer, filters=64, ksize=3, norm='batch' ):
	
	### Branch 1
	conv1 = conv_layer_resample_v1( inp=inp_layer, filters=filters/2, ksize=1, stride=1, \
						upsample=False, norm=norm )
						
	### Branch 2
	conv2 = conv_layer_resample_v1( inp=inp_layer, filters=filters/2, ksize=1, stride=1, \
						upsample=False, norm=norm )
	conv2 = conv_layer_resample_v1( inp=conv2, filters=filters/2, ksize=ksize, stride=1, \
						upsample=False, norm=norm )
	
	### Branch 3
	conv3 = conv_layer_resample_v1( inp=inp_layer, filters=filters/2, ksize=1, stride=1, \
						upsample=False, norm=norm )
	conv3 = conv_layer_resample_v1( inp=conv3, filters=filters/2, ksize=ksize, stride=1, \
						upsample=False, norm=norm )
	conv3 = conv_layer_resample_v1( inp=conv3, filters=filters/2, ksize=ksize, stride=1, \
						upsample=False, norm=norm )

	### Branch 4
	conv4 = conv_layer_resample_v1( inp=inp_layer, filters=filters/2, ksize=1, stride=1, \
						upsample=False, norm=norm )
	conv4 = conv_layer_resample_v1( inp=conv4, filters=filters/2, ksize=ksize, stride=1, \
						upsample=False, norm=norm )
	conv4 = conv_layer_resample_v1( inp=conv4, filters=filters/2, ksize=ksize, stride=1, \
						upsample=False, norm=norm )
	conv4 = conv_layer_resample_v1( inp=conv4, filters=filters/2, ksize=ksize, stride=1, \
						upsample=False, norm=norm )

	### Concatenate branches
	concat = _tf.concat( [ conv1,conv2,conv3,conv4 ], axis=-1 )
	
	### Convolution
	# conv
	conv = conv_layer( inp=concat, f=filters, k=1, s=1, upsample=False, activation=None )

	# normalization layer
	bn = norm_layer( conv,norm )

	# ReLU activation
	relu = _tf.nn.leaky_relu( bn )

	### Return final resunit
	return relu
#ENDDEF
