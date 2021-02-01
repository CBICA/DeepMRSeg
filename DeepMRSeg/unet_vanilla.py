"""
Created on Mon Jul  3 14:27:26 2017

@author: jimit
"""
################################################ DECLARATIONS ################################################
__author__     = 'Jimit Doshi'
__EXEC_NAME__     = "unet_resnet"

import os as _os
import sys as _sys
import tensorflow as _tf

_sys.path.append( _os.path.dirname( _sys.argv[0] ) )

from .layers import maxpool_layer, conv_layer
from .layers import conv_layer_resample_v1

################################################ FUNCTIONS ################################################

#INITIALIZER = _tf.keras.initializers.GlorotNormal( seed=None )
#INITIALIZER = _tf.keras.initializers.he_normal( seed=None )

# DEF UNET
def unet_vanilla( inp_layer,ksize=3,depth=None,filters=32,layers=None,num_classes=2,lite=False ):

	conv1 = conv_layer( inp_layer, f=filters, k=3, s=1, upsample=False, activation=_tf.nn.relu ); print(conv1)
	conv1 = conv_layer( conv1, f=filters, k=3, s=1, upsample=False, activation=_tf.nn.relu ); print(conv1)
	pool1 = maxpool_layer( conv1, 2, 2 ); print(pool1)

	conv2 = conv_layer( pool1, f=filters*2, k=3, s=1, upsample=False, activation=_tf.nn.relu ); print(conv2)
	conv2 = conv_layer( conv2, f=filters*2, k=3, s=1, upsample=False, activation=_tf.nn.relu ); print(conv2)
	pool2 = maxpool_layer( conv2, 2, 2 ); print(pool2)

	conv3 = conv_layer( pool2, f=filters*4, k=3, s=1, upsample=False, activation=_tf.nn.relu ); print(conv3)
	conv3 = conv_layer( conv3, f=filters*4, k=3, s=1, upsample=False, activation=_tf.nn.relu ); print(conv3)
	pool3 = maxpool_layer( conv3, 2, 2 ); print(pool3)

	conv4 = conv_layer( pool3, f=filters*8, k=3, s=1, upsample=False, activation=_tf.nn.relu ); print(conv4)
	conv4 = conv_layer( conv4, f=filters*8, k=3, s=1, upsample=False, activation=_tf.nn.relu ); print(conv4)
	drop4 = _tf.keras.layers.Dropout( 0.5 )( conv4 ); print(drop4)
	pool4 = maxpool_layer( drop4, 2, 2 ); print(pool4)

	conv5 = conv_layer( pool4, f=filters*16, k=3, s=1, upsample=False, activation=_tf.nn.relu ); print(conv5)
	conv5 = conv_layer( conv5, f=filters*16, k=3, s=1, upsample=False, activation=_tf.nn.relu ); print(conv5)
	drop5 = _tf.keras.layers.Dropout( 0.5 )( conv5 ); print(drop5)

	up6 = conv_layer( drop5, f=filters*8, k=2, s=2, upsample=True, activation=_tf.nn.relu ); print(up6)
	concat6 = _tf.concat( [ up6,conv4 ], axis=3 ); print(concat6)
	conv6 = conv_layer( concat6, f=filters*8, k=3, s=1, upsample=False, activation=_tf.nn.relu ); print(conv6)
	conv6 = conv_layer( conv6, f=filters*8, k=3, s=1, upsample=False, activation=_tf.nn.relu ); print(conv6)

	up7 = conv_layer( conv6, f=filters*4, k=2, s=2, upsample=True, activation=_tf.nn.relu ); print(up7)
	concat7 = _tf.concat( [ up7,conv3 ], axis=3 ); print(concat7)
	conv7 = conv_layer( concat7, f=filters*4, k=3, s=1, upsample=False, activation=_tf.nn.relu ); print(conv7)
	conv7 = conv_layer( conv7, f=filters*4, k=3, s=1, upsample=False, activation=_tf.nn.relu ); print(conv7)

	up8 = conv_layer( conv7, f=filters*2, k=2, s=2, upsample=True, activation=_tf.nn.relu ); print(up8)
	concat8 = _tf.concat( [ up8,conv2 ], axis=3 ); print(concat8)
	conv8 = conv_layer( concat8, f=filters*2, k=3, s=1, upsample=False, activation=_tf.nn.relu ); print(conv8)
	conv8 = conv_layer( conv8, f=filters*2, k=3, s=1, upsample=False, activation=_tf.nn.relu ); print(conv8)

	up9 = conv_layer( conv8, f=filters, k=2, s=2, upsample=True, activation=_tf.nn.relu ); print(up9)
	concat9 = _tf.concat( [ up9,conv1 ], axis=3 ); print(concat9)
	conv9 = conv_layer( concat9, f=filters, k=3, s=1, upsample=False, activation=_tf.nn.relu ); print(conv9)
	conv9 = conv_layer( conv9, f=filters, k=3, s=1, upsample=False, activation=_tf.nn.relu ); print(conv9)


	#####################
	### FINAL LAYERS ####
	#####################
	print("\n")

	### Intermediate Logits Layer
	y_conv_d4 = _tf.keras.layers.Conv2D( filters=num_classes, \
				kernel_size=[1,1], \
				strides=[1,1], \
				padding="same", \
				use_bias=False, \
				activation=None )( conv7 )
	print(y_conv_d4)

	y_conv_d2 = _tf.keras.layers.Conv2D( filters=num_classes, \
				kernel_size=[1,1], \
				strides=[1,1], \
				padding="same", \
				use_bias=False, \
				activation=None )( conv8 )
	print(y_conv_d4)

	### Logits Layer
	y_conv = _tf.keras.layers.Conv2D( filters=num_classes, \
				kernel_size=[1,1], \
				strides=[1,1], \
				padding="same", \
				use_bias=False, \
				activation=None )( conv9 )
	print(y_conv)


	#####################
	####### RETURN ######
	#####################

	return y_conv, y_conv_d2, y_conv_d4
    
# ENDDEF UNET


# DEF UNET
def unet_vanilla_bn( inp_layer,ksize=3,depth=None,filters=32,layers=None,num_classes=2,lite=False ):

	conv1 = conv_layer_resample_v1( inp=inp_layer, filters=filters, ksize=ksize, stride=1, \
		upsample=False ); print(conv1)
	conv1 = conv_layer_resample_v1( inp=conv1, filters=filters, ksize=ksize, stride=1, \
		upsample=False ); print(conv1)
	pool1 = maxpool_layer( conv1, 2, 2 ); print(pool1)

	conv2 = conv_layer_resample_v1( inp=pool1, filters=filters*2, ksize=ksize, stride=1, \
		upsample=False ); print(conv2)
	conv2 = conv_layer_resample_v1( inp=conv2, filters=filters*2, ksize=ksize, stride=1, \
		upsample=False ); print(conv2)
	pool2 = maxpool_layer( conv2, 2, 2 ); print(pool2)

	conv3 = conv_layer_resample_v1( inp=pool2, filters=filters*4, ksize=ksize, stride=1, \
		upsample=False ); print(conv3)
	conv3 = conv_layer_resample_v1( inp=conv3, filters=filters*4, ksize=ksize, stride=1, \
		upsample=False ); print(conv3)
	pool3 = maxpool_layer( conv3, 2, 2 ); print(pool3)

	conv4 = conv_layer_resample_v1( inp=pool3, filters=filters*8, ksize=ksize, stride=1, \
		upsample=False ); print(conv4)
	conv4 = conv_layer_resample_v1( inp=conv4, filters=filters*8, ksize=ksize, stride=1, \
		upsample=False ); print(conv4)
	drop4 = _tf.keras.layers.Dropout( 0.5 )( conv4 ); print(drop4)
	pool4 = maxpool_layer( drop4, 2, 2 ); print(pool4)


	conv5 = conv_layer_resample_v1( inp=pool4, filters=filters*16, ksize=ksize, stride=1, \
		upsample=False ); print(conv5)
	conv5 = conv_layer_resample_v1( inp=conv5, filters=filters*16, ksize=ksize, stride=1, \
		upsample=False ); print(conv5)
	drop5 = _tf.keras.layers.Dropout( 0.5 )( conv5 ); print(drop5)



	up6 = conv_layer_resample_v1( inp=drop5, filters=filters*8, ksize=2, stride=2, \
		upsample=True ); print(up6)
	concat6 = _tf.concat( [ up6,conv4 ], axis=3 ); print(concat6)
	conv6 = conv_layer_resample_v1( inp=concat6, filters=filters*8, ksize=ksize, stride=1, \
		upsample=False ); print(conv6)
	conv6 = conv_layer_resample_v1( inp=concat6, filters=filters*8, ksize=ksize, stride=1, \
		upsample=False ); print(conv6)

	up7 = conv_layer_resample_v1( inp=conv6, filters=filters*4, ksize=2, stride=2, \
		upsample=True ); print(up7)
	concat7 = _tf.concat( [ up7,conv3 ], axis=3 ); print(concat7)
	conv7 = conv_layer_resample_v1( inp=concat7, filters=filters*4, ksize=ksize, stride=1, \
		upsample=False ); print(conv7)
	conv7 = conv_layer_resample_v1( inp=conv7, filters=filters*4, ksize=ksize, stride=1, \
		upsample=False ); print(conv7)

	up8 = conv_layer_resample_v1( inp=conv7, filters=filters*2, ksize=2, stride=2, \
		upsample=True ); print(up8)
	concat8 = _tf.concat( [ up8,conv2 ], axis=3 ); print(concat8)
	conv8 = conv_layer_resample_v1( inp=concat8, filters=filters*2, ksize=ksize, stride=1, \
		upsample=False ); print(conv8)
	conv8 = conv_layer_resample_v1( inp=conv8, filters=filters*2, ksize=ksize, stride=1, \
		upsample=False ); print(conv8)

	up9 = conv_layer_resample_v1( inp=conv8, filters=filters, ksize=2, stride=2, \
		upsample=True ); print(up9)
	concat9 = _tf.concat( [ up9,conv1 ], axis=3 ); print(concat9)
	conv9 = conv_layer_resample_v1( inp=concat9, filters=filters, ksize=ksize, stride=1, \
		upsample=False ); print(conv9)
	conv9 = conv_layer_resample_v1( inp=conv9, filters=filters, ksize=ksize, stride=1, \
		upsample=False ); print(conv9)


	#####################
	### FINAL LAYERS ####
	#####################
	print("\n")

	### Intermediate Logits Layer
	y_conv_d4 = _tf.keras.layers.Conv2D( filters=num_classes, \
				kernel_size=[1,1], \
				strides=[1,1], \
				padding="same", \
				use_bias=False, \
				activation=None )( conv7 )
	print(y_conv_d4)

	y_conv_d2 = _tf.keras.layers.Conv2D( filters=num_classes, \
				kernel_size=[1,1], \
				strides=[1,1], \
				padding="same", \
				use_bias=False, \
				activation=None )( conv8 )
	print(y_conv_d4)

	### Logits Layer
	y_conv = _tf.keras.layers.Conv2D( filters=num_classes, \
				kernel_size=[1,1], \
				strides=[1,1], \
				padding="same", \
				use_bias=False, \
				activation=None )( conv9 )
	print(y_conv)


	#####################
	####### RETURN ######
	#####################

	return y_conv, y_conv_d2, y_conv_d4
    
# ENDDEF UNET
