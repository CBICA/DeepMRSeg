"""
Created on Mon Jul  3 14:27:26 2017

@author: jimit
"""
################################################ DECLARATIONS ################################################
__author__ 	= 'Jimit Doshi'
__EXEC_NAME__ 	= "unet_resnet"

import os as _os
import sys as _sys
import tensorflow as _tf


_sys.path.append( _os.path.dirname( _sys.argv[0] ) )


from layers import ResUnit_v1, conv_layer_resample_v1

################################################ FUNCTIONS ################################################


# DEF UNET
def unet_resnet( inp_layer,ksize=3,depth=None,filters=32,layers=None,keep_prob=1.0,num_classes=2,lite=False ):

	skips = []
	fm = [filters]
	xy = inp_layer.shape.as_list()[1]

	#####################
	#### PROJECTION #####
	#####################
	print("\n")

	# Input shape: [?,x,y,2]
	# Output shape: [?,x,y,f]

	### Project the input layer dimensions from 2 features to f features
	### This ensures the input layer size goes from [?,x,y,2] -> [?,x,y,f]
	conv = conv_layer_resample_v1( inp=inp_layer, \
					filters=fm[-1], \
					ksize=1, \
					stride=1, \
					upsample=False )
	print(conv)

	skips.append(conv)
	
	################################
	###### PRE ENCODING BLOCK ######
	################################
	print("\n")

	### Add Residual layers
	for l in range(0,layers):
		conv = ResUnit_v1( conv, filters=filters, ksize=ksize )
		print(conv)

	skips.append(conv)
	
	############################
	###### ENCODING BLOCK ######
	############################
	print("\n\nEncoding Blocks")

	# FOR ENCODING BLOCKS
	for i in range(0,depth+1):
	
		### Downsample once
		if i>0:
			# IF LITE NETWORK
			if lite:
				fm.append( fm[-1]+filters )
			else:
				fm.append( fm[-1]*2 )
			# ELIF LITE NETWORK

			### Downsample once
			conv = conv_layer_resample_v1( inp=conv, \
							filters=fm[-1], \
							ksize=2, \
							stride=2, \
							upsample=False )
			print(conv)

		### Add Residual layers
		for l in range(0,layers):
			conv = ResUnit_v1( conv, filters=fm[-1], ksize=ksize )
			print(conv)


		skips.append(conv)
	# ENDFOR ENCODING BLOCKS
	
	skips = skips[:-1][::-1]
	fm = fm[:-1][::-1]
	
	############################
	###### DECODING BLOCK ######
	############################
	print("\n\nDecoding Blocks")

	# FOR DECODING BLOCKS
	for i in range(depth):
		### Dropout for first 3 layers
		if i<3:
			conv = _tf.keras.layers.Dropout( 0.5 )( conv )
			print(conv)

		### Upsample once
		conv = conv_layer_resample_v1( inp=conv, \
						filters=fm[i], \
						ksize=2, \
						stride=2, \
						upsample=True )
		print(conv)

		### Concatenate with Decoder output
		conv = _tf.concat( [ conv,skips[i] ], axis=3 )
		print(conv)
		
		### Add Residual layers
		for l in range(0,layers):
			conv = ResUnit_v1( conv, filters=fm[i]*2, ksize=ksize )
			print(conv)

		if i==depth-2:
			y_conv_d2 = _tf.keras.layers.Conv2D( filters=num_classes, \
						kernel_size=[1,1], \
						padding="same", \
						use_bias=False, \
						activation=None )( conv )
			print(y_conv_d2)

		if i==depth-3:
			y_conv_d4 = _tf.keras.layers.Conv2D( filters=num_classes, \
						kernel_size=[1,1], \
						padding="same", \
						use_bias=False, \
						activation=None )( conv )
			print(y_conv_d4)
		
	#################################
	###### POST DECODING BLOCK ######
	#################################
	print("\n")

	### Concatenate with Decoder output
	conv = _tf.concat( [ conv,skips[i+1] ], axis=3 )
	print(conv)
		
	### Add Residual layers
	for l in range(0,layers):
		conv = ResUnit_v1( conv, filters=conv.shape.as_list()[-1], ksize=ksize )
		print(conv)

	### Logits Layer
	y_conv = _tf.keras.layers.Conv2D( filters=num_classes, \
				kernel_size=[1,1], \
				padding="same", \
				use_bias=False, \
				activation=None, \
				name="logits" )( conv )
	print(y_conv)
	
	#####################
	####### RETURN ######
	#####################

	return y_conv, y_conv_d2, y_conv_d4
	
# ENDDEF UNET




# DEF UNET_RESNET
def unet_resnet_old( inp_layer,depth=4,ksize=3,filters=32,layers=1,keep_prob=1.0,num_classes=2,lite=False ):

	# block index parameter
	b_idx = 1

	#####################
	#### PROJECTION #####
	#####################
	print("\n")

	# Input shape: [?,x,y,2]
	# Output shape: [?,x,y,f]

	### Project the input layer dimensions from 2 features to f features
	### This ensures the input layer size goes from [?,x,y,2] -> [?,x,y,f]
	block = conv_layer_resample_v1( inp=inp_layer, \
					filters=filters, \
					ksize=1, \
					stride=1, \
					upsample=False )
	print(block)

	################################
	###### PRE ENCODING BLOCK ######
	################################
	print("\n")

	### Add Residual layers
	for l in range(0,layers):
		block = ResUnit_v1( block, \
				filters=filters, \
				ksize=ksize )
		print(block)

	enc_blocks_list = [ block ]
	filter_multipliers = []

	############################
	###### ENCODING BLOCK ######
	############################
	print("\n\nEncoding Blocks")

	# FOR ENCODING BLOCKS
	for b in range( 0,depth+1 ):
		print("\n")
		
		# IF LITE NETWORK
		if lite:
			fm = b+1
			filter_multipliers.extend( [fm] )
		else:
			fm = 2**b
			filter_multipliers.extend( [fm] )
		# ELIF LITE NETWORK
		
		b_idx += 1

		# IF NOT FIRST BLOCK
		if b > 0:
			### Downsample once
			block = conv_layer_resample_v1( inp=block, \
							filters=filters*fm, \
							ksize=2, \
							stride=2, \
							upsample=False )
			print(block)
		# ENDIF NOT FIRST BLOCK

		### Add Residual layers
		for l in range(0,layers):
			block = ResUnit_v1( block, \
					filters=filters*fm, \
					ksize=ksize )
			print(block)

		enc_blocks_list.extend( [block] )
	# ENDFOR ENCODING BLOCKS
			
	############################
	###### DROPOUT BLOCK ######
	############################
	print("\n\nDropout Block")

	### Dropout
	block = _tf.keras.layers.Dropout( 0.5 )( block )
	print(block)

	############################
	###### DECODING BLOCK ######
	############################
	print("\n\nDecoding Blocks")

	# FOR DECODING BLOCKS
	for b in range( depth-1,-1,-1 ):
		print("\n")
	
		# IF LITE NETWORK
		if lite:
			fm = b+1
			filter_multipliers.extend( [fm] )
		else:
			fm = 2**b
			filter_multipliers.extend( [fm] )
		# ELIF LITE NETWORK

		b_idx += 1

		### Upsample once
		block = conv_layer_resample_v1( inp=block, \
						filters=filters*fm, \
						ksize=2, \
						stride=2, \
						upsample=True )
		print(block)

		# Concatenate with corresponding block of the same size
		idx_of_same_size = [i for i,x in enumerate(filter_multipliers) if x==fm][0]
		block = _tf.concat( [ block, enc_blocks_list[idx_of_same_size+1] ], axis=3 )
		print(block)

		### Add Residual layers
		for l in range(0,layers):
			block = ResUnit_v1( block, \
					filters=filters*fm*2, \
					ksize=ksize )
			print(block)

		# IF LAST BLOCK
		if b == 0:
			### Upsample features without upsampling features
			block = conv_layer_resample_v1( inp=block, \
							filters=filters, \
							ksize=ksize, \
							stride=1, \
							upsample=False )
			print(block)
		# IF LAST BLOCK				

		if b==1:
			y_conv_d2 = _tf.keras.layers.Conv2D( filters=num_classes, \
						kernel_size=[1,1], \
						padding="same", \
						use_bias=False, \
						activation=None )( block )
			print(y_conv_d2)

		if b==2:
			y_conv_d4 = _tf.keras.layers.Conv2D( filters=num_classes, \
						kernel_size=[1,1], \
						padding="same", \
						use_bias=False, \
						activation=None )( block )
			print(y_conv_d4)

	# ENDFOR DECODING BLOCKS

	################################
	###### POST DECODING BLOCK #####
	################################
	print("\n")

	b_idx += 1

	# Concatenate with block 0
	# Makes output dimensions [?,x,y,f*2]
	block = _tf.concat( [ block, enc_blocks_list[0] ], axis=3 )
	print(block)

	### Add Residual layers
	for l in range(0,layers):
		block = ResUnit_v1( block, \
				filters=filters*2, \
				ksize=ksize )
		print(block)

	#####################
	### FINAL LAYERS ####
	#####################
	print("\n")

	### Logits Layer
	y_conv = _tf.keras.layers.Conv2D( filters=num_classes, \
				kernel_size=[1,1], \
				padding="same", \
				use_bias=False, \
				activation=None, \
				name="logits" )( block )
	print(y_conv)
		
	#####################
	####### RETURN ######
	#####################

	return y_conv, y_conv_d2, y_conv_d4
	
# ENDDEF UNET_RESNET
