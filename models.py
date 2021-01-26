"""
Created on Mon Jul  3 14:27:26 2017

@author: jimit
"""
################################################ DECLARATIONS ################################################
__author__ 	= 'Jimit Doshi'
__EXEC_NAME__ 	= "model"

import tensorflow as _tf

from unet_vanilla 	import unet_vanilla, unet_vanilla_bn
from unet_resinc 	import unet_resinc
from unet_resnet 	import unet_resnet

################################################ FUNCTIONS ################################################



# DEF MODEL
def create_model( 	num_classes=2, \
			arch='UNet_vanilla_bn', \
			filters=64, \
			depth=4, \
			num_modalities=1, \
			layers=1, \
			lite=False ):
			
	#####################
	#### INPUT LAYER ####
	#####################
	print("\n")
		
				
#	# Defining placeholders
	img_slice = _tf.keras.Input( dtype=_tf.float32, \
					shape=[None,None,num_modalities], \
					name="img_slice" )
	print(img_slice)
		
	#####################
	##### MAKE UNET #####
	#####################
	print("\n")

	# IF ARCHITECTURE
	if arch == 'UNet_vanilla':
		arch_func = unet_vanilla

	elif arch == 'UNet_vanilla_bn':
		arch_func = unet_vanilla_bn
					
	elif arch == 'ResInc':
		arch_func = unet_resinc

	elif arch == 'ResNet':
		arch_func = unet_resnet

	#ENDIF

	# logits
	logits_d1, logits_d2, logits_d4 = arch_func( inp_layer=img_slice, \
							depth=depth, \
							ksize=3, \
							filters=filters, \
							layers=layers, \
							num_classes=num_classes, \
							lite=lite )
							
	#######################
	##### PREDICTIONS #####
	#######################
	print("\n")

	### Get predictions
	probs_d1 = _tf.nn.softmax( logits_d1, axis=3, name="probabilities" ); print(probs_d1)
	probs_d2 = _tf.nn.softmax( logits_d2, axis=3 ); print(probs_d2)
	probs_d4 = _tf.nn.softmax( logits_d4, axis=3 ); print(probs_d4)
	preds_d1 = _tf.argmax( input=logits_d1, axis=3, name="predictions" ); print(preds_d1)
	preds_d2 = _tf.argmax( input=logits_d2, axis=3 ); print(preds_d2)
	preds_d4 = _tf.argmax( input=logits_d4, axis=3 ); print(preds_d4)
	
	##################
	##### RETURN #####
	##################
	
	return _tf.keras.Model( inputs=[ img_slice, ], \
				outputs=[ preds_d1,probs_d1,\
					preds_d2,probs_d2,\
					preds_d4,probs_d4 ], \
				name="final_model" )
# ENDDEF MODEL
