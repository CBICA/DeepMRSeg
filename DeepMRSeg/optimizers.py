
################################################ DECLARATIONS ################################################
__author__ 	= 'Jimit Doshi'
__EXEC_NAME__ 	= "optimizers"

import tensorflow as _tf


################################################ FUNCTIONS ################################################

# DEF
def getAdamOpt( lr=0.001, epsilon=0.1, beta1=0.9, beta2=0.999, ):
	return _tf.keras.optimizers.Adam( lr, beta1, beta2, epsilon, \
					name="AdamOptimizer" )
# ENDDEF

# DEF
def getRMSOpt( lr ):
	return _tf.train.RMSPropOptimizer( learning_rate=lr, \
					decay=0.9, 
					epsilon=1.0, \
					name="RMSPropOptimizer" )
# ENDDEF

# DEF
def getSGDOpt( lr ):
	return _tf.train.GradientDescentOptimizer( learning_rate=lr, \
					name="GradientDescentOptimizer" )
# ENDDEF

# DEF
def getMomentumOpt( lr ):
	return _tf.train.MomentumOptimizer( learning_rate=lr, \
					momentum=0.99, \
					name="MomentumOptimizer" )
# ENDDEF
