

################################################ DECLARATIONS ################################################
__author__	 = 'Jimit Doshi'
__EXEC_NAME__	 = "deepmrseg_train"

import os as _os
import sys as _sys
import json as _json
import time as _time
import signal as _signal
import tensorflow as _tf
import numpy as _np
import platform as _platform

_sys.path.append( _os.path.dirname( _sys.argv[0] ) )

from . import losses

from .data_io import check_files, extract_pkl
from .models import create_model
from .data_augmentation import data_reader
from .optimizers import get_adam_opt, get_rms_opt, get_sgd_opt, get_momentum_opt
from .layers import get_onehot

from . import pythonUtilities
from . import utils

################################################ FUNCTIONS ################################################


#DEF ARGPARSER
def read_flags():
	"""Parses args and returns the flags and parser."""
	### Import modules
	import argparse as _argparse

	parser = _argparse.ArgumentParser( formatter_class=_argparse.ArgumentDefaultsHelpFormatter )

#	CONFIG FILE
#	===========
	configArgs = parser.add_argument_group( 'CONFIG FILE' )
	configArgs.add_argument( "--config", default=None, type=str, \
				help="absolute path to the config file containing the parameters in a JSON format")

	# Only parse for "--config"
	configarg,_ = parser.parse_known_args()

#	INPUT LIST
#	==========
	inputArgs = parser.add_argument_group( 'REQUIRED INPUT ARGS' )
	inputArgs.add_argument( "--sList", default=None, type=str, required=True, \
				help="absolute path to the subject list for training")
	inputArgs.add_argument( "--idCol", default=None, type=str, \
				help="id column name (e.g.: 'ID')" )
	inputArgs.add_argument( "--labCol", default=None, type=str, \
				help="label column name (e.g.: 'wml')")
	inputArgs.add_argument( "--refMod", default=None, type=str, \
				help="referene modality column name (e.g.: 'FL')")
	inputArgs.add_argument( "--otherMods", default=None, type=str, \
				help="other modalities to use (e.g.: 'T1,T2')" )
	inputArgs.add_argument( "--roi", default=None, \
				help="absolute path to the ROI csv containing the ROIs to be \
					considered and their respective indices. If not provided, \
					the ROIs are assumed to be in the range( 0,num_classes-1 )")

#	DIRECTORIES
#	===========
	dirArgs = parser.add_argument_group( 'DIRECTORIES' )
	dirArgs.add_argument( "--mdlDir", default="model", type=str, \
				help="absolute path to the directory where the model should be saved")
	dirArgs.add_argument( "--ckptDir", default=None, type=str, \
				help="absolute path to the directory where model checkpoint is stored" )
	dirArgs.add_argument( "--tmpDir", default=None, type=str, \
				help="absolute path to the temporary directory" )

#	INPUT IMAGE
#	===========
	inpArgs = parser.add_argument_group( 'INPUT IMAGE' )
	inpArgs.add_argument( "--num_classes", default=2, type=int, \
				help="number of classes to be considered in the input")
	inpArgs.add_argument( "--label_balance", default=1, type=int, \
				help="weights to be used for positive/foreground labels" )
	inpArgs.add_argument( "--rescale", default='norm', type=str, \
				help="rescale method, choose from { minmax, norm }")
	inpArgs.add_argument( "--xy_width", default=320, type=int, \
				help="xy dimensions of the input patches. \
					Determines how much each slice needs to be padded in the xy dimension. \
					Should be divisible by 2**depth.")
	inpArgs.add_argument( "--ressize", default=1, type=float, \
				help="isotropic voxel size for the input images \
					input images will be resampled to this resolution" )
	inpArgs.add_argument( "--reorient", default='LPS', type=str, \
				help="reorient the training images to match the provided orientation (in radiology convention)" )

#	TRAINING
#	========
	trainArgs = parser.add_argument_group( 'TRAINING' )
	trainArgs.add_argument( "--arch", default='ResNet', type=str, \
				help="UNet architecture to be used choose from { UNet_vanilla, UNet_vanilla_bn, ResNet, ResInc, }" )
	trainArgs.add_argument( "--num_epochs", default=10, type=int, \
				help="number of training epochs" )
	trainArgs.add_argument( "--min_epochs", default=5, type=int, \
				help="minimum number of training epochs to run before evaluating \
					early stopping criteria")
	trainArgs.add_argument( "--optimizer", default='RMSProp', type=str, \
				help="optimizer to be used for the first part of the training \
					 choose from { 'Adam', 'RMSProp', 'SGD', 'Momentum' }")
	trainArgs.add_argument( "--learning_rate", default=0.05, type=float, \
				help="initial learning rate to be used" )
	trainArgs.add_argument( "--lr_sch", default='EXP', type=str, \
				help="learning rate schedule \
					choose from { EXP, PLAT }" )
	trainArgs.add_argument( "--decay", default=0.9, type=float, \
				help="exponential_decay for the learning rate" )
	trainArgs.add_argument( "--patience", default=5, type=int, \
				help="number of epochs to wait without improvement" )
	trainArgs.add_argument( "--gamma", default=1, type=float, \
				help="modulating factor for the losses" )
	trainArgs.add_argument( "--max_to_keep", default=5, type=int, \
				help="number of best performing models to keep")
	trainArgs.add_argument( "--batch", default=8, type=int, \
				help="batch size" )
	trainArgs.add_argument( "--filters", default=16, type=int, \
				help="number of filters in each layer")
	trainArgs.add_argument( "--depth", default=4, type=int, \
				help="depth of the encoding/decoding architecture")
	trainArgs.add_argument( "--layers", default=2, type=int, \
				help="number of layers in each U-Net block" )
	trainArgs.add_argument( "--label_smoothing", default=0.0, type=float, \
				help="label smoothing to be applied to the training labels" )
	trainArgs.add_argument( "--deep_supervision", default=False, action="store_true", \
				help="use deep supervision of the network")
	trainArgs.add_argument( "--lite", default=False, action="store_true", \
				help="use the lite version of the network")
	trainArgs.add_argument( "--norm", default='batch', type=str, \
				help="normalization layer to use \
					 choose from { 'batch', 'instance' }")
	trainArgs.add_argument( "--alpha", default=50, type=float, \
				help="weighted loss of the form \
					loss = alpha*dice_loss + (100-alpha)*(mae+bce)")

#	MISC
#	====
	miscArgs = parser.add_argument_group( 'MISCELLANEOUS' )
	miscArgs.add_argument( "--verb", default=1, type=int, \
				help="verbosity")
	miscArgs.add_argument( "--nJobs", default=None, type=int, \
				help="number of jobs/threads" )
	miscArgs.add_argument( "--summary", default=False, action="store_true", \
				help="write out summaries to mdlDir to be viewed using tensorboard" )

#	FLAGS
#	=====
	### Read config file first, if provided
	if configarg.config:
		# Read the JSON config file
		with open(configarg.config) as f:
			configflags = _json.load(f)

		# Set args from the config file as defaults
		parser.set_defaults( **configflags )

	### Read remaining args from CLI and overwrite the defaults
	flags = parser.parse_args()

	### Return flags and parser
	return flags, parser

#ENDDEF ARGPARSER

### Define signal trap function
#DEF
def signal_handler(signal, frame):
	"""Signal handler to catch keyboard interrupt signals.

	Args:
		signal
		frame
	"""
	print('Program interrupt signal received! Aborting operations ...')
	_sys.exit(0)
#ENDDEF

############### CLASSES ###########

#CLASS TRAIN
class Train(object):
	"""A class to train a DeepMRSeg model.
	...
	Attributes:
		model: path
			DeepMRSeg model
		strategy: tf.distribute.Strategy() object
			Distribution strategy in use.
		optimizer: tf.keras.optimizers object
			Optimizer to use.
		num_gpu: int
			Number of GPUs available.
		FLAGS: Other args
	"""

	#DEF INIT
	def __init__( self, model, strategy, optimizer, num_gpu, FLAGS ):
		"""Train class constructor to initialize Train object.

		Args:
			model: DeepMRSeg model.
			strategy: Distribution strategy in use.
			optimizer: Optimizer to use.
			num_gpu: Number of GPUs available.
			FLAGS: Other args
		"""
		### Define Variables
		self.model = model
		self.strategy = strategy
		self.optimizer = optimizer
		self.num_gpu = num_gpu
		self.num_epochs = FLAGS.num_epochs
		self.min_epochs = FLAGS.min_epochs
		self.batch_size = FLAGS.batch
		self.learning_rate = FLAGS.learning_rate
		self.decay = FLAGS.decay
		self.lr_sch = FLAGS.lr_sch
		self.patience = FLAGS.patience
		self.gamma = FLAGS.gamma
		self.alpha = FLAGS.alpha
		self.global_batch_size = self.batch_size * self.num_gpu
		self.label_smoothing = FLAGS.label_smoothing
		self.deep_supervision = FLAGS.deep_supervision
		self.xy_width = FLAGS.xy_width
		self.num_classes = FLAGS.num_classes
		self.mdlDir = FLAGS.mdlDir
		self.max_to_keep = FLAGS.max_to_keep
		self.summary = FLAGS.summary

		### Define Metrics
		self.iou_train = _tf.keras.metrics.MeanIoU( num_classes=self.num_classes )
		self.iou_val = _tf.keras.metrics.MeanIoU( num_classes=self.num_classes )

		### Define Loss Aggregators
		self.epoch_train_loss_avg = _tf.keras.metrics.Mean()
		self.epoch_train_ioul_avg = _tf.keras.metrics.Mean()
		self.epoch_train_mael_avg = _tf.keras.metrics.Mean()
		self.epoch_train_bcel_avg = _tf.keras.metrics.Mean()

		self.epoch_val_loss_avg = _tf.keras.metrics.Mean()
		self.epoch_val_ioul_avg = _tf.keras.metrics.Mean()
		self.epoch_val_mael_avg = _tf.keras.metrics.Mean()
		self.epoch_val_bcel_avg = _tf.keras.metrics.Mean()

	#ENDDEF INIT

	# DEF SET_LR
	def set_lr( self,e,plat ):
		"""Learning rate scheduler."""
		if not hasattr( self.optimizer, "lr" ):
			raise ValueError('Optimizer must have a "lr" attribute.')
			
		# Get the current learning rate from model's optimizer.
		lr = _tf.keras.backend.get_value(self.optimizer.lr)
		
		#IF
		if self.lr_sch == 'EXP':
			new_lr = self.learning_rate * self.decay ** (e-1)
		elif self.lr_sch == 'PLAT':
			if plat >= self.patience/2:
				new_lr = lr/2.
				plat = 0
			else:
				new_lr = lr
		#ENDIF
		
		# Set the value back to the optimizer before this epoch starts
		_tf.keras.backend.set_value( self.optimizer.lr, new_lr )

		return _tf.keras.backend.get_value(self.optimizer.lr), plat
	# ENDDEF SET_LR
		
	# DEF COMPUTE_LOSS
	@_tf.function
	def compute_loss( self,one_h,probs_d1,probs_d2,probs_d4 ):

		total_loss, total_loss_d1, \
		iou_d1, mae_d1, bce_d1 = losses.get_combo_loss( one_h,probs_d1,probs_d2,probs_d4,\
								self.gamma,self.deep_supervision,\
								self.xy_width,self.alpha )
		
		total_loss_rep = _tf.nn.compute_average_loss( total_loss, \
					global_batch_size=self.global_batch_size )

		self.epoch_train_loss_avg.update_state( total_loss_d1 )
		self.epoch_train_ioul_avg.update_state( iou_d1 )
		self.epoch_train_mael_avg.update_state( mae_d1 )
		self.epoch_train_bcel_avg.update_state( bce_d1 )

		return total_loss_rep
	# ENDDEF COMPUTE_LOSS

	# DEF TRAIN_STEP
	@_tf.function
	def train_step( self,inputs ):
		
		img,lab = inputs
		oh_d1 = get_onehot( lab,self.label_smoothing,self.xy_width,self.num_classes )
		
		#WITH GRADIENTTAPE
		with _tf.GradientTape() as tape:
			# Run model
			preds_d1,probs_d1, \
			_,probs_d2, \
			_,probs_d4 = self.model( img, training=True )
			
			# Calculate losses
			total_loss_rep = self.compute_loss( oh_d1,probs_d1,probs_d2,probs_d4 )
		#ENDWITH GRADIENTTAPE

		grads = tape.gradient( total_loss_rep, self.model.trainable_weights )
		self.optimizer.apply_gradients( zip(grads, self.model.trainable_weights) )

		self.iou_train.update_state( lab,preds_d1 )

		return total_loss_rep
	# ENDDEF TRAIN_STEP

	# DEF VAL_STEP
	@_tf.function
	def val_step( self,inputs ):
		
		img,lab = inputs
		oh_d1 = get_onehot( lab,0,self.xy_width,self.num_classes )
		
		# Run model
		preds_d1,probs_d1,_,_,_,_ = self.model( img, training=False )
		
		# Calculate losses
		total_loss_d1, iou_d1, mae_d1, bce_d1 = losses.combo_loss( oh_d1,probs_d1,1,50 )
		
		self.iou_val.update_state( lab,preds_d1 )

		self.epoch_val_loss_avg.update_state( total_loss_d1 )
		self.epoch_val_ioul_avg.update_state( iou_d1 )
		self.epoch_val_mael_avg.update_state( mae_d1 )
		self.epoch_val_bcel_avg.update_state( bce_d1 )
	# ENDDEF VAL_STEP

	# DEF CUSTOM_LOOP
	def custom_loop( self, train_dist_dataset, val_dist_dataset, strategy ):
		"""Custom training and validation loop.

		Args:
		  train_dist_dataset: Training dataset created using strategy.
		  val_dist_dataset: Validation dataset created using strategy.
		  strategy: Distribution strategy.
		"""
		### Import modules
		import shutil as _shutil

		#DEF DIST_TRAIN_EPOCH
		@_tf.function
		def distributed_train_epoch(ds):
			strategy.run( self.train_step, args=(ds,) )
		#ENDDEF DIST_TRAIN_EPOCH

		#DEF DIST_VAL_EPOCH
		@_tf.function
		def distributed_val_epoch(ds):
			strategy.run( self.val_step, args=(ds,) )
		#ENDDEF DIST_VAL_EPOCH

		#DEF
		def pop_extend( arr,val ):
			arr_list = arr.tolist()
			if len(arr) >= self.patience:
				arr_list.pop(0)
			arr_list.extend( [val] )
			
			return _np.array(arr_list)
		#ENDDEF

		# Setup summary writers
		#IF
		if self.summary:
			import datetime as _datetime
			current_time = _datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
			train_log_dir = self.mdlDir + '/summaries/gradient_tape/' + current_time + '/train'
			val_log_dir = self.mdlDir + '/summaries/gradient_tape/' + current_time + '/validation'
			train_summary_writer = _tf.summary.create_file_writer(train_log_dir)
			val_summary_writer = _tf.summary.create_file_writer(val_log_dir)
		#ENDIF


		### Training the network
		print("\nTraining the network...\n")
		_sys.stdout.flush()
	
		# Get start time
		est = _time.time()
		i = 0
		
		# Min loss at each epoch
		loss_min = _np.ones( self.max_to_keep ) * 100000
		loss_min_ind = _np.zeros( self.max_to_keep )

		# Accuracy metric moving average
		accu_train_recent = _np.array( [] )
		accu_val_recent = _np.array( [] )

		last_saved_mdl_counter = 0
		last_lr_change_counter = 0
		current_lr = self.learning_rate

		# FOR EACH EPOCH
		for epoch in range( 1,self.num_epochs+1 ):

			### Stop training if loss_cov has reached its limit
			#IF
			if epoch > self.min_epochs:
				if last_saved_mdl_counter >= self.patience:
					print("Early stopping criteria met. Halt training...")
					break
			#ENDIF

			### Set the learning rate based on the chosen schedule
			current_lr,last_lr_change_counter = self.set_lr( epoch,last_lr_change_counter )

			# Reset training metrics at the end of each epoch
			self.iou_train.reset_states(); self.iou_val.reset_states()
			self.epoch_train_loss_avg.reset_states(); self.epoch_val_loss_avg.reset_states()
			self.epoch_train_ioul_avg.reset_states(); self.epoch_val_ioul_avg.reset_states()
			self.epoch_train_mael_avg.reset_states(); self.epoch_val_mael_avg.reset_states()
			self.epoch_train_bcel_avg.reset_states(); self.epoch_val_bcel_avg.reset_states()
			

			# Train
			# FOR EACH TRAIN BATCH
			for one_batch in train_dist_dataset:
				distributed_train_epoch( one_batch )

				# IF DISPLAY TRAINING METRICS
				if i%1000 == 0:
					timeperiter = (_time.time()-est) / (i+1) * 1000 / 60.
					print( "\t\titerations : %d, time per 1000 iterations: %.2f mins" % \
									( i, timeperiter ) )

					print( "\t\t\t training metrics \t: mIOU: %.4f, Loss: %.4f (%.4f,%.4f,%.4f)" \
							% ( self.iou_train.result(), \
							self.epoch_train_loss_avg.result(), \
							self.epoch_train_ioul_avg.result(), \
							self.epoch_train_mael_avg.result(), \
							self.epoch_train_bcel_avg.result() ) )

					_sys.stdout.flush()
				# ENDIF
				i += 1
			# ENDFOR EACH TRAIN BATCH
			
			### Checkpoint model
			self.model.save( _os.path.join( self.mdlDir \
						+ '/checkpoint/model' ), \
					overwrite=True, \
					include_optimizer=True )

			# Validate
			# FOR EACH VAL BATCH
			for one_batch in val_dist_dataset:
				distributed_val_epoch( one_batch )
			# ENDFOR EACH VAL BATCH

			# Check if the latest model is in the top models
			# IF
			if ( self.epoch_val_loss_avg.result() < loss_min[0] ):

				mdl_to_del = loss_min_ind[0]
				mdl_to_del_path = _os.path.join( self.mdlDir \
							+ '/bestmodels/model-' \
							+ str( int(mdl_to_del) ) )
							
				if _os.path.isdir( mdl_to_del_path ):
					_shutil.rmtree( mdl_to_del_path )
			
				loss_min[0] = self.epoch_val_loss_avg.result()
				loss_min_ind[0] = epoch
				loss_min_ind = loss_min_ind[ _np.argsort(loss_min)[::-1] ]
				loss_min = loss_min[ _np.argsort(loss_min)[::-1] ]
			
				last_saved_mdl_counter = 0
				last_lr_change_counter = 0
			
				self.model.save( _os.path.join( self.mdlDir \
							+ '/bestmodels/model-' \
							+ str(epoch) ), \
						overwrite=True, \
						include_optimizer=False )
			else:
				if epoch >= (self.min_epochs):
					last_saved_mdl_counter += 1
				
				last_lr_change_counter += 1
			# ENDIF

			accu_train_recent = pop_extend( accu_train_recent,self.iou_train.result() )
			accu_val_recent = pop_extend( accu_val_recent,self.iou_val.result() )
			
			timeperepoch = (_time.time()-est) / epoch / 60
			print( "\n\tepoch : %d, time/epoch: %.2f mins, learning rates: %.1E" \
					% ( epoch, timeperepoch, current_lr ) )

			print( "\t\t training metrics \t: mIOU: %.4f (%.4f), Loss: %.4f (%.4f,%.4f,%.4f)" \
					% ( self.iou_train.result(), accu_train_recent.mean(), \
					self.epoch_train_loss_avg.result(), \
					self.epoch_train_ioul_avg.result(), \
					self.epoch_train_mael_avg.result(), \
					self.epoch_train_bcel_avg.result() ) )

			print( "\t\t validation metrics \t: mIOU: %.4f (%.4f), Loss: %.4f (%.4f,%.4f,%.4f) ( %.4f, %.4f )\n" \
					% ( self.iou_val.result(), accu_val_recent.mean(), \
					self.epoch_val_loss_avg.result(), \
					self.epoch_val_ioul_avg.result(), \
					self.epoch_val_mael_avg.result(), \
					self.epoch_val_bcel_avg.result(), \
					loss_min.min(), loss_min.max() ) )

			_sys.stdout.flush()

			### Write summaries
			#IF
			if self.summary:
				### Training
				#WITH
				with train_summary_writer.as_default():
					### losses
					_tf.summary.scalar( 'total loss', self.epoch_train_loss_avg.result(), step=epoch )
					_tf.summary.scalar( 'iou loss', self.epoch_train_ioul_avg.result(), step=epoch )
					_tf.summary.scalar( 'mae loss', self.epoch_train_mael_avg.result(), step=epoch )
					_tf.summary.scalar( 'bce loss', self.epoch_train_bcel_avg.result(), step=epoch )

					### metrics
					_tf.summary.scalar( 'mIOU', self.iou_train.result(), step=epoch )

					### params
					_tf.summary.scalar( 'learning rate', current_lr, step=epoch )
				#ENDWITH

				### Validation
				#WITH
				with val_summary_writer.as_default():
					### losses
					_tf.summary.scalar( 'total loss', self.epoch_val_loss_avg.result(), step=epoch )
					_tf.summary.scalar( 'iou loss', self.epoch_val_ioul_avg.result(), step=epoch )
					_tf.summary.scalar( 'mae loss', self.epoch_val_mael_avg.result(), step=epoch )
					_tf.summary.scalar( 'bce loss', self.epoch_val_bcel_avg.result(), step=epoch )

					### metrics
					_tf.summary.scalar( 'mIOU', self.iou_val.result(), step=epoch )

					### images
					# Log the confusion matrix as an image summary.
					figure = utils.plot_confusion_matrix( self.iou_val.total_cm.numpy(), \
										class_names=_np.arange(self.num_classes) )
					cm_image = utils.plot_to_image(figure)
					_tf.summary.image( "Confusion Matrix", cm_image, step=epoch )
				#ENDWITH
			#ENDIF
		# ENDFOR EACH EPOCH
   	# ENDDEF CUSTOM_LOOP
#ENDCLASS TRAIN


############## MAIN ##############
#DEF
def _main():

	### init argv
	argv = _sys.argv
	
	### Timestamps
	startTime = _time.asctime()
	startTimeStamp = _time.time()

	### Print startTimeStamp
	print("\nHostname	: " + str( _os.getenv("HOSTNAME") ))
	print("Start time	: " + str( startTime ))

	_sys.stdout.flush()

	### Specifying the trap signal
	if _platform.system() != 'Windows':
		_signal.signal( _signal.SIGHUP, signal_handler )
	_signal.signal( _signal.SIGINT, signal_handler )
	_signal.signal( _signal.SIGTERM, signal_handler )

	### Read command line args
	print("\nParsing args    : %s\n" % (argv[ 1: ]) )
	FLAGS,parser = read_flags()
	print(FLAGS)

	### Check the number of arguments
	if len( argv ) == 1:
		parser.print_help( _sys.stderr )
		_sys.exit(1)

	### Load modules
	print("\nLoading modules")

	_sys.stdout.flush()

	import csv as _csv
	import tempfile as _tempfile
	import shutil as _shutil
	from random import shuffle as _shuffle
	from concurrent.futures import ThreadPoolExecutor as _TPE

	_sys.stdout.flush()

	### Sanity checks on the provided arguments
	# Check if input files provided exist
	# FOR
	for f in FLAGS.sList, FLAGS.roi:
		pythonUtilities.check_file( f )
	# ENDFOR
	
	# Check if xy_width matches the depth
	# IF
	if FLAGS.xy_width % 2**FLAGS.depth != 0:
		print( "ERROR: The xy_width (%d) is not divisible by %d" % (FLAGS.xy_width,2**FLAGS.depth) )
		_sys.exit(1)
	# ENDIF
	
	# Create temp dir, if needed
	#IF
	if FLAGS.tmpDir:
		user_defined_tmpdir = True
		tmpDir = FLAGS.tmpDir
		if not _os.path.isdir(FLAGS.tmpDir):
			_os.makedirs( FLAGS.tmpDir )
	else:
		user_defined_tmpdir = False
		tmpDir = _tempfile.mkdtemp( prefix='deepmrseg_train_' )
	#ENDIF
	
	# if nJobs not defined
	#IF
	if FLAGS.nJobs:
		nJobs = FLAGS.nJobs
	else:
		nJobs = _os.cpu_count()
	#ENDIF

	# if otherMods defined
	#IF
	if FLAGS.otherMods:
		otherMods = list(map( str, FLAGS.otherMods.split(',') ))
		num_modalities = len(otherMods) + 1
	else:
		otherMods = None
		num_modalities = 1
	#ENDIF

	# if gamma < 0
	#IF
	if FLAGS.gamma < 0:
		raise ValueError("Value of gamma should be greater than or equal to zero")
	#ENDIF

	### Print parsed args
	print( "\nPackage Versions" )
	print( "python \t\t: %s" % (_platform.python_version()) )
	print( "tensorflow \t: %s" % (_tf.__version__) )
	print( "numpy \t\t: %s" % (_np.__version__) )
	
	print( "\nFile List \t: %s" % (FLAGS.sList) )
	print( "ID Column \t: %s" % (FLAGS.idCol) )
	print( "Label Column \t: %s" % (FLAGS.labCol) )
	print( "Ref Modality \t: %s" % (FLAGS.refMod) )
	print( "Other Mods \t: %s" % (otherMods) )
	print( "Num of Mods \t: %d" % (num_modalities) )
	print( "ROI csv \t: %s" % (FLAGS.roi) )
	
	print("\nModel Dir \t: %s" % (FLAGS.mdlDir))
	print("Checkpoint Dir \t: %s" % (FLAGS.ckptDir))
	print("Temp Dir \t: %s" % (tmpDir))

	print("\nNum of Classes \t: %d" % (FLAGS.num_classes))
	print("FG Label Wt \t: %d" % (FLAGS.label_balance))
	print("Rescale Method \t: %s" % (FLAGS.rescale))
	print("XY width \t: %d" % (FLAGS.xy_width))
	print("Voxel Size \t: %f" % (FLAGS.ressize))
	print("Orientation \t: %s" % (FLAGS.reorient))
	
	print("\nArchitecture \t: %s" % (FLAGS.arch))
	print("Num of Epochs \t: %d" % (FLAGS.num_epochs))
	print("Min Epochs \t: %d" % (FLAGS.min_epochs))
	print("Init Optimizer \t: %s" % (FLAGS.optimizer))
	print("Learning Rate \t: %s" % (FLAGS.learning_rate))
	print("LR Schedule \t: %s" % (FLAGS.lr_sch))
	print("Decay Factor \t: %s" % (FLAGS.decay))
	print("Loss Weight \t: %s" % (FLAGS.alpha))
	print("Gamma Factor \t: %s" % (FLAGS.gamma))
	print("Batch Size \t: %d" % (FLAGS.batch))
	print("Filter Num \t: %d" % (FLAGS.filters))
	print("Depth of Arch \t: %d" % (FLAGS.depth))
	print("Layers \t\t: %d" % (FLAGS.layers))
	print("Max to Keep \t: %d" % (FLAGS.max_to_keep))
	print("Label Smoothing : %s" % (str(FLAGS.label_smoothing)))
	print("Deep Supervision: %s" % (str(FLAGS.deep_supervision)))
	print("Lite Verion \t: %s" % (str(FLAGS.lite)))
	print("Patience Param \t: %d" % (FLAGS.patience))
	print("Normalization \t: %s\n" % (FLAGS.norm))

	### Create model dir
	if not _os.path.isdir( FLAGS.mdlDir ):
		_os.makedirs( FLAGS.mdlDir )

	### Create a config file for testing
	test_config = {}
	#FOR KEYS
	for k in [ 'num_classes', 'reorient', 'rescale', 'ressize', 'roi', 'xy_width' ]:
		test_config[k] = FLAGS.__dict__[k]
	#ENDFOR KEYS
	
	# dump to json file
	test_config_json = _json.dumps( test_config,sort_keys=True,indent=4 )
	#WITH
	with open( _os.path.join( FLAGS.mdlDir + '/test.cfg' ), "w" ) as outfile:
		outfile.write( test_config_json )
	#ENDWITH

	### Read subject list file
	print("Reading the input subject list and running sanity checks on the files")
	_sys.stdout.flush()

	# create training and validation subject lists
	train_sublist = []
	val_sublist = []
	all_sublist = []

	### Multi-thread the loading of images
	#WITH
	with _TPE( max_workers=nJobs ) as executor:
		#WITH
		with open(FLAGS.sList) as f:
			reader = _csv.DictReader( f )
	
			#FOR
			for row in reader:
				all_sublist.extend( [ row[FLAGS.idCol] ] )
		
				# Get files for other modalities
				otherModsFileList = []
				if FLAGS.otherMods:
					for mod in otherMods:
						otherModsFileList.extend( [ row[mod] ] )

				executor.submit( check_files, \
						refImg=row[FLAGS.refMod], \
						labImg=row[FLAGS.labCol], \
						otherImg=otherModsFileList )
			#ENDFOR
		#ENDWITH
	#ENDWITH

	### Split to train-validation
	print("Splitting the training list into training and validation")
	_sys.stdout.flush()

#	******************************************
#	* COMMENTED OUT ONLY FOR EXPERIMENTATION *
#	******************************************
	# Randomize list
	_shuffle( all_sublist )

	# Split into training and validation lists
	p = _np.int( len( all_sublist ) * 0.2 )
	val_sublist = all_sublist[ 0:p ]
	train_sublist = all_sublist[ p: ]

	print( "\nTraining subjects: " )
	for sub in train_sublist:
		print( '\t%s' % (sub) )
		
	print( "\nValidation subjects: " )
	for sub in val_sublist:
		print( '\t%s' % (sub) )
	
	### Loading data
	print("\nLoading data to ", tmpDir)
	_sys.stdout.flush()

	train_filenames = []
	val_filenames = []

	### Multi-thread the loading of images
	#WITH
	with _TPE( max_workers=nJobs ) as executor:

		# FOR ALL TRAINING SUBJECTS
		for sub in all_sublist:
			
			# IF TRAINING
			if sub in train_sublist:
				pref = tmpDir + '/' + sub + '.tfr'
				train_filenames.extend( [pref] )
			elif sub in val_sublist:
				pref = tmpDir + '/' + sub + '.tfr'
				val_filenames.extend( [pref] )
			# ELIF TRAINING
			
			### If tfrecord doesn't exist, create it
			#IF
			if not _os.path.isfile( pref ):
				executor.submit( extract_pkl, \
						subListFile=FLAGS.sList, \
						idcolumn=FLAGS.idCol, \
						labCol=FLAGS.labCol, \
						refMod=FLAGS.refMod, \
						otherMods=otherMods, \
						num_modalities=num_modalities, \
						subjectlist=[sub], \
						roicsv=FLAGS.roi, \
						out_path=pref, \
						rescalemethod=FLAGS.rescale, \
						xy_width=FLAGS.xy_width, \
						pos_label_balance=FLAGS.label_balance, \
						ressize=FLAGS.ressize, \
						orient=FLAGS.reorient )
			#ELIF
		#ENDFOR ALL TRAINING SUBJECTS
	#ENDWITH

	
	#####################################
	#### CREATE DISTRIBUTE STRATEGY #####
	#####################################
	print("\n")
	print("Defining distribution strategy...")

	# If the list of devices is not specified in the
	# `tf.distribute.MirroredStrategy` constructor, it will be auto-detected.
	strategy = _tf.distribute.MirroredStrategy()
	print( "\nSTRATEGY: %s" % (strategy) )
	
	
	NUM_GPU_DEVICES = strategy.num_replicas_in_sync
	print ( 'NUM_GPU_DEVICES: {}'.format(NUM_GPU_DEVICES) )
	gpus = _tf.config.experimental.list_physical_devices('GPU')
	print( gpus )

	BATCH_SIZE_PER_REPLICA = FLAGS.batch
	GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * NUM_GPU_DEVICES
	print("BATCH_SIZE_PER_REPLICA: %d" % (BATCH_SIZE_PER_REPLICA) )
	print("GLOBAL_BATCH_SIZE: %d" % (GLOBAL_BATCH_SIZE) )

	###########################
	#### PREPARE DATASETS #####
	###########################
	print("\n")

	#WITH CPU DEVICE
	with _tf.device( '/cpu:0' ):

		#DEF
		@_tf.function
		def tfrecordreader( serialized_example ):

			feature = {	 'image': _tf.io.FixedLenFeature( [], _tf.string ), \
					 'label': _tf.io.FixedLenFeature( [], _tf.string ) }

			# Decode the record read by the reader
			features = _tf.io.parse_single_example( serialized_example, \
							 features=feature )

			# Convert the data from string back to the numbers
			image = _tf.io.decode_raw( features['image'], _tf.float32 )
			label = _tf.io.decode_raw( features['label'], _tf.int64 )

			# Reshape image data into the original shape
			image = _tf.reshape( image, \
				 list( [FLAGS.xy_width,FLAGS.xy_width,num_modalities] ) )
			label = _tf.reshape( label, \
				list( [FLAGS.xy_width,FLAGS.xy_width,1] ) )

			return image, label
		#ENDDEF

		### TRAINING DATASET
		print("\n")
		train_ds = data_reader( filenames=train_filenames, \
						reader_func=tfrecordreader, \
						batch_size=GLOBAL_BATCH_SIZE, \
						mode=_tf.estimator.ModeKeys.TRAIN )

	
		### VALIDATION DATASET
		print("\n")
		val_ds = data_reader( filenames=val_filenames, \
						reader_func=tfrecordreader, \
						batch_size=GLOBAL_BATCH_SIZE*4, \
						mode=_tf.estimator.ModeKeys.EVAL )
						
	#ENDWITH CPU DEVICE


	####################################
	### WITHIN DISTRIBUTED STRATEGY ####
	####################################
	print("\nWithin the Distributed Strategy...\n")
	_sys.stdout.flush()

	#WITH STRATEGY
	with strategy.scope():

		### Create the model
		print("\nDefining the network...\n")
		model = create_model(	num_classes=FLAGS.num_classes, \
					arch=FLAGS.arch, \
					filters=FLAGS.filters, \
					depth=FLAGS.depth, \
					num_modalities=num_modalities, \
					layers=FLAGS.layers, \
					lite=FLAGS.lite, \
					norm=FLAGS.norm )
		model.summary( line_length=150 )

		# Define Optimizer
		print("\nDefining the Optimizer...")
		# IF OPTIMIZER
		if FLAGS.optimizer == 'Adam':
			optimizer = get_adam_opt( FLAGS.learning_rate )
		elif FLAGS.optimizer == 'RMSProp':
			optimizer = get_rms_opt( FLAGS.learning_rate )
		elif FLAGS.optimizer == 'SGD':
			optimizer = get_sgd_opt( FLAGS.learning_rate )
		elif FLAGS.optimizer == 'Momentum':
			optimizer = get_momentum_opt( FLAGS.learning_rate )
		# ENDIF OPTIMIZER
		print( optimizer.get_config() )

		### Import saved model from location 'loc' into local graph
		#IF
		if FLAGS.ckptDir and _os.path.isdir( FLAGS.ckptDir ):
			print("\nRestoring model parameters from checkpoint...\n")
			_sys.stdout.flush()
			
			model = _tf.keras.models.load_model( FLAGS.ckptDir )
		#ENDIF

		### Create distributed datasets
		print("\nDistributed datasets...")
		train_ds_dist = strategy.experimental_distribute_dataset( train_ds )
		print( train_ds_dist )
		
		val_ds_dist = strategy.experimental_distribute_dataset( val_ds )
		print( val_ds_dist )

		### Create the Train object
		trainer = Train( model, strategy, optimizer, NUM_GPU_DEVICES, FLAGS )
		
		### Train the model
		trainer.custom_loop( train_ds_dist, val_ds_dist, strategy )
	#ENDWITH STRATEGY



		



	### Remove tmpDir and its contents, if not user defined
	#IF
	if not user_defined_tmpdir:
		#TRY
		try:
			_shutil.rmtree( tmpDir )
		except:
			print(( "Failed to delete: " + tmpDir ))
		#ENDTRY
	#ENDIF

	### Print resource
	print("\nResource usage for this process")
	print("\tetime \t:", _np.round( ( _time.time() - startTimeStamp )/60, 2 ), "mins")
	
	# Resource package only available in Unix
	#IF
	if _platform.system() != 'Windows':
		import resource as _resource
		rus = _resource.getrusage(0)
		print("\tutime \t:", _np.round( rus.ru_utime, 2 ))
		print("\tstime \t:", _np.round( rus.ru_stime, 2 ))
		print("\tmaxrss \t:", _np.round( rus.ru_maxrss / 1.e6, 2 ), "GB")
		
		_sys.stdout.flush()
	#ENDIF
#ENDDEF MAIN
	


################################################# END OF FUNCTIONS ################################################
	
################################################ MAIN BODY ################################################

#IF
if __name__ == '__main__':
	 _main()
#ENDIF
