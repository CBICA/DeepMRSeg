"""
Created on Mon Jul  3 14:27:26 2017

@author: jimit
"""

################################################ DECLARATIONS ################################################
__author__	 = 'Jimit Doshi'
__EXEC_NAME__	 = "segunet"

import os as _os
import sys as _sys
import time as _time
import signal as _signal
import tensorflow as _tf
import numpy as _np
import platform as _platform

_sys.path.append( _os.path.dirname( _sys.argv[0] ) )

from . import losses

from .data_io import checkFiles, extractPkl
from .models import create_model
from .data_augmentation import data_reader
from .optimizers import getAdamOpt, getRMSOpt, getSGDOpt, getMomentumOpt
from .layers import get_onehot

from . import pythonUtilities

################################################ FUNCTIONS ################################################


#DEF ARGPARSER
def read_flags():
	"""Returns flags"""

	import argparse as _argparse

	parser = _argparse.ArgumentParser( formatter_class=_argparse.ArgumentDefaultsHelpFormatter )

#	INPUT LIST
#	==========
	inputArgs = parser.add_argument_group( 'REQUIRED INPUT ARGS' )
	inputArgs.add_argument( "--sList", default=None, type=str, \
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
	trainArgs.add_argument( "--LR_sch", default='EXP', type=str, \
				help="learning rate schedule \
					choose from { EXP, PLAT }" )
	trainArgs.add_argument( "--decay", default=0.9, type=float, \
				help="exponential_decay for the learning rate" )
	trainArgs.add_argument( "--patience", default=5, type=int, \
				help="number of epochs to wait without improvement" )
	trainArgs.add_argument( "--gamma", default=0, type=float, \
				help="modulating factor for the losses" )
	trainArgs.add_argument( "--max_to_keep", default=5, type=int, \
				help="number of best performing models to keep")
	trainArgs.add_argument( "--batch", default=5, type=int, \
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

#	MISC
#	====
	miscArgs = parser.add_argument_group( 'MISCELLANEOUS' )
	miscArgs.add_argument( "--verb", default=1, type=int, \
				help="verbosity")
	miscArgs.add_argument( "--nJobs", default=None, type=int, \
				help="number of jobs/threads" )

#	FLAGS
#	=====
	flags = parser.parse_args()
	return flags, parser

#ENDDEF ARGPARSER

### Define signal trap function
#DEF
def signal_handler(signal, frame):
	
	print('Program interrupt signal received! Aborting operations ...')
	_sys.exit(0)
#ENDDEF


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
#	from sklearn.model_selection import KFold as _KFold
	import glob as _glob
	from concurrent.futures import ThreadPoolExecutor as _TPE

	_sys.stdout.flush()

	### Sanity checks on the provided arguments
	# Check if input files provided exist
	# FOR
	for f in FLAGS.sList, FLAGS.roi:
		pythonUtilities.checkFile( f )
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
		tmpDir = _tempfile.mkdtemp( prefix='tf_segunet_train_' )
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
	print( "\nTF version \t: %s" % (_tf.__version__) )
	
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
	print("LR Schedule \t: %s" % (FLAGS.LR_sch))
	print("Decay Factor \t: %s" % (FLAGS.decay))
	print("Gamma Factor \t: %s" % (FLAGS.gamma))
	print("Batch Size \t: %d" % (FLAGS.batch))
	print("Filter Num \t: %d" % (FLAGS.filters))
	print("Depth of Arch \t: %d" % (FLAGS.depth))
	print("Layers \t\t: %d" % (FLAGS.layers))
	print("Max to Keep \t: %d" % (FLAGS.max_to_keep))
	print("Label Smoothing : %s" % (str(FLAGS.label_smoothing)))
	print("Deep Supervision: %s" % (str(FLAGS.deep_supervision)))
	print("Lite Verion \t: %s" % (str(FLAGS.lite)))
	print("Patience Param \t: %d\n" % (FLAGS.patience))

	# create model dir
	if not _os.path.isdir( FLAGS.mdlDir ):
		_os.makedirs( FLAGS.mdlDir )

	# create training and validation subject lists
	train_sublist = []
	val_sublist = []
	all_sublist = []

	### Read subject list file
	print("Reading the input subject list and running sanity checks on the files")
	_sys.stdout.flush()

	### Multi-thread the loading of images
	#WITH
	with _TPE( max_workers=nJobs ) as executor:
		#WITH
		with open(FLAGS.sList) as f:
			reader = _csv.DictReader( f )
			idcolumn = FLAGS.idCol
	
			#FOR
			for row in reader:
				all_sublist.extend( [ row[FLAGS.idCol] ] )
		
				# Get files for other modalities
				otherModsFileList = []
				if FLAGS.otherMods:
					for mod in otherMods:
						otherModsFileList.extend( [ row[mod] ] )

				executor.submit( checkFiles, \
						refImg=row[FLAGS.refMod], \
						labImg=row[FLAGS.labCol], \
						otherImg=otherModsFileList )
#				checkFiles( refImg=row[FLAGS.refMod], \
#						labImg=row[FLAGS.labCol], \
#						otherImg=otherModsFileList )
			#ENDFOR
		#ENDWITH
	#ENDWITH

	### Split to train-validation
	print("Splitting the training list into training and validation")
	_sys.stdout.flush()

#	******************************************
#	* COMMENTED OUT ONLY FOR EXPERIMENTATION *
#	******************************************
#	# Randomize list
#	_shuffle( all_sublist )

	# Split into training and validation lists
#	from sklearn.cross_validation import train_test_split
#	train_sublist, val_sublist = train_test_split( all_sublist, test_size=0.05 )
	p = _np.int( len( all_sublist ) * 0.2 )
	val_sublist = all_sublist[ 0:p ]
	train_sublist = all_sublist[ p: ]

	print( "\nTraining subjects: " )
	for i in range( len(train_sublist) ):
		print( '\t%s' % (train_sublist[i]) )
		
	print( "\nValidation subjects: " )
	for i in range( len(val_sublist) ):
		print( '\t%s' % (val_sublist[i]) )
	
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
				executor.submit( extractPkl, \
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
#				extractPkl( \
#						subListFile=FLAGS.sList, \
#						idcolumn=FLAGS.idCol, \
#						labCol=FLAGS.labCol, \
#						refMod=FLAGS.refMod, \
#						otherMods=otherMods, \
#						num_modalities=num_modalities, \
#						subjectlist=[sub], \
#						roicsv=FLAGS.roi, \
#						out_path=pref, \
#						rescalemethod=FLAGS.rescale, \
#						xy_width=FLAGS.xy_width, \
#						pos_label_balance=FLAGS.label_balance, \
#						ressize=FLAGS.ressize, \
#						orient=FLAGS.reorient )
			#ELIF
		#ENDFOR ALL TRAINING SUBJECTS
	#ENDWITH

	
	# Check if CUDA_VISIBLE_DEVICES is set
	#TRY
	try:
		_os.environ["CUDA_VISIBLE_DEVICES"]
		device_name = _os.environ["CUDA_VISIBLE_DEVICES"]
	except:
		device_name = 0
	#ENDTRY
	print( "CUDA_VISIBLE_DEVICES: %s" % (device_name) )
	

	###########################
	#### PREPARE DATASETS #####
	###########################
	print("\n")

	# WITH CPU DEVICE
	with _tf.device( '/cpu:0' ):

		#DEF
		@_tf.function
		def tfrecordreader( serialized_example ):

			feature = {	 'image': _tf.io.FixedLenFeature( [], _tf.string ),
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
		# ENDDEF

		### TRAINING DATASET
		print("\n")
		train_ds = data_reader( filenames=train_filenames, \
						reader_func=tfrecordreader, \
						batch_size=FLAGS.batch, \
						mode=_tf.estimator.ModeKeys.TRAIN )

	
		### VALIDATION DATASET
		print("\n")
		val_ds = data_reader( filenames=val_filenames, \
						reader_func=tfrecordreader, \
						batch_size=1, \
						mode=_tf.estimator.ModeKeys.EVAL )
						
	# ENDWITH CPU DEVICE

	####################
	### DEFINE MODEL ###
	####################
	print("\nDefining the network...\n")
	model = create_model(	 num_classes=FLAGS.num_classes, \
				arch=FLAGS.arch, \
				filters=FLAGS.filters, \
				depth=FLAGS.depth, \
				num_modalities=num_modalities, \
				layers=FLAGS.layers, \
				lite=FLAGS.lite )		
	model.summary( line_length=150 )

	#####################
	##### OPTIMIZER #####
	#####################
	print("\n")

	# Initial Optimizer
	# IF OPTIMIZER
	if FLAGS.optimizer == 'Adam':
		optimizer = getAdamOpt( FLAGS.learning_rate )
	elif FLAGS.optimizer == 'RMSProp':
		optimizer = getRMSOpt( FLAGS.learning_rate )
	elif FLAGS.optimizer == 'SGD':
		optimizer = getSGDOpt( FLAGS.learning_rate )
	elif FLAGS.optimizer == 'Momentum':
		optimizer = getMomentumOpt( FLAGS.learning_rate )
	# ENDIF OPTIMIZER
	
	optimizer = _tf.keras.optimizers.Adam( FLAGS.learning_rate )
		

	######################
	### TRAIN NETWORK  ###
	######################
	print("\n")

	### Training the network
	print("\nTraining the network...\n")
	_sys.stdout.flush()
	
	iou_train = _tf.keras.metrics.MeanIoU( num_classes=FLAGS.num_classes )
	iou_test = _tf.keras.metrics.MeanIoU( num_classes=FLAGS.num_classes )

	epoch_train_loss_avg = _tf.keras.metrics.Mean()
	epoch_train_ioul_avg = _tf.keras.metrics.Mean()
	epoch_train_mael_avg = _tf.keras.metrics.Mean()
	epoch_train_bcel_avg = _tf.keras.metrics.Mean()
	
	epoch_val_loss_avg = _tf.keras.metrics.Mean()
	epoch_val_ioul_avg = _tf.keras.metrics.Mean()
	epoch_val_mael_avg = _tf.keras.metrics.Mean()
	epoch_val_bcel_avg = _tf.keras.metrics.Mean()

	@_tf.function
	def train_step( x,y ):
		with _tf.GradientTape() as tape:
			preds_d1,probs_d1,preds_d2,probs_d2,preds_d4,probs_d4 = model( x, training=True)

			oh_d1 = get_onehot( y,FLAGS.label_smoothing,FLAGS.xy_width,FLAGS.num_classes )

			total_loss, total_loss_d1, iou_d1, mae_d1, bce_d1 = losses.getCombinedLoss( oh_d1,probs_d1,probs_d2,probs_d4,\
													FLAGS.gamma,FLAGS.deep_supervision,FLAGS.xy_width )

		grads = tape.gradient( total_loss, model.trainable_weights )
		optimizer.apply_gradients( zip(grads, model.trainable_weights) )
		
		iou_train.update_state( _tf.squeeze(y),preds_d1 )

		return total_loss_d1, iou_d1, mae_d1, bce_d1

	@_tf.function
	def test_step( x,y ):
		preds_d1,probs_d1,_,_,_,_ = model( x, training=False )

		oh_d1 = get_onehot( y,0,FLAGS.xy_width,FLAGS.num_classes )
		
		total_loss_d1, iou_d1, mae_d1, bce_d1 = losses.CombinedLoss( oh_d1,probs_d1,0 )
		
		iou_test.update_state( _tf.squeeze(y),preds_d1 )

		return total_loss_d1, iou_d1, mae_d1, bce_d1

#	@_tf.function
	def set_lr( e,plat ):

		if not hasattr( optimizer, "lr" ):
			raise ValueError('Optimizer must have a "lr" attribute.')
			
		# Get the current learning rate from model's optimizer.
		lr = _tf.keras.backend.get_value(optimizer.lr)
		
		#IF
		if FLAGS.LR_sch == 'EXP':
			new_lr = FLAGS.learning_rate * FLAGS.decay ** (e-1)
		elif FLAGS.LR_sch == 'PLAT':
			if plat >= FLAGS.patience/4:
				new_lr = lr/2.
				plat = 0
			else:
				new_lr = lr
		#ENDIF
		
		# Set the value back to the optimizer before this epoch starts
		_tf.keras.backend.set_value( optimizer.lr, new_lr )

		return _tf.keras.backend.get_value(optimizer.lr), plat
	
	# DEF	
	def pop_extend( arr,val ):
		arr_list = arr.tolist()
		if len(arr) >= FLAGS.patience:
			arr_list.pop(0)
		arr_list.extend( [val] )
		
		return _np.array(arr_list)
	# ENDDEF
	
	# Get start time
	est = _time.time()

	### Import saved model from location 'loc' into local graph
	#IF
	if FLAGS.ckptDir and _os.path.isdir( FLAGS.ckptDir ):
		print("\nRestoring model parameters from checkpoint...\n")
		_sys.stdout.flush()
		
		model = _tf.keras.models.load_model( FLAGS.ckptDir )
	#ENDIF

	i = 0
	# Min loss at each epoch
	loss_min = _np.ones( FLAGS.max_to_keep ) * 100000
	loss_min_ind = _np.zeros( FLAGS.max_to_keep )

	# Max accuracy at each epoch
	accu_min = _np.zeros( FLAGS.max_to_keep )
	accu_min_ind = _np.zeros( FLAGS.max_to_keep )
	
	# Accuracy metric moving average
	accu_train_recent = _np.array( [] )
	accu_val_recent = _np.array( [] )

	last_saved_mdl_counter = 0
	last_lr_change_counter = 0
	current_lr = FLAGS.learning_rate

	for epoch in range( 1,FLAGS.num_epochs+1 ):
		start_time = _time.time()

		### Stop training if loss_cov has reached its limit
		#IF
		if epoch > FLAGS.min_epochs:
			if last_saved_mdl_counter >= FLAGS.patience:
				print( i, epoch, last_saved_mdl_counter )
				print("Early stopping criteria met. Halt training...")
				break
		#ENDIF

		### Set the learning rate based on the chosen schedule
		current_lr,last_lr_change_counter = set_lr( epoch,last_lr_change_counter )

		# Iterate over the batches of the dataset.
		for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
			ttl,tioul,tmael,tbcel = train_step( x_batch_train,y_batch_train )

			# Add current batch loss
			epoch_train_loss_avg.update_state( ttl )
			epoch_train_ioul_avg.update_state( tioul )
			epoch_train_mael_avg.update_state( tmael )
			epoch_train_bcel_avg.update_state( tbcel )

			# IF DISPLAY TRAINING METRICS
			if i%1000 == 0:
				timeperiter = (_time.time()-est) / (i+1) * 1000 / 60.

				print( "\t\titerations : %d, time per %d iterations: %.2f mins" % \
								( i, 1000, timeperiter ) )

				print( "\t\t\t training metrics \t: ( emaLoss: %.4f (%.4f,%.4f,%.4f), learning rates: %.1E )" \
						% (epoch_train_loss_avg.result(), \
							epoch_train_ioul_avg.result(), epoch_train_mael_avg.result(), epoch_train_bcel_avg.result(), \
							current_lr) )
				
				_sys.stdout.flush()
			# ENDIF
			
			i += 1
			
		### Checkpoint model
		model.save( _os.path.join( FLAGS.mdlDir \
					+ '/checkpoint/model' ), \
				overwrite=True, \
				include_optimizer=True )

		# Run a validation loop at the end of each epoch.
		for step, (x_batch_val, y_batch_val) in enumerate(val_ds):
			ttl,tioul,tmael,tbcel = test_step(x_batch_val, y_batch_val)

			# Add current batch loss
			epoch_val_loss_avg.update_state( ttl )
			epoch_val_ioul_avg.update_state( tioul )
			epoch_val_mael_avg.update_state( tmael )
			epoch_val_bcel_avg.update_state( tbcel )

		# IF
		if ( epoch_val_loss_avg.result() < loss_min[0] ):

			mdl_to_del = loss_min_ind[0]
			if _os.path.isdir( _os.path.join( FLAGS.mdlDir \
						+ '/bestmodels/model-' \
						+ str( int(mdl_to_del) ) ) ):
				_shutil.rmtree( _os.path.join( FLAGS.mdlDir \
							+ '/bestmodels/model-' \
							+ str( int(mdl_to_del) ) ) )
		
			loss_min[0] = epoch_val_loss_avg.result()
			loss_min_ind[0] = epoch
			loss_min_ind = loss_min_ind[ _np.argsort(loss_min)[::-1] ]
			loss_min = loss_min[ _np.argsort(loss_min)[::-1] ]
		
			last_saved_mdl_counter = 0
			last_lr_change_counter = 0
		
			model.save( _os.path.join( FLAGS.mdlDir \
						+ '/bestmodels/model-' \
						+ str(epoch) ), \
					overwrite=False, \
					include_optimizer=False )
		else:
			if epoch >= (FLAGS.min_epochs):
				last_saved_mdl_counter += 1
			
			last_lr_change_counter += 1
		# ENDIF

		accu_train_recent = pop_extend( accu_train_recent,iou_train.result() )
		accu_val_recent = pop_extend( accu_val_recent,iou_test.result() )
		
		timeperepoch = (_time.time()-est) / epoch / 60
		print( "\n\tepoch : %d, time/epoch: %.2f mins, learning rates: %.1E" \
				% ( epoch, timeperepoch, current_lr ) )

		print( "\t\t training metrics \t: mIOU: %.4f (%.4f), Loss: %.4f (%.4f,%.4f,%.4f)" \
				% ( iou_train.result(), accu_train_recent.mean(), \
				epoch_train_loss_avg.result(), \
				epoch_train_ioul_avg.result(), epoch_train_mael_avg.result(), \
				epoch_train_bcel_avg.result() ) )

		print( "\t\t validation metrics \t: mIOU: %.4f (%.4f), Loss: %.4f (%.4f,%.4f,%.4f) ( %.4f, %.4f )\n" \
				% ( iou_test.result(), accu_val_recent.mean(), \
				epoch_val_loss_avg.result(), \
				epoch_val_ioul_avg.result(), epoch_val_mael_avg.result(), \
				epoch_val_bcel_avg.result(), \
				loss_min.min(), loss_min.max() ) )

		_sys.stdout.flush()

		# Reset metrics at the end of each epoch
		iou_train.reset_states(); iou_test.reset_states()

		epoch_train_loss_avg.reset_states(), epoch_val_loss_avg.reset_states()
		epoch_train_ioul_avg.reset_states(), epoch_val_ioul_avg.reset_states()
		epoch_train_mael_avg.reset_states(), epoch_val_mael_avg.reset_states()
		epoch_train_bcel_avg.reset_states(), epoch_val_bcel_avg.reset_states()

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
	
	#resource package only available in Unix
	if _platform.system() != 'Windows':
		import resource as _resource
		rus = _resource.getrusage(0)
		print("\tutime \t:", _np.round( rus.ru_utime, 2 ))
		print("\tstime \t:", _np.round( rus.ru_stime, 2 ))
		print("\tmaxrss \t:", _np.round( rus.ru_maxrss / 1.e6, 2 ), "GB")
		
		_sys.stdout.flush()
# ENDDEF MAIN
	


################################################# END OF FUNCTIONS ################################################
	
################################################ MAIN BODY ################################################

#IF
if __name__ == '__main__':	
	 _main()
#ENDIF
