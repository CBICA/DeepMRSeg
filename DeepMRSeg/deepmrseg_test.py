
################################################ DECLARATIONS ################################################
__author__	 = 'Jimit Doshi'
__EXEC_NAME__	 = "deepmrseg_test"

import os as _os
import sys as _sys

import tensorflow as _tf
import numpy as _np
import platform as _platform

from . import pythonUtilities

from .data_io import loadrespadsave

################################################ FUNCTIONS ################################################

############## HELP ##############


#DEF ARGPARSER
def read_flags():
	"""Parses args and returns the flags and parser."""
	### Import modules
	import argparse as _argparse

	parser = _argparse.ArgumentParser( formatter_class=_argparse.ArgumentDefaultsHelpFormatter )

#	INPUT LIST
#	==========
	inputArgs = parser.add_argument_group( 'REQUIRED INPUT ARGS' )
	inputArgs.add_argument( "--sList", default=None, type=str, \
				help="absolute path to the subject list for training")
	inputArgs.add_argument( "--idCol", default=None, type=str, \
				help="id column name (e.g.: 'ID')" )
	inputArgs.add_argument( "--refMod", default=None, type=str, \
				help="referene modality column name (e.g.: 'FL')")
	inputArgs.add_argument( "--otherMods", default=None, type=str, \
				help="other modalities to use (e.g.: 'T1,T2')" )
	inputArgs.add_argument( "--outCol", default=None, type=str, \
				help="column name containing the output filepaths (e.g.: 'out')")
	inputArgs.add_argument( "--roi", default=None, \
				help="absolute path to the ROI csv containing the ROIs to be \
					considered and their respective indices. If not provided, \
					the ROIs are assumed to be in the range( 0,num_classes-1 )")

#	DIRECTORIES
#	===========
	dirArgs = parser.add_argument_group( 'DIRECTORIES' )
	dirArgs.add_argument( "--mdlDir", nargs='+', default="model", type=str, \
				help="absolute path to the directory where the model \
					should be loaded from. \
					You can provide multiple paths")

#	INPUT IMAGE
#	===========
	inpArgs = parser.add_argument_group( 'INPUT IMAGE' )
	inpArgs.add_argument( "--prep", default=False, action="store_true", \
				help="flag indicating that the input images are \
					already preprocessed (default: False)")
	inpArgs.add_argument( "--num_classes", default=2, type=int, \
				help="number of classes to be considered in the input")
	inpArgs.add_argument( "--rescale", default='norm', type=str, \
				help="rescale method, choose from { minmax, norm }")
	inpArgs.add_argument( "--xy_width", default=320, type=int, \
				help="xy dimensions of the input patches. \
					Determines how much each slice needs to be padded \
					in the xy dimension. Should be divisible by 2**depth.")
	inpArgs.add_argument( "--ressize", default=1, type=float, \
				help="isotropic voxel size for the input images \
					input images will be resampled to this resolution" )
	inpArgs.add_argument( "--reorient", default='LPS', type=str, \
				help="reorient the testing images to match the \
					provided orientation (in radiology convention)" )
	inpArgs.add_argument( "--batch", default=64, type=int, \
				help="batch size" )

#	OUTPUT
#	========
	outArgs = parser.add_argument_group( 'OUTPUT' )
	outArgs.add_argument( "--probs", default=False, action="store_true", \
				help="flag to indicate whether the probabilities should \
					be stored as nii.gz (default: no)")


#	MISC
#	====
	miscArgs = parser.add_argument_group( 'MISCELLANEOUS' )
	miscArgs.add_argument( "--verb", default=False, action="store_true", \
				help="verbosity")
	miscArgs.add_argument( "--nJobs", default=None, type=int, \
				help="number of jobs/threads" )
	miscArgs.add_argument( "--delay", default=1, type=int, \
				help="delay between launching parallel prediction \
					for each subject (default: 1)" )

#	FLAGS
#	=====
	flags = parser.parse_args()
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

#CLASS
class LoadModel():

	""" Loading SavedModel """

	# DEF
	def __init__( self, checkpoint ):
		"""
		LoadModel class constructor to load models from checkpoints.

		Args:
			checkpoint: path to checkpoint
		"""
		self.model = _tf.keras.models.load_model(checkpoint)
	# ENDDEF
		
	# DEF
	def run( self, im_slice ):
		"""Running the activation operation previously imported.

		Args:
			im_slice: image slice of shape (b,x,y,m)
		"""
		# predict
		_,prob,_,_,_,_ = self.model.predict( im_slice )
		return prob
	# ENDDEF
#ENDCLASS

#DEF
def extractDataForSubject( otherImg=None,refImg=None,ressize=1, \
			orient='LPS', xy_width=320, rescalemethod='minmax' ):
	
	### Load images
	ref,_ = loadrespadsave( refImg,xy_width,ressize,orient,mask=0,rescalemethod=rescalemethod )
	others =[]
	for img in otherImg:
		others.extend( [ loadrespadsave( img,xy_width,ressize,orient,mask=0,rescalemethod=rescalemethod )[0] ] )
	
	### Restructure matrices from [x,y,z] to [z,x,y]
	ref = _np.moveaxis( ref,2,0 )
	for m in range( len(otherImg) ):
		others[m] = _np.moveaxis( others[m],2,0 )
	
	### Reshape T1 and FL to add a new axis
	ref = ref.reshape( ( ref.shape+(1,) ) )
	for m in range( len(otherImg) ):
		others[m] = others[m].reshape( ( others[m].shape+(1,) ) )

	### Return appended T1 and FL and the wml
	if len(otherImg) > 0:
		allMods = ref.copy()
		for m in range( len(otherImg) ):
			allMods = _np.append( allMods,others[m],axis=3 )
		
		return allMods
	else:
		return ref
	
#ENDDEF

#DEF
def load_model( models,cp ):
	models.extend( [ LoadModel(checkpoint=cp) ] )
#ENDDEF

#DEF
def runModel( im_dat, num_classes, allmodels, bs ):

	### Create array to store output probabilities
	val_prob = _np.zeros( ( im_dat.shape[0:3] + (num_classes,len(allmodels)) ) )

	### Create a tf Dataset from im_dat
	im_dat_ds = _tf.data.Dataset.from_tensor_slices( (im_dat) ).batch( bs )

	# Launch testing
	# FOR EACH MODEL
	for c in range( len(allmodels) ):
		i = 0
		# FOR EACH BATCH OF SLICES
		for one_batch in im_dat_ds:
			bs = one_batch.shape[0]
			val_prob[i:i+bs,:,:,:,c] = allmodels[c].run( one_batch )
			i += bs
		# ENDFOR EACH BATCH OF SLICES
	# ENDFOR EACH MODEL

	### Reshuffle predictions from [z,x,y,c,m] -> [x,y,z,c,m]
	ens = _np.moveaxis( val_prob,0,2 ).astype('float32').mean( axis=-1 )
	del val_prob, im_dat_ds

	return ens
#ENDDEF

#DEF
def resample_ens( inImg,inImg_res_F,ens,ens_ref,c ):

	### Import more modules
	import nibabel as _nib
	import nibabel.processing as _nibp

	ens_f = _nib.Nifti1Image( ens[ :,:,:,c ], inImg_res_F.affine, inImg_res_F.header )
	ens_f_res = _nibp.resample_from_to( ens_f, inImg, order=1 ) #order=0 seems to produce the same results
	ens_ref[ :,:,:,c ] = ens_f_res.get_data()

	del ens_f, ens_f_res
#ENDDEF

#DEF
def save_output_probs( ens_ref,ind,roi,inImg,out ):

	### Import more modules
	import nibabel as _nib

	outImgDat = ens_ref[:,:,:,ind].copy()
	outImgDat = _np.where( outImgDat<0.01, 0, outImgDat )

	outImgDat_img = _nib.Nifti1Image( outImgDat, inImg.affine, inImg.header )
	outImgDat_img.set_data_dtype( 'float32' )
	outImgDat_img.to_filename( _os.path.join( out[:-7] + '_probabilities_' + str(roi) + '.nii.gz' ) )
#ENDDEF

#DEF
def saveOutput( ens, refImg, num_classes, roi_indices, out=None, probs=False, \
			rescalemethod='minmax', ressize=float(1), orient='LPS', xy_width=320, nJobs=1 ):

	### Import more modules
	import nibabel as _nib
	from concurrent.futures import ThreadPoolExecutor as _TPE

	### Read reference image
	inImg = _nib.load( refImg )
	
	### Resample refImg
	_,inImg_res_F = loadrespadsave( in_path=refImg, \
					xy_width=xy_width, \
					ressize=ressize, \
					orient=orient, \
					mask=0, \
					rescalemethod=rescalemethod, \
					out_path=None )

	### Re-orient, resample and resize ens to the refImg space
	ens_ref = _np.zeros( (inImg.shape+(num_classes,)),dtype='float32' )
	#WITH
	with _TPE( max_workers=nJobs ) as executor:
		#FOR
		for c in range(num_classes):
			executor.submit( resample_ens,inImg,inImg_res_F,ens,ens_ref,c )
		#ENDFOR
	#ENDWITH

	### Get preds from ens
	ens_pred = _np.argmax( ens_ref,axis=-1 )

	### Encode indices to rois if provided
	ens_pred_enc = _np.zeros_like( ens_pred )
	#FOR
	for i in range( len(roi_indices) ):
		ind,roi = roi_indices[i]
		ens_pred_enc = _np.where( ens_pred==int(ind), int(roi), ens_pred_enc )
	#ENDFOR

	### Clear memory
	del ens_pred


	### If probabilities need to be saved
	#IF
	if probs:
		#WITH
		with _TPE( max_workers=nJobs ) as executor:
			#FOR
			for i in range( len(roi_indices) ):
				ind,roi = roi_indices[i]
				executor.submit( save_output_probs,ens_ref,ind,roi,inImg,out )
			#ENDFOR
		#ENDWITH
	#ENDIF

	### Clear memory
	del ens_ref

	### Get outImg from probabilities
	outImgDat_img = _nib.Nifti1Image( ens_pred_enc, inImg.affine, inImg.header )
	outImgDat_img.set_data_dtype( 'uint8' )
	if out:
		outImgDat_img.to_filename( out )
	else:
		return outImgDat_img

#ENDDEF

#DEF
def predictClasses( refImg, otherImg, num_classes, allmodels, roi_indices, out=None, probs=False, \
			rescalemethod='minmax', ressize=float(1), orient='LPS', xy_width=320, batch_size=64, nJobs=1 ):

	im_dat = extractDataForSubject( \
			otherImg=otherImg, \
			refImg=refImg, \
			ressize=ressize, \
			orient=orient, \
			xy_width=xy_width, \
			rescalemethod=rescalemethod )

	ens = runModel( im_dat=im_dat, num_classes=num_classes, \
			allmodels=allmodels, bs=batch_size )

	#IF OUTFILE PROVIDED
	if out:
		saveOutput( ens, refImg, num_classes, roi_indices, out=out, probs=probs, \
			rescalemethod=rescalemethod, ressize=ressize, orient=orient, xy_width=xy_width, nJobs=nJobs )
	else:
		return saveOutput( ens, refImg, num_classes, roi_indices, out=out, probs=probs, \
			rescalemethod=rescalemethod, ressize=ressize, orient=orient, xy_width=xy_width, nJobs=nJobs )
	#ENDIF OUTFILE PROVIDED
#ENDDEF

	


	


################################################ END OF FUNCTIONS ################################################
	
############## MAIN ##############
#DEF
def _main():

	### init argv
	argv = _sys.argv
	
	### Timestamps
	import time as _time
	startTime = _time.asctime()
	startTimeStamp = _time.time()

	### Print startTimeStamp
	print("\nHostname	: " + str( _os.getenv("HOSTNAME") ))
	print("Start time	: " + str( startTime ))

	_sys.stdout.flush()

	### Specifying the trap signal
	import signal as _signal
	if _platform.system() != 'Windows':
		_signal.signal( _signal.SIGHUP, signal_handler )
	_signal.signal( _signal.SIGINT, signal_handler )
	_signal.signal( _signal.SIGTERM, signal_handler )

	### Read command line args
	print("\nParsing args    : %s\n" % (argv[ 1: ]) )
	FLAGS,parser = read_flags()
	print(FLAGS)

	_sys.stdout.flush()

	### Check the number of arguments
	if len( argv ) == 1:
		parser.print_help( _sys.stderr )
		_sys.exit(1)


	### Sanity checks on the provided arguments
	# Check if input files provided exist
	# FOR
	for f in FLAGS.sList, FLAGS.roi:
		pythonUtilities.checkFile( f )
	# ENDFOR
	
	# Check if xy_width matches the depth
	# IF
	if FLAGS.xy_width % 16 != 0:
		print( "ERROR: The xy_width (%d) is not divisible by %d" % (FLAGS.xy_width,16) )
		_sys.exit(1)
	# ENDIF
	
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

	### Print parsed args
	print( "\nFile List \t: %s" % (FLAGS.sList) )
	print( "ID Column \t: %s" % (FLAGS.idCol) )
	print( "Output Column \t: %s" % (FLAGS.outCol) )
	print( "Ref Modality \t: %s" % (FLAGS.refMod) )
	print( "Other Mods \t: %s" % (otherMods) )
	print( "Num of Mods \t: %d" % (num_modalities) )
	print( "Data Prepped \t: %d" % (FLAGS.prep) )
	
	print("\nModel Dir(s) \t: %s" % (FLAGS.mdlDir))
	print( "ROI csv \t: %s" % (FLAGS.roi) )

	print("\nNum of Classes \t: %d" % (FLAGS.num_classes))
	print("Rescale Method \t: %s" % (FLAGS.rescale))
	print("XY width \t: %d" % (FLAGS.xy_width))
	print("Voxel Size \t: %f" % (FLAGS.ressize))
	print("Orientation \t: %s" % (FLAGS.reorient))
	print("Batch Size \t: %s" % (FLAGS.batch))
	
	print("\nOutput probs \t: %d" % (FLAGS.probs))
	
	### Create temp dir, if needed
	import csv as _csv
	from concurrent.futures import ThreadPoolExecutor as _TPE

	
	##########################
	#### LOAD ALL MODELS #####
	##########################
	print("\n")
	print("\n---->	Loading all stored models in ", FLAGS.mdlDir)
	_sys.stdout.flush()
	
	allmodels = []
	# Launch threads to load models simultaneously
	print("")
	#WITH
	with _TPE( max_workers=None ) as executor:
		# FOR ALL MODEL DIRS
		for mDir in FLAGS.mdlDir:
			# FOR ALL CHECKPOINTS
			for checkpoint in _os.listdir( mDir ):
				cppath = _os.path.join( mDir + '/' + checkpoint )
				print( "\t\t-->	Loading ", _os.path.basename(cppath) )
				executor.submit( load_model,allmodels,cppath )
			# ENDFOR ALL CHECKPOINTS
		# ENDFOR ALL MODEL DIRS
	#ENDWITH
	print("")
	_sys.stdout.flush()
	
	### Encode indices to ROIs if provided
	roi_indices = []
	#IF
	if FLAGS.roi:
		#WITH
		with open(FLAGS.roi) as roicsvfile:
			roi_reader = _csv.DictReader( roicsvfile )

			#FOR
			for roi_row in roi_reader:
				roi_indices.extend( [ [int(roi_row['Index']), int(roi_row['ROI'])] ] )
			#ENDFOR
		#ENDWITH
	else:
		for i in range( FLAGS.num_classes ):
			roi_indices.extend( [ [i,i] ] )
	#ENDIF
	

	### Predict
	print("\n----> Running predictions for all subjects in the FileList")
	_sys.stdout.flush()

	#WITH TPE
	with _TPE( max_workers=nJobs ) as executor:
	
		#WITH OPENFILE
		with open(FLAGS.sList) as f:
			reader = _csv.DictReader( f )

			#FOR
			for row in reader:

				### Get image filenames
				refImg = row[FLAGS.refMod]
				outImg = row[FLAGS.outCol]

				# Get files for other modalities
				otherModsFileList = []
				#IF
				if otherMods:
					#FOR
					for mod in otherMods:
						otherModsFileList.extend( [ row[mod] ] )
					#ENDFOR
				#ENDIF
			
				### Create output directory if it doesn't exist already
				#IF
				if not _os.path.isdir( _os.path.dirname(outImg) ):
					_os.makedirs( _os.path.dirname(outImg) )
				#ENDIF

				### Check if the file exists already
				#IF
				if not _os.path.isfile( outImg ):
					print( "\t---->	%s" % ( row[FLAGS.idCol] ) )
					_sys.stdout.flush()
			
#					executor.submit( predictClasses, \
#							refImg=refImg, \
#							otherImg=otherModsFileList, \
#							num_classes=FLAGS.num_classes, \
#							allmodels=allmodels, \
#							roi_indices=roi_indices, \
#							out=outImg, \
#							probs=FLAGS.probs, \
#							rescalemethod=FLAGS.rescale, \
#							ressize=FLAGS.ressize, \
#							orient=FLAGS.reorient, \
#							xy_width=FLAGS.xy_width, \
#							batch_size=FLAGS.batch, \
#							nJobs=nJobs )
					predictClasses( \
							refImg=refImg, \
							otherImg=otherModsFileList, \
							num_classes=FLAGS.num_classes, \
							allmodels=allmodels, \
							roi_indices=roi_indices, \
							out=outImg, \
							probs=FLAGS.probs, \
							rescalemethod=FLAGS.rescale, \
							ressize=FLAGS.ressize, \
							orient=FLAGS.reorient, \
							xy_width=FLAGS.xy_width, \
							batch_size=FLAGS.batch, \
							nJobs=nJobs )
						
					_time.sleep( FLAGS.delay )
				#ENDIF
			#ENDFOR
		#ENDWITH OPENFILE
	#ENDWITH TPE

	### Print resouce usage
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
	
#IF
if __name__ == '__main__':
	_main()
#ENDIF

