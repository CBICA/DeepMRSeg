
################################################ DECLARATIONS ################################################
__author__	 = 'Jimit Doshi'
__EXEC_NAME__	 = "deepmrseg_test"

import os as _os
import sys as _sys
import json as _json

import tensorflow as _tf
import numpy as _np
import platform as _platform
import tempfile as _tempfile
import nibabel as _nib
import csv as _csv

import shutil as _shutil

from . import pythonUtilities

from .data_io import load_res_norm
from .utils import get_roi_indices

_os.environ['COLUMNS'] = "90"

################################################ FUNCTIONS ################################################

############## HELP ##############


#DEF ARGPARSER
def read_flags(argv):
	"""Parses args and returns the flags and parser."""
	### Import modules
	import argparse as _argparse

	exeName = _os.path.basename(argv[0])

	descTxt = '{prog} applies pre-trained segmentation model(s) on image(s) from single or multiple subjects.'.format(prog=exeName)

	epilogTxt = '''Examples:
	
  ## Apply single-modal segmentation task on single image (I/O OPTION 1)
  {prog} -m /my/models/bmask_mdldir --inImg sub1_T1.nii.gz --outImg sub1_bmaskseg.nii.gz
 
  ## Apply multi-modal (FL and T1) segmentation task on single subject (I/O OPTION 1)
  {prog} -m /my/models/wmlesion_mdldir --inImg sub1_FL.nii.gz --inImg sub1_T1.nii.gz --outImg sub1_wmlseg.nii.gz

  ## Apply multi-modal segmentation task on multiple images using an image list (I/O OPTION 2)
  {prog} -m /my/models/wmlesion_mdldir --sList my_img_list.csv
     with my_img_list.csv:
       ID,FLAIR,T1,OutImg
       sub1,/my/indir/sub1_FL.nii.gz,/my/indir/sub1_T1.nii.gz,/my/outdir/sub1_wmlseg.nii.gz
       sub2,/my/indir/sub2_FL.nii.gz,/my/indir/sub2_T1.nii.gz,/my/outdir/sub2_wmlseg.nii.gz
       ...
     
  ## Apply single-modal segmentation task on multiple images within a folder (I/O OPTION 3)
  {prog} -m /my/models/bmask_mdldir --inDir /my/indir --outDir /my/outdir --inSuff _T1.nii.gz --outSuff _bmaskseg.nii.gz
 
  ## Apply multiple models and combine segmentations
  {prog} -m /my/models/bmask_mdldir1 -m /my/models/bmask_mdldir2 ... [I/O OPTIONS]
  
  '''.format(prog=exeName)

	parser = _argparse.ArgumentParser( formatter_class=_argparse.RawDescriptionHelpFormatter, \
		description=descTxt, epilog=epilogTxt )

#	MODELS
#	===========
	dirArgs = parser.add_argument_group( 'MODEL', 'Trained model directory')
	dirArgs.add_argument( "--mdlDir", action='append', default=None, type=str, \
		help=	'Path to the directory where the trained model should be loaded from. (REQUIRED)')

#	I/O
#	==========
	## I/O Option1: Single case processing
	ioArgs1 = parser.add_argument_group( 'I/O OPTION 1', 'Single case processing')
	ioArgs1.add_argument( "--inImg", action='append', default=None, type=str, \
		help=	'Input image name. For multi-modal tasks multiple image names \
			can be entered as "--inImg imgMod1 --inImg imgMod2 ..." . (REQUIRED)')
	
	ioArgs1.add_argument( "--outImg", default=None, type=str, \
		help=	'Output image name. (REQUIRED)')

	## I/O Option2: Batch I/O from list
	ioArgs2 = parser.add_argument_group( 'I/O OPTION 2', 'Batch processing using image list')
	ioArgs2.add_argument( "--sList", default=None, type=str, \
		help=	'Image list file name. Enter a comma separated list file with \
			columns for: ID, input image(s) and output image. (REQUIRED)')

	## I/O Option3: Batch I/O from folder
	ioArgs3 = parser.add_argument_group( 'I/O OPTION 3', 'Batch processing of all images in a folder  (works only for single-modality tasks)')
	ioArgs3.add_argument( "--inDir", default=None, type=str, \
			help=	'Input folder name. (REQUIRED)')

	ioArgs3.add_argument( "--outDir", default=None, type=str, \
		help=	'Output folder name. (REQUIRED)')
	
	ioArgs3.add_argument( "--inSuff", default='.nii.gz', type=str, \
		help='Input image suffix (default: .nii.gz)')
	
	ioArgs3.add_argument( "--outSuff", default='_SEG.nii.gz', type=str, \
		help="Output image suffix  (default: _SEG.nii.gz)")

#	OTHER OPTIONS
#	===========
	otherArgs = parser.add_argument_group( 'OTHER OPTIONS')

	otherArgs.add_argument( "--batch", default=64, type=int, \
		help="Batch size  (default: 64)" )

	otherArgs.add_argument( "--probs", default=False, action="store_true", \
		help=	'Flag to indicate whether to save a probability map for \
			each segmentation label. (default: False)')

	otherArgs.add_argument( "--nJobs", default=None, type=int, \
		help="Number of jobs/threads (default: automatically determined)" )
	
	otherArgs.add_argument( "--tmpDir", default=None, type=str, \
                help=	'Absolute path to the temporary directory. If not provided, \
			temporary directory will be created automatically and will be \
			deleted at the end of processing. (default: None)' )

	### Read args from CLI
	flags = parser.parse_args(argv[1:])

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

#CLASS
class LoadModel():

	""" Loading SavedModel """

	# DEF
	def __init__( self, checkpoint ):
		"""LoadModel class constructor to load models from checkpoints.

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
def extract_data_for_subject( otherImg=None,refImg=None,ressize=1, \
			orient='LPS', xy_width=320, rescalemethod='minmax' ):

	### Load images
	ref,_ = load_res_norm( refImg,xy_width,ressize,orient,mask=0,rescalemethod=rescalemethod )
	others =[]
	for img in otherImg:
		others.extend( [ load_res_norm( img,xy_width,ressize,orient,mask=0,rescalemethod=rescalemethod )[0] ] )

	### Restructure matrices from [x,y,z] to [z,x,y]
	ref = _np.moveaxis( ref,2,0 )
	for m in range( len(otherImg) ):
		others[m] = _np.moveaxis( others[m],2,0 )
	
	### Reshape T1 and FL to add a new axis
	ref = ref.reshape( ( ref.shape+(1,) ) )
	for m in range( len(otherImg) ):
		others[m] = others[m].reshape( ( others[m].shape+(1,) ) )

	### Return appended T1 and FL and the wml
	#IF
	if len(otherImg) > 0:
		allMods = ref.copy()
		for m in range( len(otherImg) ):
			allMods = _np.append( allMods,others[m],axis=3 )

		return allMods
	else:
		return ref
	#ENDIF
#ENDDEF

#DEF
def load_model( models,cp ):
	models.extend( [ LoadModel(checkpoint=cp) ] )
#ENDDEF

#DEF
def run_model( im_dat, num_classes, allmodels, bs ):

	### Create array to store output probabilities
	val_prob = _np.zeros( ( im_dat.shape[0:3] + (num_classes,len(allmodels)) ) )

	### Create a tf Dataset from im_dat
	im_dat_ds = _tf.data.Dataset.from_tensor_slices( (im_dat) ).batch( bs )
	

	# Launch testing
	# FOR EACH MODEL
	for c,mod in enumerate( allmodels ):
		i = 0
		# FOR EACH BATCH OF SLICES
		for one_batch in im_dat_ds:
			bs = one_batch.shape[0]
			val_prob[i:i+bs,:,:,:,c] = mod.run( one_batch )
			i += bs
		# ENDFOR EACH BATCH OF SLICES
	# ENDFOR EACH MODEL

	### Reshuffle predictions from [z,x,y,c,m] -> [x,y,z,c,m]
#	ens = _np.moveaxis( val_prob,0,2 ).astype('float32').mean( axis=-1 )
	ens = _np.moveaxis( val_prob,0,2 ).mean( axis=-1 ).astype('float32')
	del val_prob, im_dat_ds

	return ens
#ENDDEF

#DEF
def resample_ens( inImg,inImg_res_F,ens,ens_ref,c ):

	### Import more modules
	import nibabel as _nib
	import nibabel.processing as _nibp

	ens_f = _nib.Nifti1Image( ens[ :,:,:,c ], inImg_res_F.affine, inImg_res_F.header )
	ens_f_res = _nibp.resample_from_to( ens_f, inImg, order=0 ) #order=1 seems to produce the same results
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
def save_output( ens, refImg, num_classes, roi_indices, out=None, probs=False, \
			rescalemethod='minmax', ressize=float(1), orient='LPS', xy_width=320, nJobs=1 ):

	### Import more modules
	import nibabel as _nib
	from concurrent.futures import ThreadPoolExecutor as _TPE

	### Load reference image if it is a file path
	#IF
	if isinstance( refImg,str ):
		inImg = _nib.load( refImg )
	### Else, verify if it is a nibabel object with a header
	elif isinstance( refImg,_nib.Nifti1Image ):
		assert refImg.header
		inImg = refImg
	#ENDIF
	
	### Resample refImg
	_,inImg_res_F = load_res_norm( in_path=refImg, \
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
def predict_classes( refImg, otherImg, num_classes, allmodels, roi_indices, out=None, probs=False, \
			rescalemethod='minmax', ressize=float(1), orient='LPS', xy_width=320, batch_size=64, nJobs=1 ):

	im_dat = extract_data_for_subject( \
			otherImg=otherImg, \
			refImg=refImg, \
			ressize=ressize, \
			orient=orient, \
			xy_width=xy_width, \
			rescalemethod=rescalemethod )

	ens = run_model( im_dat=im_dat, num_classes=num_classes, \
			allmodels=allmodels, bs=batch_size )

	#IF OUTFILE PROVIDED
	if out:
		save_output( ens, refImg, num_classes, roi_indices, out=out, probs=probs, \
			rescalemethod=rescalemethod, ressize=ressize, orient=orient, xy_width=xy_width, nJobs=nJobs )
	else:
		return save_output( ens, refImg, num_classes, roi_indices, out=out, probs=probs, \
			rescalemethod=rescalemethod, ressize=ressize, orient=orient, xy_width=xy_width, nJobs=nJobs )
	#ENDIF OUTFILE PROVIDED
#ENDDEF


################################################ END OF FUNCTIONS ################################################
	
############## MAIN ##############
#DEF
def _main_warg(argv):

	
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

	##################################################
	### Read command line args
	print("\nParsing args    : %s\n" % (argv) )
	FLAGS,parser = read_flags(argv)
	if len( argv ) == 1:
		parser.print_help( _sys.stderr )
		_sys.exit(1)
	print(FLAGS)
	_sys.stdout.flush()

	##################################################
	### Check required args

	## Check mdlDir
	if FLAGS.mdlDir is not None:                        ## Mdl dir
		for tmpPath in FLAGS.mdlDir:
			pythonUtilities.check_file( tmpPath )
	else:
		print('ERROR: Missing required arg: --mdlDir')
		parser.print_help( _sys.stderr )
		_sys.exit(1)

	## Check I/O OPTIONS
	if FLAGS.inImg is not None:                         ## CASE 1: Single inImg
		IOType = 1
		for tmpImg in FLAGS.inImg:
			pythonUtilities.check_file( tmpImg )

	elif FLAGS.sList is not None:                       ## CASE 2: Img list
		IOType = 2
		pythonUtilities.check_file( FLAGS.sList )
		
	elif FLAGS.inDir is not None:                       ## CASE 3: Img dir
		IOType = 3
		pythonUtilities.check_file( FLAGS.inDir )
		
		if FLAGS.outDir is None:
			print('ERROR: Missing required arg: outDir')
			parser.print_help( _sys.stderr )
			_sys.exit(1)
			
	else:
		print('ERROR: Missing required arg - The user should set one of the 3 I/O OPTIONS')
		parser.print_help( _sys.stderr )
		_sys.exit(1)

	##################################################
	### Check/set optional args

	# Set nJobs
	#IF
	if FLAGS.nJobs:
		nJobs = FLAGS.nJobs
	else:
		nJobs = _os.cpu_count()
	#ENDIF

	# Set tmpDir
	#IF
	if FLAGS.tmpDir:
		user_defined_tmpdir = True
		tmpDir = FLAGS.tmpDir
		if not _os.path.isdir(FLAGS.tmpDir):
			_os.makedirs( FLAGS.tmpDir )
	else:
		user_defined_tmpdir = False
		tmpDir = _tempfile.mkdtemp( prefix='deepmrseg_test_' )
	#ENDIF
	

	##################################################
	### Print parsed args
	print("\nModel dir(s) \t: %s" % (FLAGS.mdlDir))
	if IOType == 1:
		print( "\nInput image(s) \t: %s" % (' , '.join(FLAGS.inImg)) )
		print( "\nOutput image \t: %s" % (FLAGS.outImg) )
	if IOType == 2:
		print( "\nImage List \t: %s" % (FLAGS.sList) )
	if IOType == 3:
		print( "\nInput dir \t: %s" % (FLAGS.inDir) )
		print( "\nOutput dir \t: %s" % (FLAGS.outDir) )
		print( "\nInput suffix \t: %s" % (FLAGS.outDir) )
		print( "\nOutput suffix \t: %s" % (FLAGS.outSuff) )
	print("\nBatch Size \t: %s" % (FLAGS.batch))
	print("\nOutput probs \t: %d" % (FLAGS.probs))
	print("\nNumber of jobs \t: %d" % (nJobs))
	print("\nTemp folder \t: %s" % (tmpDir))
	
	_sys.stdout.flush()
	
	###############################################################
	#### For direct I/O options: Create list file with in/out images 
	if IOType == 1:

		# Generate list of columns
		numMod = len(FLAGS.inImg)   
		modCols = 'Mod_'
		modCols = [modCols + str(i+1) for i in range(numMod)]
		columns = _np.array(['ID'] + modCols + ['OutImg']).reshape(1,-1)

		# Add absolute path to images
		FLAGS.inImg = list(map(_os.path.abspath, FLAGS.inImg))   
		FLAGS.outImg = _os.path.abspath(FLAGS.outImg)

		# Create csv data
		data=_np.array(['Scan1'] + FLAGS.inImg + [FLAGS.outImg]).reshape(-1, numMod+2)

		# Write csv file
		outCsv = _os.path.join(tmpDir, 'deepmrseg_test_filelist.csv')
		with open(outCsv, 'w') as csvfile:
			csvw = _csv.writer(csvfile)
			csvw.writerows(columns)
			csvw.writerows(data)

		# Set flag for image list
		FLAGS.sList = outCsv

	if IOType == 3:

		# Generate list of columns
		numMod = 1          ## Multi-modal data is not implemented for "input from folder"
		modCols = 'Mod_'
		modCols = [modCols + str(i+1) for i in range(numMod)]
		columns = _np.array(['ID'] + modCols + ['OutImg']).reshape(1,-1)

		# Add absolute path to in/out dirs
		FLAGS.inDir = _os.path.abspath(FLAGS.inDir)
		FLAGS.outDir = _os.path.abspath(FLAGS.outDir)
		
		# Add image extension to in/out suffix (.nii.gz for now)
		if FLAGS.inSuff.endswith('.nii.gz') == False:
			FLAGS.inSuff = FLAGS.inSuff + '.nii.gz'

		# Add image extension to in/out suffix (.nii.gz for now)
		if FLAGS.outSuff.endswith('.nii.gz') == False:
			FLAGS.outSuff = FLAGS.outSuff + '.nii.gz'
			
		# Create data frame
		#   Get list of all images with inSuff in the input dir
		data = []
		subInd = 1
		for (dir, _, files) in _os.walk(FLAGS.inDir):
			for i,f in enumerate(files):
				if f.endswith(FLAGS.inSuff):
					pathIn = _os.path.join(dir, f)
					pathOut = _os.path.join(FLAGS.outDir, f).replace('.nii.gz', FLAGS.outSuff)
					data.append(['Scan' + str(subInd), pathIn, pathOut])
					subInd += 1
			
		# Write csv file
		outCsv = _os.path.join(tmpDir, 'deepmrseg_test_filelist.csv')
		with open(outCsv, 'w') as csvfile:
			csvw = _csv.writer(csvfile)
			csvw.writerows(columns)
			csvw.writerows(data)

		# Set flag for image list
		FLAGS.sList = outCsv
	###############################################################


	###############################################################
	#### Run testing independently for each model path

	from concurrent.futures import ThreadPoolExecutor as _TPE

	num_models = len(FLAGS.mdlDir)
	FLAGS.mdlDir = list(map(_os.path.abspath, FLAGS.mdlDir))   

	#FOR
	for indMdl, currMdl in enumerate(FLAGS.mdlDir):
	
	
		##########################
		#### LOAD ALL MODELS #####
		##########################

		## Load test_config.json to set orientation
		testConf = (_os.path.join(currMdl, 'configs', 'test_config.json'))
		roiIndFile = (_os.path.join(currMdl, 'configs', 'ROI_Indices.csv'))
		with open(testConf) as f:
			testconfigflags = _json.load(f)
		trainflag_orient = testconfigflags['reorient']
		trainflag_rescale = testconfigflags['rescale']
		trainflag_ressize = testconfigflags['ressize']
		trainflag_xy_width = testconfigflags['xy_width']
		trainflag_num_classes = testconfigflags['num_classes']
		

		print("\n")
		print("\n---->	Model " + str(indMdl +1))
		print("\n---->	Loading all stored models in model path " + str(indMdl+1) + ' : ' + currMdl)
		print("\n---->	Reorient is : " + trainflag_orient)
		_sys.stdout.flush()
		
		allmodels = []
		# Launch threads to load models simultaneously

		#WITH
		with _TPE( max_workers=None ) as executor:
			# FOR ALL CHECKPOINTS
			for checkpoint in _os.listdir( _os.path.join(currMdl, 'bestmodels')):
				cppath = _os.path.join( currMdl, 'bestmodels', checkpoint )
				print( "\t\t-->	Loading ", _os.path.basename(cppath) )
				executor.submit( load_model,allmodels,cppath )
			# ENDFOR ALL CHECKPOINTS
		#ENDWITH
		print("")
		_sys.stdout.flush()
		
		_tf.keras.backend.clear_session()
		
		### Encode indices to ROIs if provided
		roi_indices = get_roi_indices(roicsv=roiIndFile)

		### Predict
		print("\n----> Running predictions for all subjects in the FileList")
		_sys.stdout.flush()

		# Read column names in input csv file
		with open (FLAGS.sList) as f:
			colNames = next(_csv.reader(f))
			idCol = colNames[0]
			refMod = colNames[1]
			otherMods = colNames[2:-1]
			outCol = colNames[-1]

		#WITH OPENFILE
		with open(FLAGS.sList) as f:
			reader = _csv.DictReader( f )

			#FOR
			for row in reader:

				### Get image filenames
				subId = row[idCol]
				refImg = row[refMod]
				outImg = row[outCol]
				
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
					
					# If there is a single model write the output directly to out image
					if num_models == 1:
						outSel = outImg
						probsSel = FLAGS.probs
						
					# If not, create temporary out file for the current model
					# also probs is set to True, as final mask will be calculated by combining probs from multiple models 
					else:
						outSel = _os.path.join(tmpDir, subId, subId + '_model' + str(indMdl+1) + '_out.nii.gz')
						probsSel = True

					#### Create output directory if it doesn't exist already
					##IF
					if not _os.path.isdir( _os.path.dirname(outSel) ):
						_os.makedirs( _os.path.dirname(outSel) )
					#ENDIF
					
					print( "\t---->	%s" % ( subId ) )
					_sys.stdout.flush()
			
					predict_classes( \
							refImg=refImg, \
							otherImg=otherModsFileList, \
							num_classes=trainflag_num_classes, \
							allmodels=allmodels, \
							roi_indices=roi_indices, \
							out=outSel, \
							probs=probsSel, \
							rescalemethod=trainflag_rescale, \
							ressize=trainflag_ressize, \
							orient=trainflag_orient, \
							xy_width=trainflag_xy_width, \
							batch_size=FLAGS.batch, \
							nJobs=nJobs )		
					
				#ENDIF
			#ENDFOR
		#ENDWITH OPENFILE

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
	#ENDFOR
	
	#####################################################################
	## Merge output from all models
	
	
	if num_models > 1:

		print("\n")
		print("\n---->	Creating final predictions by combining output from each model")
		_sys.stdout.flush()
		
		#WITH OPENFILE
		with open(FLAGS.sList) as f:
			reader = _csv.DictReader( f )

			#FOR
			for row in reader:
			
				### Get image filenames
				subId = row[idCol]
				outImg = row[outCol]

				if not _os.path.isfile( outImg ):

					# Get image size from the first nii.gz image 
					im_shape = []
					for f in _os.listdir(_os.path.join(tmpDir, subId)):
						if f.endswith('.nii.gz'):
							break
					im_shape = _nib.load(_os.path.join(tmpDir, subId, f)).get_data().shape
					
					# Create array to store output probabilities
					val_prob = _np.zeros( ( list(im_shape[0:3]) + [trainflag_num_classes] ) )
				
					# Create output directory if it doesn't exist already
					##IF
					if not _os.path.isdir( _os.path.dirname(outImg) ):
						_os.makedirs( _os.path.dirname(outImg) )
					#ENDIF

					#FOR
					for clInd in range(trainflag_num_classes):

						clSuff = '_probabilities_' + str(clInd)

						# Combine prob values for different models
						files = []
						for f in _os.listdir(_os.path.join(tmpDir, subId)):
							if f.endswith(clSuff + '.nii.gz'):
								files.append(_os.path.join(tmpDir, subId, f))
						
						if len(files)>0:
							niiTmp = _nib.load(files[0])
							probOut = niiTmp.get_data()                
							if len(files) > 1:
								for tmpF in files[1:]:
									niiTmp = _nib.load(tmpF)
									probTmp = niiTmp.get_data()
									probOut += probTmp
								probOut = probOut / len(files)
								
							# Write probabilities
							if FLAGS.probs:
								outNii = _nib.Nifti1Image( probOut, niiTmp.affine, niiTmp.header )
								outNii.to_filename(outImg.replace('.nii.gz', '_probabilities_' + str(clInd) + '.nii.gz'))
						val_prob[:,:,:,clInd] = probOut
					#ENDFOR

					### Get preds from prob matrix
					val_bin = _np.argmax( val_prob, axis=-1 )

					# Write binary mask
					outNii = _nib.Nifti1Image( val_bin, niiTmp.affine, niiTmp.header )
					outNii.set_data_dtype( 'uint8' )
					outNii.to_filename(outImg)

					print("\n")
					print("\n---->	Out image : " + outImg)
					_sys.stdout.flush()
				
				#ENDIF
			#ENDFOR
		#ENDWITH
	#ENDIF
	
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


# ENDDEF MAIN

def _main():
	### init argv
	argv = _sys.argv
	_main_warg(argv)

	
#IF
if __name__ == '__main__':
	_main()
#ENDIF
