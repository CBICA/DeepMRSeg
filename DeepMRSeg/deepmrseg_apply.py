#!/usr/bin/python3

# deepmrseg

import argparse as _argparse
import os as _os
import sys as _sys
import urllib.request
from urllib.parse import urlparse
import zipfile
from . import deepmrseg_test

from . import pythonUtilities

_os.environ['COLUMNS'] = "90"

##############################################################
## Path to saved models

DEEPMRSEG = _os.path.expanduser(_os.path.join('~', '.deepmrseg'))
MDL_DIR = _os.path.join(DEEPMRSEG, 'trained_models')

modelDict = {}
modelDict['dlicv'] = _os.path.join(MDL_DIR, 'dlicv', 'DeepMRSeg_DLICV_v1.0')
modelDict['muse'] = _os.path.join(MDL_DIR, 'muse', 'DeepMRSeg_MUSE_v1.0')
modelDict['tissueseg'] = _os.path.join(MDL_DIR, 'tissueseg', 'DeepMRSeg_TissueSeg_v1.0')

##############################################################

#DEF ARGPARSER
def read_flags(argv):
	"""Parses args and returns the flags and parser."""
	### Import modules
	import argparse as _argparse

	exeName = _os.path.basename(argv[0])

	descTxt = '{prog} is a wrapper script that applies deepmrseg_test for pre-defined tasks. You can first download the trained model for your task using deepmrseg_downloadmodel (see examples)'.format(prog=exeName)

	epilogTxt = '''Examples:
	
  ## Apply dlicv model on single image (I/O OPTION 1)
  deepmrseg_downloadmodel --model dlicv
  {prog} --task dlicv --inImg sub1_T1.nii.gz --outImg sub1_DLICV.nii.gz
 
  ## Apply dlicv model on multiple subjects using an image list (I/O OPTION 2)
  {prog} --task dlicv --sList my_img_list.csv
     with my_img_list.csv:
       ID,T1,OutImg
       sub1,/my/indir/sub1_T1.nii.gz,/my/outdir/sub1_DLICV.nii.gz
       sub2,/my/indir/sub2_T1.nii.gz,/my/outdir/sub2_DLICV.nii.gz
       ...
     
  ## Apply dlicv model on all T1 images in the input folder (I/O OPTION 3)
  {prog} --task dlicv --inDir /my/indir --outDir /my/outdir --inSuff _T1.nii.gz --outSuff _DLICV.nii.gz
 
  '''.format(prog=exeName)

	parser = _argparse.ArgumentParser( formatter_class=_argparse.RawDescriptionHelpFormatter, \
		description=descTxt, epilog=epilogTxt )

#	TASK/MODEL
#	===========
	dirArgs = parser.add_argument_group( 'MODEL')
	dirArgs.add_argument( "--task", default=None, type=str, \
		help=	'Name of the segmentation task. Options are: ' \
			+ '[' + ', '.join(modelDict.keys()) + ']. (REQUIRED)')
	
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
	ioArgs3 = parser.add_argument_group( 'I/O OPTION 3', 'Batch processing of all images in a folder (works only for single-modality tasks).')
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

	otherArgs.add_argument( "--batch", default=None, type=int, \
		help="Batch size  (default: Depends on the task)" )

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


def verify_flags(FLAGS, parser):

	##################################################
	### Check required args
	if FLAGS.task is None:
		parser.print_help(_sys.stderr)
		print('ERROR: Missing required arg --model')
		_sys.exit(1)

	if FLAGS.task not in modelDict.keys():
		parser.print_help(_sys.stderr)
		print('ERROR: Task not found: ' + FLAGS.task)
		_sys.exit(1)
	
	##################################################
	### Check required args

	## Check I/O OPTIONS
	if FLAGS.inImg is not None:                         ## CASE 1: Single inImg
		for tmpImg in FLAGS.inImg:
			pythonUtilities.check_file( tmpImg )

	elif FLAGS.sList is not None:                       ## CASE 2: Img list
		pythonUtilities.check_file( FLAGS.sList )
		
	elif FLAGS.inDir is not None:                       ## CASE 3: Img dir
		pythonUtilities.check_file( FLAGS.inDir )
		if FLAGS.outDir is None:
			parser.print_help( _sys.stderr )
			print('ERROR: Missing required arg: outDir')
			_sys.exit(1)
			
	else:
		parser.print_help( _sys.stderr )
		print('ERROR: Missing required arg - The user should set one of the 3 I/O OPTIONS')
		_sys.exit(1)


def _main():
	"""Main program for the script"""

	### init argv
	argv0 = [_sys.argv[0]]
	argv = _sys.argv

	### Read command line args
	FLAGS,parser = read_flags(argv)
	verify_flags(FLAGS, parser)

	if FLAGS.task == 'tissueseg':
		## Add models in 1 orientation only
		argv = argv[3:]
		argv.append('--mdlDir')
		argv.append(_os.path.join(modelDict[FLAGS.task], 'LPS'))
		print(argv)
		#print(argsExt)
		
		print('Calling deepmrseg_test')
		deepmrseg_test._main_warg(argv0 + argv)
		
	if FLAGS.task == 'muse':
		## Add models in 1 orientation only
		argv = argv[3:]
		argv.append('--mdlDir')
		argv.append(_os.path.join(modelDict[FLAGS.task], 'LPS'))
		print(argv)
		#print(argsExt)
		
		## Set batch size to small value to reduce memory usage
		if FLAGS.batch == None:
			argv.append('--batch')
			argv.append('4')
			print('WARNING: Batch size set to 4 to reduce memory usage!')
			_sys.stdout.flush()
		
		print('Calling deepmrseg_test')
		deepmrseg_test._main_warg(argv0 + argv)
		
	if FLAGS.task == 'wmlesion':
		print('Model for task not trained yet, aborting: ' + FLAGS.task)
		_sys.exit(1)

	if FLAGS.task == 'dlicv':
		
		## Add models in 3 orientation		
		argv = argv[3:]
		argv.append('--mdlDir')
		argv.append(_os.path.join(modelDict[FLAGS.task], 'LPS'))
		argv.append('--mdlDir')
		argv.append(_os.path.join(modelDict[FLAGS.task], 'PSL'))
		argv.append('--mdlDir')
		argv.append(_os.path.join(modelDict[FLAGS.task], 'SLP'))
		print(argv)
		#print(argsExt)
		
		print('Calling deepmrseg_test')
		deepmrseg_test._main_warg(argv0 + argv)


if __name__ == "__main__":
	_main()
