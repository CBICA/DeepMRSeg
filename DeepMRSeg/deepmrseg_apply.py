#!/usr/bin/python3

# deepmrseg

import argparse as _argparse
import os as _os
import sys as _sys
import urllib.request
from urllib.parse import urlparse
import zipfile
from . import deepmrseg_test

##############################################################
## Path to saved models
## FIXME: It would be better toread this from an external table hosted in model repo

DEEPMRSEG = _os.path.expanduser(_os.path.join('~', '.deepmrseg'))
MDL_DIR = _os.path.join(DEEPMRSEG, 'trained_models')

modelDict = {}
modelDict['dlicv'] = _os.path.join(MDL_DIR, 'dlicv', 'DeepMRSeg_DLICV')

##############################################################

#DEF ARGPARSER
def read_flags(argv):
	"""Parses args and returns the flags and parser."""
	### Import modules
	import argparse as _argparse

	parser = _argparse.ArgumentParser(description='Applies deepmrseg_test for pre-defined tasks', epilog='Please download first the trained model for your task using deepmrseg_loadmodel')

	# Segmentation task
	# ==========
	inputArgs = parser.add_argument_group('Segmentation task')
	inputArgs.add_argument( "--task", default=None, type=str, help="Name of the task [" + ','.join(modelDict.keys()) + "]" )
	
	# I/O args for deepmrseg_test
	# ==========
	inputArgs = parser.add_argument_group( 'INPUT SCANS - REQUIRED', 'Select one of the 3 options: Single case (inImg/outImg), all scans in a folder (inDir/outDir/outSuff) or using a scan list (sList)')
	
	### I/O Option1: Single scan I/O
	#inputArgs.add_argument( "--inImg", action='append', default=None, type=str, \
				#help="I/O Option1: Input scan name(s)")
	#inputArgs.add_argument( "--outImg", default=None, type=str, \
				#help="I/O Option1: Output scan name")

	### I/O Option2: Batch I/O from folder
	#inputArgs.add_argument( "--inDir", default=None, type=str, \
				#help="I/O Option2: Input folder name")
	#inputArgs.add_argument( "--outDir", default=None, type=str, \
				#help="I/O Option2: Output folder name")
	#inputArgs.add_argument( "--outSuff", default=None, type=str, \
				#help="I/O Option2: Output suffix")

	### I/O Option3: Batch I/O from list
	#inputArgs.add_argument( "--sList", default=None, type=str, \
				#help="I/O Option3: Scan list")
	

	flags, unknown = parser.parse_known_args()

	### Return flags and parser
	return flags, parser

	#ENDDEF ARGPARSER

def verify_flags(FLAGS, parser):

	##################################################
	### Check required args
	if FLAGS.task is None:
		parser.print_help(_sys.stderr)
		print('ERROR: Missing required arg --task')
		_sys.exit(1)

	else:
		print("Task is: %s" % (FLAGS.task))

	if FLAGS.task not in modelDict.keys():
		parser.print_help(_sys.stderr)
		print('Error: Task not found: ' + FLAGS.task)
		_sys.exit(1)
	

def _main():
	"""Main program for the script"""

	### init argv
	argv0 = [_sys.argv[0]]
	argv = _sys.argv[1:]

	### Read command line args
	FLAGS,parser = read_flags(argv)
	verify_flags(FLAGS, parser)


	if FLAGS.task == 'Hippo':
		
		## Add models in 3 orientation		
		argv = argv[2:]
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

	if FLAGS.task == 'DLICV':
		
		## Add models in 3 orientation		
		argv = argv[2:]
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
