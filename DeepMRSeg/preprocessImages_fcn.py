#!/usr/bin/env python


################################################ DECLARATIONS ################################################
__author__ 	= 'Jimit Doshi'
__EXEC_NAME__ 	= "preprocessImages"

import os as _os
import sys as _sys
import time as _time
import signal as _signal
import getopt as _getopt
import subprocess as _subprocess
import numpy as _np

from . import pythonUtilities

################################################ FUNCTIONS ################################################

# Usage info
#DEF
def help():
	"""Usage information."""
	print(r"""
%(EXEC)s--

This script does the following:

USAGE :	%(EXEC)s [OPTIONS]

OPTIONS:


	[--T1]	 	< file  >	absolute path to the input T1 scan
	[--FL]	 	< file  >	absolute path to the input FL scan

	[--dest] 	< path  >	absolute path to the destination directory where the files are to be stored
					( default: None, if None, then store in temporary directory )

	[--cost]	< str   >	cost function for flirt (default: mutualinfo)

	[--verb] 	< int   >	verbosity (default: 1)
	[--nJobs]	< int   >	number of jobs/threads (default: 4)
	
ERROR: Not enough arguments!!

	""" % {'EXEC':__EXEC_NAME__})

	_sys.exit(0)
#ENDDEF

### Define signal trap function
#DEF
def signal_handler(signal, frame):
	print('Program interrupt signal received! Aborting operations ...')
	_sys.exit(0)
#ENDDEF

#DEF
def preprocessImage( T1Img=None, FLImg=None, dest=None, n_jobs=4, verbose=0, cost='mutualinfo' ):

	### Set some environment variables
	_os.environ[ "FSLOUTPUTTYPE" ] = "NIFTI_GZ"

	### Get file attributes
	_,T1bName,_ = pythonUtilities.file_att( T1Img )

	###### Linearly registering T1 to FL
	print("\n\t---->	Linearly registering T1 to FL ...")
	_sys.stdout.flush()

	cmd = 'flirt' \
		' -in ' + T1Img + \
		' -ref ' + FLImg + \
		' -out ' + dest + '/' + T1bName + '_rFL.nii.gz' \
		' -omat ' + dest + '/' + T1bName + '_rFL.mat' \
		' -cost ' + cost + \
		' -v'
#		' -searchcost ' + cost + \

	#TRY
	try:
		if verbose == 1:
			print("\t\t-->	Executing : %s \n" % cmd)

		proc = _subprocess.Popen( [cmd], stdout=_subprocess.PIPE, shell=True )
		(out, err) = proc.communicate()

		if verbose == 1:
			print(out)

	except:
		print("\nERROR! Execution of this command failed: \n", cmd)
		print(err)
	finally:
		_sys.stdout.flush()
	#ENDTRY
#ENDDEF


############## MAIN ##############
#DEF
def _main( argv ):

	### Check the number of arguments
	if len( argv ) < 2:
		help()
		_sys.exit(0)

	### Timestamps
	startTime = _time.asctime()
	startTimeStamp = _time.time()

	### Print startTimeStamp
	print("\nHostname	: " + str( _os.getenv("HOSTNAME") ))
	print("Start time	: " + str( startTime ))

	_sys.stdout.flush()

	### Specifying the trap signal
	_signal.signal( _signal.SIGHUP, signal_handler )
	_signal.signal( _signal.SIGINT, signal_handler )
	_signal.signal( _signal.SIGTERM, signal_handler )

	### Default arguments
	T1 = None
	FL = None
	dest = None
	verbose = False
	cost = 'mutualinfo'

	nJobs = int(4)

	### Read command line args
	print("\nParsing args 	: ", argv[ 1: ])
	#TRY
	try:
		opts,_ = _getopt.getopt( argv[1:], "", \
				[ "T1=", "FL=", \
				"dest=", "cost=", \
				"verb=", "nJobs=" ] )

	except _getopt.GetoptError as err:
		print("\n\nERROR!", err)
		help()
	#ENDTRY

	_sys.stdout.flush()

	### Parse the command line args
	#FOR
	for o, a in opts:
		#IF
		if o in [ "--T1" ]:
			T1 = str(a)
			pythonUtilities.check_file( T1 )

			T1dName, T1bName, T1Ext = pythonUtilities.file_att( T1 )
			T1Img = T1dName + '/' + T1bName + T1Ext

		elif o in [ "--FL" ]:
			FL = str(a)
			pythonUtilities.check_file( FL )

			FLdName, FLbName, FLExt = pythonUtilities.file_att( FL )
			FLImg = FLdName + '/' + FLbName + FLExt

		elif o in [ "--dest" ]:
			dest = _os.path.realpath( str(a) )

		elif o in [ "--verb" ]:
			verbose = True
	
		elif o in [ "--nJobs" ]:
			nJobs = int(a)

		elif o in [ "--cost" ]:
			cost = str(a)

	#ENDFOR

	### preprocess Images
	print("\nStart preprocessing input images")
	preprocessImage( T1Img=T1Img, FLImg=FLImg, dest=dest, n_jobs=nJobs, \
				verbose=verbose, cost=cost )

	### Print resouce usage
	print("\nResource usage for this process")
	print("\tetime \t:", _np.round( (_time.time() - startTimeStamp)/60, 2 ), "mins")

	_sys.stdout.flush()
# ENDDEF MAIN
	


################################################ END OF FUNCTIONS ################################################
	
	
################################################ MAIN BODY ################################################

#IF
if __name__ == '__main__':
	_main( _sys.argv )
#ENDIF
