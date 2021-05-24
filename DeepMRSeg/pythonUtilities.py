#!/usr/bin/env python

### Import required modules
import os, sys
import tempfile
import time

### Calculate the execution time for the script to finish
#DEF
def execution_time(executionTimestartTimeStamp):

	executionTimeendTimeStamp = time.time()
	executionTimetotal = (executionTimeendTimeStamp - executionTimestartTimeStamp) / 60

	print("\nExecution time: " + str(round(executionTimetotal, 2)) + " mins")
#ENDDEF

### Check if the input file exists
#DEF
def check_file(checkFileIP):
	if not os.path.exists(checkFileIP):
		print("\nERROR: Input file " + checkFileIP + " does not exist! Aborting operations ...")
		sys.exit(1)
#ENDDEF

### Get File Attributes
#DEF
def file_att(FileAttIP):
	
	ACCEPTED_FILE_TYPES = [ 'nii.gz', 'hdr', 'img', 'nii' ]

	if any( [FileAttIP.endswith(x) for x in ACCEPTED_FILE_TYPES] ):
	
		FileAttdName, FileAttfName = os.path.split(FileAttIP)
		FileAttbName, FileAttExt = os.path.splitext(FileAttfName)
		
		if len(FileAttdName) == 0:
			FileAttdName = os.getcwd()
		else:
			os.chdir(FileAttdName)
			FileAttdName = os.getcwd()
		
	
		if FileAttIP.endswith('nii.gz'):
			FileAttbName, FileAttExt_first = os.path.splitext(FileAttbName)
			FileAttExt = FileAttExt_first + FileAttExt
		
		return FileAttdName, FileAttbName, FileAttExt
	else:
		print("\nERROR: Input file extension not recognized! Please check ...")
		sys.exit(1)
#ENDDEF

### Creating temporary directory
#DEF
def create_temp_dir(tmpDirPref, tmpDir):

	# Create the parent directory if it is provided
	if tmpDir and not os.path.exists(tmpDir):
		os.makedirs(tmpDir)
	
	# Create TMP dir
	TMP = tempfile.mkdtemp( prefix=tmpDirPref, dir=tmpDir)
	
	return TMP
#ENDDEF
