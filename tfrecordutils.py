# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 12:23:39 2017

@author: jimit
"""

### Import modules
import tensorflow as tf
import numpy as np

#DEF
def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#ENDDEF
	
#DEF
def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
#ENDDEF

#DEF					
def tfrecordwriter( rname, rfeats, rlabels, feats_dtype, labels_dtype ):

	# open the TFRecords file
	options = tf.io.TFRecordOptions( compression_type='GZIP' )
	writer = tf.io.TFRecordWriter( rname, options=options )

	for i in range(len(rfeats)):
		# Load data
		img = rfeats[i].astype( feats_dtype )
		label = rlabels[i].astype( labels_dtype )

		# Create a feature
		feature = { 	'label': _bytes_feature( tf.compat.as_bytes( label.tostring() ) ),
                   		'image': _bytes_feature( tf.compat.as_bytes( img.tostring() ) ) }

		# Create an example protocol buffer
		example = tf.train.Example( features=tf.train.Features(feature=feature) )
        
		# Serialize to string and write on the file
		writer.write( example.SerializeToString() )
      
	writer.close()
#ENDDEF
