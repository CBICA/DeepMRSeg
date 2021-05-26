
################################################ DECLARATIONS ################################################
__author__ 	= 'Jimit Doshi'
__EXEC_NAME__ 	= "layers"

import tensorflow as _tf

################################################ FUNCTIONS ################################################

#DEF
def _int64_feature(value):
	return _tf.train.Feature(int64_list=_tf.train.Int64List(value=[value]))
#ENDDEF

#DEF
def _bytes_feature(value):
	return _tf.train.Feature(bytes_list=_tf.train.BytesList(value=[value]))
#ENDDEF

#DEF
def tfrecordwriter( rname, rfeats, rlabels, feats_dtype, labels_dtype ):

	# open the TFRecords file
	options = _tf.io.TFRecordOptions( compression_type='GZIP' )
	writer = _tf.io.TFRecordWriter( rname, options=options )

	for i in range(len(rfeats)):
		# Load data
		img = rfeats[i].astype( feats_dtype )
		label = rlabels[i].astype( labels_dtype )

		# Create a feature
		feature = { 	'label': _bytes_feature( _tf.compat.as_bytes( label.tostring() ) ),
                   		'image': _bytes_feature( _tf.compat.as_bytes( img.tostring() ) ) }

		# Create an example protocol buffer
		example = _tf.train.Example( features=_tf.train.Features(feature=feature) )

		# Serialize to string and write on the file
		writer.write( example.SerializeToString() )
      
	writer.close()
#ENDDEF
