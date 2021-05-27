
################################################ DECLARATIONS ################################################
__author__ 	= 'Jimit Doshi'
__EXEC_NAME__ 	= "data_augmentation"

import tensorflow as _tf
import tensorflow_addons as _tfa
import numpy as _np

################################################ FUNCTIONS ################################################




#DEF
def perturb_images( img,lab ):

#	# flip left/right
#	randcond  = _tf.random.uniform( [], 0, 2, dtype=_tf.int32 )
#	img = _tf.cond( randcond>0, \
#			lambda: _tf.image.flip_left_right( img ), \
#			lambda: img );
#	lab = _tf.cond( randcond>0, \
#			lambda: _tf.image.flip_left_right( lab ), \
#			lambda: lab );

#	# flip up/down
#	randcond  = _tf.random.uniform( [], 0, 2, dtype=_tf.int32 )
#	img = _tf.cond( randcond>0, \
#			lambda: _tf.image.flip_up_down( img ), \
#			lambda: img );
#	lab = _tf.cond( randcond>0, \
#			lambda: _tf.image.flip_up_down( lab ), \
#			lambda: lab );

	# rotate with the random angle
	randcond  = _tf.random.uniform( [], 0, 2, dtype=_tf.int32 )
	randangle = _tf.random.uniform( [], -0.5, 0.5, dtype=_tf.float32 )
	img = _tf.cond( randcond>0, \
			lambda: _tfa.image.rotate( img, randangle, 'BILINEAR' ), \
			lambda: img );
	lab = _tf.cond( randcond>0, \
			lambda: _tfa.image.rotate( lab, randangle, 'NEAREST' ), \
			lambda: lab );

	# translate randomly
	randcond  = _tf.random.uniform( [], 0, 2, dtype=_tf.int32 )
	randtrans = _tf.random.uniform( [], -20, 20, dtype=_tf.float32 )
	img = _tf.cond( randcond>0, \
			lambda: _tfa.image.translate( img, [randtrans,randtrans], 'BILINEAR' ), \
			lambda: img );
	lab = _tf.cond( randcond>0, \
			lambda: _tfa.image.translate( lab, [randtrans,randtrans], 'NEAREST' ), \
			lambda: lab );

	# transpose randomly
	randcond  = _tf.random.uniform( [], 0, 2, dtype=_tf.int32 )
	randtheta_x = _tf.random.uniform( [], -0.9, 0.9, dtype=_tf.float32 )
	randtheta_y = _tf.random.uniform( [], -0.9, 0.9, dtype=_tf.float32 )
	randtrans = [ 1,_tf.sin(randtheta_x),0,0,_tf.cos(randtheta_y),0,0,0 ]
	img = _tf.cond( randcond>0, \
			lambda: _tfa.image.transform( img, randtrans, 'BILINEAR' ), \
			lambda: img );
	lab = _tf.cond( randcond>0, \
			lambda: _tfa.image.transform( lab, randtrans, 'NEAREST' ), \
			lambda: lab );

	# randomly change brightness
	randcond  = _tf.random.uniform( [], 0, 2, dtype=_tf.int32 )
	img = _tf.cond( randcond>0, \
			lambda: _tf.image.random_brightness( img, 0.5 ), \
			lambda: img );

	# randomly change contrast
	randcond  = _tf.random.uniform( [], 0, 2, dtype=_tf.int32 )
	img = _tf.cond( randcond>0, \
			lambda: _tf.image.random_contrast( img, 0.1, 2 ), \
			lambda: img );

	# add random gaussian noise
	randcond  = _tf.random.uniform( [], 0, 2, dtype=_tf.int32 )
	randnoisestd = _tf.random.uniform( [], 0.05, 0.25, dtype=_tf.float32 )
	img = _tf.cond( randcond>0, \
			lambda: _tf.add( img, _tf.random.normal( \
							shape=_tf.shape(img), \
							mean=0.0, \
							stddev=randnoisestd, \
							dtype=_tf.float32 ) ), \
			lambda: img );

#	# add random gamma
#	randcond  = _tf.random.uniform( [], 0, 2, dtype=_tf.int32 )
#	randgamma = _tf.random.uniform( [], 0.9, 1.1, dtype=_tf.float32 )
#	img = _tf.cond( randcond>0, \
#			lambda: _tf.pow( img,randgamma ), \
#			lambda: img );

#	# randomly sharpen the image
#	randcond  = _tf.random.uniform( [], 0, 2, dtype=_tf.int32 )
#	randnoisestd = _tf.random.uniform( [], 0.0, 1.0, dtype=_tf.float32 )
#	img = _tf.cond( randcond>0, \
#			lambda: _tfa.image.sharpness( img, randnoisestd ), \
#			lambda: img );

	return img,lab

#ENDDEF

#DEF
def data_reader( filenames,reader_func,batch_size,mode ):

	# Generate TFRecordDataset
	ds = _tf.data.TFRecordDataset( filenames, compression_type='GZIP', \
					num_parallel_reads=_tf.data.experimental.AUTOTUNE )
	print(ds)

	# Shuffle data
	if mode == _tf.estimator.ModeKeys.TRAIN:
		ds = ds.shuffle( buffer_size=100, reshuffle_each_iteration=True )
		print(ds)

	# Read TFRecords
	ds = ds.map( map_func=reader_func, num_parallel_calls=_tf.data.experimental.AUTOTUNE )
	print(ds)

	# Shuffle extracted data
	if mode == _tf.estimator.ModeKeys.TRAIN:
		ds = ds.shuffle( buffer_size=2000, reshuffle_each_iteration=True )
		print(ds)		

	# Batch input data
	if mode == _tf.estimator.ModeKeys.TRAIN:
		ds = ds.batch( batch_size,drop_remainder=True )
	else:
		ds = ds.batch( batch_size,drop_remainder=False )
	print(ds)				

	# Randomly augment data
	if mode == _tf.estimator.ModeKeys.TRAIN:
		ds = ds.map( map_func=perturb_images, num_parallel_calls=_tf.data.experimental.AUTOTUNE )
		print(ds)		

	# Prefetch data
	ds = ds.prefetch( _tf.data.experimental.AUTOTUNE )
	print(ds)

#	# Cache
#	ds = ds.cache()
#	print(ds)

	return ds
#ENDDEF



#DEF
def permute_images( img,mods ):

	# Permute each sample in the batch
	# FOR
	for s in range( img.shape[0] ):
	
		# Shuffle order of modalities for this sample
		arr = img[ s ]
		arr_sw = _np.swapaxes( arr,-1,0 )
		_np.random.shuffle( arr_sw )
		
		# Randomly turn modalities zero with probability of 50%
		img[ s ] = img[ s ] * _np.random.randint( 2,size=mods )
	# ENDFOR
	
	return img

#ENDDEF

