#!/usr/bin/env python

"""Rescales images"""

################################################ DECLARATIONS ################################################
__author__ 	= 'Jimit Doshi'
__EXEC_NAME__ 	= "rescaleimages.py"

import numpy as _np

################################################ FUNCTIONS ################################################

#DEF	
def rescaleImage( image, minInt=0, maxInt=255, perc=99.9, method='minmax' ):
	""" Rescale image intensities to specified range

	Parameters
	----------
	image		: ndarray
			image data of the input image to be rescaled
	minInt		: int8
			output minimum intensity value (default: 0)
	maxInt		: int8
			output maximum intensity value (default: 255)
	perc		: float32
			percentile value to be scaled to `maxInt` (default: 99.9)
	method	 	: str
			rescale method { minmax, norm }

	Returns
	-------
	rescaledImage	: ndarray
			rescaled image
			
	References
	----------

	Examples
	--------

	"""
	
	### Convert image to float
	image = image.astype('float')
	
	### z-score image
	rescaledImage = ( image - _np.mean( image ) ) / _np.std( image )

	# IF METHOD	
	if method == 'minmax':
		### Shift minimum intensity first to minInt
		rescaledImage = rescaledImage - _np.min( rescaledImage ) + minInt

		### Rescale intensities so the max value in the image gets mapped to maxInt
		rescaledImage = rescaledImage / _np.percentile( rescaledImage, perc ) * maxInt

	# ENDIF METHOD
		
	### Return rescaledImage
	return rescaledImage
#ENDDEF
