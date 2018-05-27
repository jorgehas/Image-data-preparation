from __future__ import print_function
import os, sys, tarfile
from six.moves.urllib.request import urlretrieve
import datetime
from os.path import basename


import numpy as np
from IPython.display import display, Image
from scipy import ndimage

import pickle 

def downloadFile(fileURL, expected_size):
	timeStampedDir=datetime.datetime.now().strftime("%Y.%m.%d_%I.%M.%S")
	os.makedirs(timeStampedDir)
	fileNameLocal = timeStampedDir + "/" + fileURL.split('/')[-1]
	print ('Attempting to download ' + fileURL)
	print ('File will be stored in ' + fileNameLocal)
	filename, _ = urlretrieve(fileURL, fileNameLocal)
	statinfo = os.stat(filename)
	if statinfo.st_size == expected_size:
		print('Found and verified', filename)
	else:
		raise Exception('Could not get ' + filename)
	return filename
