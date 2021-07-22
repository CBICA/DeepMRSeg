#!/usr/bin/python3

# deepmrseg

import argparse as _argparse
import os as _os
import urllib.request
from urllib.parse import urlparse
import zipfile
import sys as _sys

##############################################################
## This is a dictionary that keeps the saved models for now
mdlurl = 'https://github.com/CBICA/DeepMRSeg-Models/raw/main/models'

modelDict = {}
modelDict['dlicv'] = mdlurl + '/DLICV/DeepMRSeg_DLICV_v1.0.zip'

##############################################################


## Path to saved models
DEEPMRSEG = _os.path.expanduser(_os.path.join('~', '.deepmrseg'))
MDL_DIR = _os.path.join(DEEPMRSEG, 'trained_models')

def _main():
    """Main program for the script to load pre-trained models."""

    parser = _argparse.ArgumentParser(formatter_class=_argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--model", default=None, type=str, help="name of the model")
    inargs = parser.parse_args()


    if inargs.model not in modelDict.keys():
        #print('Usage : ' + _os.path.basename(_sys.argv[0]) + ' model_name')
        parser.print_help( _sys.stderr )
        print('\nAvailable models: ' + ','.join(modelDict.keys()))        
        _sys.exit(1)
    
    else:

        mdlurl = modelDict[inargs.model]
        mdlfname = _os.path.basename(urlparse(mdlurl).path)
        outFile = _os.path.join(MDL_DIR , inargs.model, mdlfname)

        if _os.path.isdir(outFile.replace('.zip', '')):
            print("Model already downloaded, aborting: " + outFile.replace('.zip', ''))
        
        else:
            
            print("Loading model: " + inargs.model)
            
            outPath = _os.path.join(MDL_DIR , inargs.model)
            if not _os.path.exists(outPath):
                _os.makedirs(outPath)
                print('Created dir : ' + outPath)
            

            urllib.request.urlretrieve(mdlurl, outFile)
            print('Downloaded model : ' + outFile)
        
            with zipfile.ZipFile(outFile, 'r') as fzip:
                fzip.extractall(outPath)
            print('Unzipped model : ' + outFile.replace('.zip', ''))
            
            _os.remove(outFile)
            #print('Removed zip file : ' + outFile)
    

if __name__ == "__main__":
    main()
