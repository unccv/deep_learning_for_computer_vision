## ---------------------------- ##
##
## eval.py
## Deep learning programming challenge evaluation code. 
## 
## ---------------------------- ##

import numpy as np
from fastai.vision import *
from pathlib import Path
from sample_student import Model

def get_y_fn(x): 
	return path/'masks'/(x.stem + '.png')

def evaluate(data_path='../data/bbc_train',
             model_dir = '../models',
             batches_to_test=8,
             batch_size=16, 
             im_size=(256,256)):

    
    print("1. Loading Data...")
    path=Path(data_path)
	classes = array(['background', 'brick', 'ball', 'cylinder'])

	src = (SegmentationItemList.from_folder(path/'images')
       .split_by_rand_pct(0.0)
       .label_from_func(get_y_fn, classes=classes))

	#Don't normalize data here - assume normalization happens inside of Model. 
	data = (src.transform(get_transforms(), tfm_y=True, size=im_size).databunch(bs=batch_size)
	print(data)


    print("2. Instantiating Model...")
    M = Model(path='../models', file='export.pkl'))

    print("3. Evaluating on Test Images...")





    return accuracy
