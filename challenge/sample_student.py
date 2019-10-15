## ---------------------------- ##
##
## sample_student.py
## Example student submission code for deep learning programming challenge. 
## You are free to use fastai, pytorch, opencv, and numpy for this challenge.
##
## Requirements:
## 1. Your code must be able to run on CPU or GPU.
## 2. Must handle different image sizes. 
## 3. Use a single unified pytorch model for all 3 tasks. 
## 
## ---------------------------- ##

from fastai.vision import load_learner, normalize
import numpy as np

class Model(object):
	def __init__(self, path='../models', file='export.pkl'):
		
		self.learn=load_learner(path=path, file=file) #Load model
		self.class_names=['brick', 'ball', 'cylinder']

	def predict(self, x):
		'''
		Input: x = block of input images, stored as Torch.Tensor of dimension (batch_sizex3xHxW), 
				   scaled between 0 and 1. 
		Returns: a tuple containing: 
			1. The final class prediction for the image (brick, ball, or cylinder) as a string.
			2. Upper left and lower right bounding box coordinates (in pixels) for the brick ball 
			or cylinder, as a 1d numpy array of length 4. 
			3. Segmentation mask for the image, as a 2d numpy array of dimension (HxW). Each value 
			in the segmentation mask should be either 0, 1, 2, or 3. Where 0=background, 1=brick, 
			2=ball, 3=cylinder. 
		'''

		#Normalize input data using the same mean and std used in training:
		x_norm=normalize(x, torch.tensor(learn.data.stats[0]), 
							torch.tensor(learn.data.stats[1]))


		#Pass data into model:
		yhat=self.learn.model(x_norm)

		#Post-processing/parsing outputs





		#Random Selection Placeolder Code
		class_prediction=self.class_names[np.random.randint(3)]

		#Scale randomly chosen bbox coords to image shape:
		bbox=np.random.rand(4)
		bbox[0] *= x.shape[2]; bbox[2] *= x.shape[2] 
		bbox[1] *= x.shape[3]; bbox[3] *= x.shape[3]

		#Create random segmentation mask:
		mask=np.random.randint(low=0, high=4, size=(x.shape[0], x.shape[1]))

		return (class_prediction, bbox, mask)






