## ---------------------------- ##
##
## eval.py
## Deep learning programming challenge evaluation code. 
## 
## ---------------------------- ##

import numpy as np
from fastai.vision import *
from sample_student import Model

def evaluate(data_path=''
             model_dir = 'tf_data/sample_model'
             batches_to_test=
             batch_size=):

    
    print("1. Loading Data...")
    data = data_loader(label_indices = label_indices, 
               		   channel_means = channel_means,
               		   train_test_split = 0.5, 
               		   data_path = data_path)

    print("2. Instantiating Model...")
    M = Model(mode = 'test')

    #Evaluate on test images:
    GT = Generator(data.test.X, data.test.y, minibatch_size = minibatch_size)
    
    num_correct = 0
    num_total = 0
    
    print("3. Evaluating on Test Images...")
    for i in range(num_batches_to_test):
        GT.generate()
        yhat = M.predict(X = GT.X, checkpoint_dir = checkpoint_dir)
        correct_predictions = (np.argmax(yhat, axis = 1) == np.argmax(GT.y, axis = 1))
        num_correct += np.sum(correct_predictions)
        num_total += len(correct_predictions)
    
    accuracy =  round(num_correct/num_total,4)

    return accuracy
