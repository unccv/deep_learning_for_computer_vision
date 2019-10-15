## ---------------------------- ##
##
## eval.py
## Deep learning programming challenge evaluation code. 
## 
## ---------------------------- ##

import numpy as np
import time
from fastai.vision import *
from pathlib import Path
from sample_student import Model



def evaluate(data_path='../data/bbc_train',
             model_dir = '../sample_models',
             batch_size=64, 
             im_size=(256,256)):

    
    print("1. Loading Data...")

    path=Path(data_path)
    def get_y_fn(x): return path/'masks'/(x.stem + '.png')
    classes = array(['background', 'brick', 'ball', 'cylinder'])

    src = (SegmentationItemList.from_folder(path/'images')
       .split_by_rand_pct(0.0)
       .label_from_func(get_y_fn, classes=classes))

    #Don't normalize data here - assume normalization happens inside of Model. 
    data = src.transform(get_transforms(), tfm_y=True, size=im_size).databunch(bs=batch_size)
    print(data)


    print("2. Instantiating Model...")
    M = Model(path=model_dir, file='export.pkl')


    print("3. Evaluating on Test Images...")
    x,y = data.one_batch()
    #Extract class label from mask:
    class_labels=np.array([np.unique(y[i][y[i]!=0])[0] for i in range(x.shape[0])])

    #Extract bounding box from mask:
    bboxes=np.zeros((x.shape[0], 4))
    for i in range(x.shape[0]):
        rows,cols= np.where(y[i, 0]!=0)
        bboxes[i, :] = np.array([rows.min(), cols.min(), rows.max(), cols.max()])

    preds=M.predict(x)

    classification_accuracy=(np.array([data.classes[i] for i in class_labels])==np.array(preds[0])).sum()/len(preds[0])
    bbox_score = 1 - np.mean(np.abs(bboxes-preds[1]))/(x.shape[-1]) #Divide by image height to rougly normalize score
    segmentation_accuracy=float((preds[2] == np.array(y.squeeze(1))).sum())/y.numel()
    combined_accuracy=np.mean([classification_accuracy, bbox_score, segmentation_accuracy])

    return combined_accuracy, classification_accuracy, bbox_score, segmentation_accuracy


def calculate_score(combined_accuracy):
    if combined_accuracy >= 0.99: score = 10
    elif combined_accuracy >= 0.98: score = 9
    elif combined_accuracy >= 0.97: score = 8
    elif combined_accuracy >= 0.96: score = 7
    elif combined_accuracy >= 0.95: score = 6
    elif combined_accuracy >= 0.90: score = 5
    else: score = 4
    return score


if __name__ == '__main__':
    program_start = time.time()
    combined_accuracy, classification_accuracy, bbox_score, segmentation_accuracy = evaluate()
    score = calculate_score(combined_accuracy)

    program_end = time.time()
    total_time = round(program_end - program_start, 3)
    
    print("\nDone!")
    print("Execution time (seconds) = ", total_time)
    print("Score = ", score, 
         "\nCombined accuracy = ", combined_accuracy, 
         "\nClassification accuracy = ", classification_accuracy,
         "\nbbox score = ", bbox_score,
         "\nSegmentation Accuracy = ", segmentation_accuracy)







