# Deep Learning Programming Challenge


![](../graphics/bbc1k.gif)

## About This Challenge

In the summer of 1966, Marvin Minsky and Seymour Paper, giants of Artifical Intelligence, launched the 1966 MIT Summer Vision Project: 

![](../graphics/summer_project_abstract-01.png)

Minsky and Papert assigned Gerald Sussman, an MIT undergraduate studunt as project lead, and setup specific goals for the group around recognizing specific objects in images, and seperating these objects from their backgrounds. 

![](../graphics/summer_project_goals-01.png)

Just how hard is it to acheive the goals Minsky and Papert laid out? How has the field of computer vision advance dsince that summer? Are these tasks trivial now, 50+ years later? Do we understand how the human visual system works? Just how hard *is* computer vision and how far have we come?

In this challenge, you'll use a modern tool, **deep neural networks**, and a labeled dataset to solve a version of the MIT Summer Vision Project problem.  

## Data
You'll be using the bbc-1k dataset, which contains 1000 images of bricks, balls, and cylinders against cluttered backgrounds. You can download the dataset [here](http://www.welchlabs.io/unccv/deep_learning/bbc_train.zip), or with the download script in the util directory of this repo:

```
python util/get_and_unpack.py -url http://www.welchlabs.io/unccv/deep_learning/bbc_train.zip
```

![](../graphics/bbc_sample.jpg)

The BBC-1k dataset includes ~1000 images including classification, bounding box, and segmentation labels. Importantly, each image only contains one brick, ball or cylinder. 


## Packages
You are permitted to use numpy, opencv, tdqm, time, pytorch, fastai, opencv, and scipy.

## Your Mission 

Your job is to design and train a multitask deep learning model in fastai & pytorch to solve 3 different problems simultaneously: classification, detection, and segmentation. Please see `sample_student.py` and `eval.py` for more detailed instructions. A few tips: 

1. You will need to create a fastai dataloader the gives you all the labels you need (classifiction, bounding boxes, and segmentation). `eval.py` provides an example of one way to do this, by using fastai to load the segmentation mask, and then loading bounding boxes and classification labels from the mask. 
2. See the notebook `Get Results Fast with fastai.ipynb` for some examples to get you going, including a basic fastai multiclass network. 
3. It's worth spending some time thinkg through how you should "split apart" your network to solve multiple tastks. 


## Deliverables (Waiting for 2019 Updates)

1. Your modified version of `sample_student.py`. 
2. fastai export of our model file as a `.pkl` file, exported with `learn.export()`. 

## Submission

For this assignment you will upload your solutions to your Dropbox and then submit the link to your drive or dropbox file in Autolab. Here is how:
1. Download the submission folder from [here](https://drive.google.com/file/d/14ZHoUJI6xcnNWWmU2AhkyCvrq2u6k7j-/view?usp=sharing)
2. Extract the downloaded handin.zip. Delete handin.zip.
3. The extracted handin folder has the following structure:  
   /handin  
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; /export.pkl   
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; /sample_student.py  
    
4. Replace /handin/export.pkl with **your** export.pkl(trained weights). Keep the file name export.pkl
5. Replace /handin/sample_student.py with **your** sample_student.py. Keep the file name sample_student.py
6. Zip the handin folder.
7. Upload your handin.zip to your Dropbox account.
8. In Dropbox, select your handin.zip and click on Share button.  
   Make sure the sharing permission is "Anyone with the link can view this file".  
   Click on "copy link".  
   Your link should look like https://www.dropbox.com/s/u23i6mgjkqcr3lh/handin.zip?dl=0.  
   **FILE SIZE LIMIT FOR DROPBOX SUBMISSON IS 300 MB**
9. Open Notepad or any text editor & paste your link.Then save the file as handin.txt
10. Login to your Autolab account. Go to Deep Learning challenge and submit this handin.txt file.

Notes:  
  * Do not share your submission link with anyone else.   
  * Keep the directory structure and name of the folder same as provided.  
  * You do not need to submit any other .py file other than sample_student.py. Other py files provided with this challenge will be avaialble to your sample_student.py if needed.  
  * **Please upload your final submission to canvas i.e your handin.zip. It is essential for this challenge to make a submission on Canvas. Missing submissions on Canvas will incur a penalty of 2 points for this challenge.**


## Grading
Your model will be evaluated on a true hold out set of ~200 images, and your grade will be determined by your combined accuracy on this set. Your combined accuracy will be computed as the average of your classification_accuracy, bbox_score, and segmentation_accuracy - see `eval.py` for more details. 

| Accuracy (%) | Points (10 max)  | 
| ------------- | ------------- | 
| Accuracy >= 97     | 10  | 
| 96 <= Accuracy < 97 | 9  |  
| 95 <= Accuracy < 96 | 8  |   
| 94 <= Accuracy < 95 | 7  |   
| 93 <= Accuracy < 94 | 6  |   
| 90 <= Accuracy < 93 | 5  |  
| Accuracy < 90, or code fails to run | 4  |  

