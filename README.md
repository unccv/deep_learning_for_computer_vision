# Deep Learning for Computer Vision

**Deep Learning** has recently changed the landscape of computer vision (and other fields), and is largely responsible for a 3rd wave of interest and excitment about artificial intellgence. In this module we'll cover the basics of deep learning for computer vision. 

## Lectures

| Lecture | Notebook/Slides | Key Topics | Additional Reading/Viewing | 
| -------  | --------------- | ------------ | -------------------------- | 
| Introduction to Pytorch | [Notebook](https://github.com/unccv/deep_learning_for_computer_vision/blob/master/notebooks/introduction_to_pytorch.ipynb)| Why Pytorch?, Pytorch as "Numpy with GPU Support", simple neural network in Pytorch, automatic differentiation, nn.Module, PyTorch layers, PyTorch Optim, nn.Sequential | [Great Torch Intro by Jeremy Howard](https://pytorch.org/tutorials/beginner/nn_tutorial.html) |
| Get results fast with fastai | Notebook| Jeremy Howard and the fastai philosophy, DataBunches, Learners, NLP with fastai, world class computer vision with fastai | [fastai course](https://github.com/fastai/course-v3)|
|  GANs | Ian Goodfellow invents GANs, the world's simplest GAN & nash equilibria, a dive into higher dimensions, DCGAN to the rescue, Visualizing GANs, GAN grow up (sortof), StyleGAN insanity, the unbelievably interesting world of GAN variants | |



## GPU Usage & Setup
A significant portion of the code in this repo (especially the fastai parts) will be painfully slow without a GPU. If you don't have access to a physical GPU machine, we recommend renting one. There are some really great/easy/affordable ways to do this, and with a couple platforms (Amazond Web Services, Google Cloud), you are likeley eldigable for free computer credits as a student. [fastai](https://course.fast.ai/start_salamander.html) has a really nice summary of the avaible cloud platforms for this type of thing, along with setup instructions. A number of services come with fastai already installed, which makes life even easier!


