# Deep Learning for Computer Vision

![](graphics/open_pose.gif)

**Deep Learning** has recently changed the landscape of computer vision (and other fields), and is largely responsible for a 3rd wave of interest and excitment about artificial intellgence. In this module we'll cover the basics of deep learning for computer vision. 

## Lectures

| Lecture | Notebook/Slides | Key Topics | Additional Reading/Viewing | 
| -------  | --------------- | ------------ | -------------------------- | 
| Introduction to Pytorch | [Notebook](https://github.com/unccv/deep_learning_for_computer_vision/blob/master/notebooks/introduction_to_pytorch.ipynb)| Why Pytorch?, Pytorch as "Numpy with GPU Support", simple neural network in Pytorch, automatic differentiation, nn.Module, PyTorch layers, PyTorch Optim, nn.Sequential | [Great Torch Intro by Jeremy Howard](https://pytorch.org/tutorials/beginner/nn_tutorial.html) |
| Get results fast with fastai | [Notebook](https://github.com/unccv/deep_learning_for_computer_vision/blob/master/notebooks/Get%20Results%20Fast%20with%20fastai.ipynb) | Jeremy Howard and the fastai philosophy, databunches, learners, classifiction, simplified object detection, semantic segmentation, multistak learning| [fastai course](https://github.com/fastai/course-v3)|
| Deep Learning Classification In Depth Part 1| [Notebook](https://github.com/unccv/deep_learning_for_computer_vision/blob/master/notebooks/image_classification_part_1.ipynb) |Stochastic gradient descent, regression vs classification, one hot encoding, cost functions and maximum likelihood, cross entropy | [Ian Goodfellow's Deep Learning - Chapter 1, Section 6.2, and Section 8.1](https://www.deeplearningbook.org/) |
| Deep Learning Classification In Depth Part 2| [Notebook](https://github.com/unccv/deep_learning_for_computer_vision/blob/master/notebooks/image_classification_part_2.ipynb) |CNNs, pooling and strides, AlexNet walkthrough, ImageNet, transfer learning, adaptive pooling, dropout, data augmentation, a little historical perspective | [AlexNet Paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)|
|  GANs | [Notebook](https://github.com/unccv/deep_learning_for_computer_vision/blob/master/notebooks/Generative%20Adversarial%20Networks.ipynb) | Ian Goodfellow invents GANs, the world's simplest GAN & nash equilibria, a dive into higher dimensions, DCGAN to the rescue, Visualizing GANs, GAN grow up (sortof), StyleGAN insanity, the unbelievably interesting world of GAN variants | [Goodfellow et al 2014](https://arxiv.org/pdf/1406.2661.pdf), [StyleGAN](https://arxiv.org/pdf/1812.04948.pdf), [CycleGAN](https://arxiv.org/pdf/1703.10593.pdf)|


## GPU Usage & Setup
A significant portion of the code in this repo (especially the fastai parts) will be painfully slow without a GPU. If you don't have access to a physical GPU machine, we recommend renting one. There are some really great/easy/affordable ways to do this, and with a couple platforms (Amazond Web Services, Google Cloud), you are likeley eldigable for free computer credits as a student. [fastai](https://course.fast.ai/start_salamander.html) has a really nice summary of the avaible cloud platforms for this type of thing, along with setup instructions. A number of services come with fastai already installed, which makes life even easier!


