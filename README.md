# Deep Learning for Computer Vision

![](graphics/open_pose.gif)

**Deep Learning** has recently changed the landscape of computer vision (and other fields), and is largely responsible for a 3rd wave of interest and excitment about artificial intellgence. In this module we'll cover the basics of deep learning for computer vision. 

## Lectures

| Lecture | Notebook/Slides | Key Topics | Additional Reading/Viewing | 
| -------  | --------------- | ------------ | -------------------------- | 
| Introduction to Pytorch | [Notebook](https://github.com/unccv/deep_learning_for_computer_vision/blob/master/notebooks/01-introduction-to-pytorch.ipynb)| Why Pytorch?, Pytorch as "Numpy with GPU Support", simple neural network in Pytorch, automatic differentiation, nn.Module, PyTorch layers, PyTorch Optim, nn.Sequential | [Great Torch Intro by Jeremy Howard](https://pytorch.org/tutorials/beginner/nn_tutorial.html) |
| Get results fast with fastai | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unccv/deep_learning_for_computer_vision/blob/master/notebooks/02-image-classification-with-fastai.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unccv/deep_learning_for_computer_vision/blob/master/notebooks/03-bounding-box-detection-with-fastai.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unccv/deep_learning_for_computer_vision/blob/master/notebooks/04-semantic-segmentation-with-fastai.ipynb) | Jeremy Howard and the fastai philosophy, databunches, learners, classifiction| [fastai course](https://github.com/fastai/course-v3)|
| Deep Learning In Depth Part 1| [Notebook](https://github.com/unccv/deep_learning_for_computer_vision/blob/master/notebooks/05-deep-learning-in-depth-1.ipynb) |Stochastic gradient descent, regression vs classification, one hot encoding, cost functions and maximum likelihood, cross entropy | [Ian Goodfellow's Deep Learning - Chapter 1, Section 6.2, and Section 8.1](https://www.deeplearningbook.org/) |
| Deep Learning In Depth Part 2| [Notebook](https://github.com/unccv/deep_learning_for_computer_vision/blob/master/notebooks/06-deep-learning-in-depth-2.ipynb) |CNNs, pooling and strides, AlexNet walkthrough, ImageNet, transfer learning, adaptive pooling, dropout, data augmentation, a little historical perspective | [AlexNet Paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)|
|  GANs | [Notebook](https://github.com/unccv/deep_learning_for_computer_vision/blob/master/notebooks/07-generative-adversarial-networks.ipynb) | Ian Goodfellow invents GANs, the world's simplest GAN & nash equilibria, a dive into higher dimensions, DCGAN to the rescue, Visualizing GANs, GAN grow up (sortof), StyleGAN insanity, the unbelievably interesting world of GAN variants | [Goodfellow et al 2014](https://arxiv.org/pdf/1406.2661.pdf), [StyleGAN](https://arxiv.org/pdf/1812.04948.pdf), [CycleGAN](https://arxiv.org/pdf/1703.10593.pdf)|


## Setting Up Your Computing Environment
Installing the software you need to train deep learning models can be difficult. For the purposes of this workshop, we're offering 3 recommended methods of setting up your computing environment. Your level of experience and access to machines, should help you determine which appraoch is right for you. 

| | Option | Pros | Cons | Cost | Instructions | 
| - | ------ | ---- | ---- | ---- | ------------ | 
| 1 | Google Colab | Virtually no setup required, start coding right away! | GPUs not always available, limited session times, limited RAM | Free! There's also a paid tier at [$10/month](https://colab.research.google.com/signup) | [Colab Setup](https://github.com/stephencwelch/dsgo-dl-workshop-summer-2020#21-setup-google-colab) |
| 2 | Your Own Linux GPU Machine | No recurring cost, complete control over hardware. | High up-front cost, takes time to configure. | $1000+ fixed up front cost | [Linux Setup](https://github.com/stephencwelch/dsgo-dl-workshop-summer-2020#22-setup-on-your-own-gpu-machine-running-linux) |
| 3 | Virtual Machine | Highly configurable & flexible, pay for the performance level you need | Can be difficult to configure, only terminal-based interface | Starts ~$1/hour | [VM Setup](https://github.com/stephencwelch/dsgo-dl-workshop-summer-2020#23-setup-a-virtual-machine) |

### 1. Setup Google Colab
Google colab is delightfully easy to setup. All you really need to is a google account. Clicking one of the "Open in Colab" links above should take you directly to that notebook in google colab, ready to run. The only configuration change you'll be required to make is **changing your runtime type**. Simply click the runtime menu dropdown at the top of your notebook, select "change runtime type", and select "GPU" as your hardware accelerator. You also can open any notebook available on github in colab. 

### 2. Setup on Your Own GPU Machine Running Linux
After doing this for a while, my preferred configuration is training models on my own Linux GPU machine. This can require some up front investment, but if you're going to be training a lot of models, having your own machine really makes your life easier. 

2.1 Install [Anaconda Python](https://www.anaconda.com/products/individual)

2.2 Clone this repository
```
git clone https://github.com/unccv/deep_learning_for_computer_vision

```

2.3 (Optional) Create conda environment
```
conda create -n unccv-dl python=3.7
conda activate unccv-dl
```

2.4 Install packages
```
cd deep_learning_for_computer_vision
pip install -r requirements.txt
```

2.5
```
Depending on your existing setup, you may need to install nvidia drivers and the [nvidia cuda toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html). 
```

2.6 Launch Jupyter
```
jupyter notebook
```

### 2.3 Setup a Virtual Machine
Virtual machines provide a nice highly configurable platform for data science tasks. I recommnend following the [fastai server setup](https://course.fast.ai/start_azure.html) guide for the cloud platform of your choice. 
