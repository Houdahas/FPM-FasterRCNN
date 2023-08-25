# FPM-FasterRCNN
## [Investigating joint amplitude and phase imaging of stained samples in automatic diagnosis]

#### [Houda Hassini, Bernadette Dorizzi, Marc Thellier, Jacques Klossa and Yaneck Gottesman, "[Investigating joint amplitude and phase imaging of stained samples in automatic diagnosis]",

#### This repository contains the code for the experiments discussed in the article. These experiments include a complex-valued model that utilizes the real and imaginary parts of FPM images, a real-valued model that uses intensity and phase FPM images, and a real-valued model that solely utilizes intensity FPM images. This code is inspired by the code https://github.com/FurkanOM/tf-faster-rcnn of Fatser-RCNN implementation in tensorflow.

## Install the python environment 

### With conda env

create the conda environment for jodie: 
$ conda env create -f fasterrcnn-tensorflow-condaenv.yml

### Traning and testing 

The training data must be TFRecord format or the dataloader must be adapted to your dataset format.
The code allows to be used to train RPN only or all the faster RCNN 

