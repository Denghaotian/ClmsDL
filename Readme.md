# Bully picture classification 
CPSC8810 Deep Learning Term Project

## Authors
Qingbo Lai   

qingbol@clemson.edu 

Haotian Deng 

hdeng@clemson.edu

## Note
This project construct and implement Convolutional Neural Networks (CNNs)
to classify the bully picture. All codes were implementated and tested on 
Palmetto www.palmetto.clemson.edu

## Prerequisites
Python3.6; TensorFlow framework 1.12

## Network Structure
We have two different networks structure. one is simple three layers CNN 
model with two fully connected layers which written by ourselves, the 
another model is based VGG16 with some changes by ourselves.

## Training Strategy
We used ten categories images to train model. Nine categories of bully 
images which are laughing, pullinghair, quarrel, slapping, punching, 
stabbing, gossiping, strangle and isolation. The rest of images are 
nonbullying category. 

## Usages
Default location of training data : data_bully/training_data

Default location of testing data: data_bully/testing_data

If you want to train model please make directory "data_bully"
### Train

Command line: python bully_train.py --train_path "path-to-training-dataset"

## Predict and Test

Please download pre-trained model from Google drive which provided by TA,
because the files are too large to upload to github.
After downlaod pre-trained model, please unconpress and put it to "trained_model"
directory.

### predict a single image


python predict.py --img_file "path-to-img/xxx.jpg"
### test the accuracy for testing dataset 
Testing 10 groups of classified images by the tagged file directory
like laughing, pullinghair, quarrel, slapping, punching, stabbing, 
gossiping, strangle, isolation and nonbullying. The output will be
the accuracy of testing files.


Command line: python test.py --test_path "path-to-testing-datase"




## Reference

[1]Stanford 40 actions. In Stanford 40 Actions. http://vision.stanford.edu/Datasets/40actions.html


[2] Girdhar, R. and Ramanan, D. (2017). Attentional pooling for action recog-
nition. In Guyon, I., Luxburg, U. V., Bengio, S., Wallach, H., Fergus, R.,
Vishwanathan, S., and Garnett, R., editors, Advances in Neural Information
Processing Systems 30, pages 34{45. Curran Associates, Inc.


[3] Gkioxari, G., Girshick, R., and Malik, J. (2015). Contextual action recogni-
tion with r*cnn. In The IEEE International Conference on Computer Vision
(ICCV).

[4] palmetto. In https://www.palmetto.clemson.edu/palmetto/.

[5] Qassim, H., Verma, A., and Feinzimer, D. (2018). Compressed residual-vgg16
cnn model for big data places image recognition. In 2018 IEEE 8th Annual
Computing and Communication Workshop and Conference (CCWC), pages
169{175.

[6] Sharif Razavian, A., Azizpour, H., Sullivan, J., and Carlsson, S. (2014). Cnn
features o-the-shelf: An astounding baseline for recognition. In The IEEE
Conference on Computer Vision and Pattern Recognition (CVPR) Work-
shops.
