# Monet-DCGAN

## Overview

#### A GAN consists of at least two neural networks: a generator model and a discriminator model. The generator is a neural network that creates the images. For this project, we are generating images in the style of Monet. The generator is trained using a discriminator.The two models work against each other, with the generator trying to trick the discriminator, and the discriminator trying to accurately classify the real vs. generated images.

#### The dataset contains four directories: monet_tfrec, photo_tfrec, monet_jpg, and photo_jpg. The monet_tfrec and monet_jpg directories contain 300 painting images 256 x 256. The photo_tfrec and photo_jpg directories contain 7,028 photos of size 256 x 256. The monet directories contain Monet paintings, we'll use these to train the model. The photo directories contain photos which we will add Monet-style to.

#### The data can be retrieved from: https://www.kaggle.com/competitions/gan-getting-started/data

## Observing the Data

#### After loading in the data for our model dataset, we want to observe some of the images. First is 12 images from the monet image set and then 12 from the photos dataset.

![image](https://github.com/friedunit/Monet-DCGAN/assets/10797098/4ead1b00-8bc3-48f5-a94d-d9e1c365bdbf)

## Building the models

#### For this project, I will implement a DCGAN, Deep Convolutional Generative Adversarial Network. I based most of this off of the Tensorflow tutorial from https://www.tensorflow.org/tutorials/generative/dcgan with modified image sizes and filters.

## First, the Generator

#### The generator uses tf.keras.layers.Conv2DTranspose (upsampling) layers to produce an image from a seed (random noise). Start with a Dense layer that takes this seed as input, then upsample several times until you reach the desired image size of 28x28x1. Notice the tf.keras.layers.LeakyReLU activation for each layer, except the output layer which uses tanh.

### Generating an image from the untrained Generator

![image](https://github.com/friedunit/Monet-DCGAN/assets/10797098/e03dbbb6-095e-4293-a5f9-59ec0e4564ab)

## Next, the Discriminator

#### The discriminator is a CNN-based image classifier that will classify the generated images as real or fake. The model will be trained to output positive values for real images and negative values for fake images.

## Train the Model

#### Below is the final sample of images after 200 epochs of the first model

![image](https://github.com/friedunit/Monet-DCGAN/assets/10797098/cb107206-256d-49a3-9ff9-02c68ebc0c8e)

## Modifying the Generator and Discriminator to retrain

#### Since the generated images above didn't really look like Monet paintings, I made changes for a second model to see if it does better. I added a couple layers in both the generator and discriminator. I changed the filter size from (5, 5) to (3, 3), added the alpha=0.2 to the Leaky ReLu activition which I saw in another tutorial. I also removed the use_bias=False argument in the Conv2D layers.

![image](https://github.com/friedunit/Monet-DCGAN/assets/10797098/e5748b42-1ed2-438e-af1b-1bb0af864a78)

## Findings and Conclusion

#### For this project, I decided to go with a DCGAN, but doing some other research and seeing what others did for the competition, a CycleGAN seems to produce better results. For applying a Monet type filter to photos, the CycleGan seems better, but the DCGAN did pretty well at learning what the Monet paintings look like and generating images that resemble it. With more time, modifications would be made for the generator and discriminator and ran for more epochs to produce better results. I think for my first time working with GANs, the results are pretty good andf I enjoyed learning and working on this project.



