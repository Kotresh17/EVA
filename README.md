# **Assignment 1B**
## Kernel 
<p align="center">
  <img width="460" height="300" src="https://mlblr.com/images/4-2ConvolutionSmall.gif">
</p>
Kernel in convolutional neural networks is a simple n x n matrix which is used to convolve on a base input image to provide a feature map also called channel. In Convolutional Neural Networks(CNN) these kernels are defined to extract feature from the input image and these kernel values are auto tuned by backpropagation algorithm. examples for kernels are like 3 X 3, 5 X 5, 7 X 7, 1 X 1 and etc now a days community is keeping most of these kernel dimension to 3 X 3. kernels are like basic building blocks for CNN to build feature maps or an output channel.

## Channels or Feature maps
<p align="center">
  <img width="460" height="300" src="https://i.stack.imgur.com/ZgG1Z.png">
</p>
Channel or a Feature map is the output of convolution on an input image. if there are n number of channels those are the output of n number of kernels which are convolved. channels are like a bank of convolved feature map. 

## Making use of 3 X 3 kernels mostly, but why?
While training a Convolutional Neural Networks we need to be more specific with computation speed and memory usage. lets talk about computation if we take an image of resolution size 5 X 5 if we convolved with 3 x 3 kernel then we get a feature map of 3 X 3. and the number of trainable parameters are 9. and if we convolved one more time with 3 X 3 then output feature map would be of size 1 X 1. total number of trainable parameters are 9 + 9 = 18 and. suppose if we use 5 X 5 on the 5 X 5 image the output feature map would be 1 X 1 and total number of trainable parameters are 25. so 3 X 3 is probably better than 5 X 5 for extracting the very low level information. and also occupy less memory also now a days GPU's are optimized to run better on 3 X 3 matrix. I think this image would probably help.
<p align="center">
  <img width="460" height="300" src="https://cdn-images-1.medium.com/max/2400/1*LnMqoqcDp02as5OOzaBcLQ.png">
</p>
Image source ![3X3 Convolution](https://blog.sicara.com/about-convolutional-layer-convolution-kernel-9a7325d34f7d) 


number of times we need to perform 3 X 3 convolution operation to reach 1 X 1 from 199 X 199.


assignment colab link https://colab.research.google.com/drive/1h3r0e4qfRNbkVLk5dc8rxMtUg5LqgGK4 
