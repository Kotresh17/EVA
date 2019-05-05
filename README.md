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

199x199 | layer(3x3) 1 |197x197 | layer(3x3) 2 |195x195 | layer(3x3) 3 | 193x193|    layer(3x3) 4 | 191x191 | layer(3x3) 5 | 189x189 | layer(3x3) 6 | 187x187 | layer(3x3) 7|185x185 | layer(3x3) 8 | 183x183 | layer(3x3) 9 | 181x181| layer(3x3) 10| 179x179|layer(3x3) 11 | 177x177 | layer(3x3) 12 | 175x175 | layer(3x3) 13 | 173x173 | layer(3x3) 14 |171x171| layer(3x3) 15 | 169x169 | layer(3x3) 16| 167x167 | layer(3x3) 17 | l65x165 |layer(3x3) 18| l63x163| layer(3x3) 19 | 161x161| layer(3x3) 20 | 159x159 | layer(3x3) 21 |157x157 | layer(3x3) 22 | 155x155 | layer(3x3) 23 | 153x153 | layer(3x3) 24 | 151x151 | layer(3x3) 25 | 149x149 | layer(3x3) 26 | 147x147 | layer(3x3) 27 | 145x145 | layer(3x3) 28 | 143x143| layer(3x3) 29 | 141x141 | layer(3x3) 30 | 139x139 | layer(3x3) 31 | 137x137 | layer(3x3) 32 | 135x135 | layer(3x3) 33 | 133x133 | layer(3x3) 34 | 131x131 | layer(3x3) 35 | 129x129 | layer(3x3) 36 | 127x127 | layer(3x3) 37 | 125x125 | layer(3x3) 38 | 123x123 | layer(3x3) 39 | 121x121 | layer(3x3) 40 | 119x119 | layer(3x3) 41 | 117x117 | layer(3x3) 42 | 115x115 | layer(3x3) 43 | 113x113 | layer(3x3) 44 | 111x111 | layer(3x3) 45 | 109x109| layer(3x3) 46 | 107x107 | layer(3x3) 47 | 105x105 | layer(3x3) 48 | 103x103 | layer(3x3) 49 | 101x101 | layer(3x3) 50 | 99 x 99 | layer(3x3) 51 | 97x97 | layer(3x3) 52 | 95x95 | layer(3x3) 53 | 93x93 | layer(3x3) 54 | 91x91| layer(3x3) 55 | 89 x89 | layer(3x3) 56 | 87x87 | layer(3x3)57 | 85x85 | layer(3x3) 58 | 83x83 | layer(3x3) 59 | 81x81 | layer(3x3) 60 | 79 x79 | layer(3x3) 61 | 77x77 | layer(3x3) 62 | 75x75 layer  3x3 63 | 73x73 layer  3x3 64 71x71 layer  3x3 65 | 69x69 layer  3x3 66 | 67x67 layer  3x3 67 | 65x65 layer  3x3 68| 63x63 layer  3x3 69| 61x61 layer  3x3 70| 59x59 layer  3x3 71 | 57x57 layer  3x3 72| 55x55 layer  3x3 73|53x53 layer  3x3 74| 51x51 layer  3x3 75 | 49x49 layer  3x3 76 | 47x47 layer  3x3 77 | 45x45 layer  3x3 78 | 43x43 layer  3x3 79 | 41x41 layer  3x3 80 | 39x39 layer  3x3 81 | 37x37 layer  3x3 82 |35x35 layer  3x3 83 | 33x33 layer  3x3 84 | 31x31 layer  3x3 85 | 29x29 layer  3x3 86 | 27x27 layer  3x3 87 | 25x25 layer  3x3 88 | 23x23 layer  3x3 89 | 21x21 layer  3x3 90 19x19 layer  3x3 91|17x17 layer  3x3 92 | 15x15 layer  3x3 93 | 13x13 layer  3x3 94 | 11x11 layer  3x3 95| 9x9 layer  3x3 96 | 7x7 layer  3x3 97 | 5x5 layer  3x3 98 | 3x3 layer  3x3 99| 1x1 layer  3x3 100

# assignment colab link https://colab.research.google.com/drive/1h3r0e4qfRNbkVLk5dc8rxMtUg5LqgGK4 
