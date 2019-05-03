# What are Channels and Kernels ?

## Filters / Kernels:
**Filters** are used in **image processing** for blurring, sharpening, embossing, edge detection and more. In the initial layers of CNN, filters when convolved across an image captures low level features like vertical, horizontal and diagonal edges. Filter in the subsequent layers will be able to detect abstract and complex features. Each filter is a matrix of numbers corresponding to a feature that the filter is looking for. In CNN, filters are learned during training process. Each filter in a layer is randomly initialized to some distribution and hence will be able to detect separate features while training.

![alt+text](https://github.com/rinazbelhaj/EIP/blob/master/kernel.png?raw=true "Kernel")

## Channels:
**Channels** are features extracted by convolving **kernels** over an input image. Number of channels or features extracted equals the number of kernels used. More number of features extracted will help us in identifying images accurately. The channels are kept separate to avoid loss of information. Each channel carries specific set of information that will be merged together in final layers of neural network to correctly understand an image.

# Why should we only (well mostly) use 3x3 Kernels?


# How many times do we need to perform 3x3 convolution operation to reach 1x1 from 199x199 ?

We need to perform 3x3 convolution 99 times on the input image of size 199x199 to make it 1x1. ie, we need to have 99 hidden layers between input and output layers.

Calculation:

Input = 199,

Output = 1,

Filter Size = 3

No of layer required = (Input-Output)/(Filter_Size-1)
                     = (199-1)/(3-1)
                     = 198/2
                     = 99
                     

