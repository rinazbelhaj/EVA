#  Architectural Basics

Properly designing a neural network is of uttermost importance in deep learning. There is a widespread belief that deep neural networks are magic boxes which ingest data and come up with intricate features and variables to solve the problem in hand. This is not entirely true. A neural network has many parameters and infinite ways in which they can interact with each other, which means there are infinitely many ways to design a network. This makes a complete grid search over the entire parameter space infeasible. So the best approach would be to hand engineer the network in terms of the number of layer, type of layers and many more other parameters.

Designing a good model usually involves a lot of trial and error. It is still more of an art than science, and people have their own ways of designing models. The basic thing to keep in mind before finalizing on architecture is to check whether it is capable of handling the complexity of the problem in hand.

We will see a few steps and tips that we need to consider while designing a convolutional neural network architecture for an image classification problem. We are still in an early phase of deep learning, which means the list is not exhaustive.

The following steps are to be followed in the given order.

## 1. Receptive Field :
The first and foremost thing in designing a convolutional neural network is to understand the receptive field required it. **Receptive field** is a portion of sensory space that can elicit neuronal responses when stimulated. More specifically, it is defined as the region in the input space that a particular CNN’s feature is looking at. The final layer of the network should be able to see the entire image to say something about it, hence the global receptive field at the final layer should be as big as the image. In case, if the objects of interest are always smaller than the image, then the required receptive field needs to be as big as the object in the image. Basically, the network should be able to see the full picture before commenting on it.

Eg: Let's say we are doing an image classification problem on MNIST data with image sizes of 28x28. So ideally, we need to have a receptive field of 28x28. But if you closely examine the images, we can see that the numbers in the images are not of full size. The edges of the images have blank spaces which is irrelevant for correctly identifying the label of the image. By further analysis, we can see that the receptive field required for this problem would around 23x23.

## 2. Convolution Size :
Once we have finalized on the receptive field requirement, then the next task is to decide on the convolutional filter size. **Filters**  or **kernels** are used in image processing for blurring, sharpening, embossing, edge detection and more. In the initial layers of CNN, filters, when convolved across an image, captures low-level features like vertical, horizontal and diagonal edges. Filter in the subsequent layers will be able to detect abstract and complex features. Each filter is a matrix of numbers corresponding to a feature that the filter is looking for. They can be any sizes like **1x1, 3x3, 5x5, 7x7, 11x11** etc. Usually, we don't use even sized filter since they don't have an axis of symmetry and hence less useful in extracting features. We mostly use 3x3 filters when designing the network due to the following advantages of 3x3 over others.

### Why should we mostly use 3x3 Kernels?

3x3 Convolution** is a type of **filter/kernel** used in **deep CNNs**. It is basically a matrix of size **3x3xinput_channel** convolved over the previous input layers. 

![alt+text](https://github.com/rinazbelhaj/EVA/blob/master/Images/3x3%20Convolution.png?raw=true "3x3 Convolution")

1.  **3x3 kernels have least number of parameters [9 Parameters]**</br>
    3x3 kernels are the smallest possible kernel that can detect patterns in an image. This has 9 parameters which need to be optmized       while training. Any higher order filter can be represented as a multiple of 3x3 filters thereby reducing the parameters. Fewer           parameters result in the network to  be trained faster.</br>
    
    Eg : 5x5 filter can be represented as a series of two 3x3 filters. </br>
    5x5 filter : 25 parameters </br>
    3x3 filter x 2 : 9 x 2 = 18 parameters </br>
2.  **3x3 kernels are accelerated on GPUs and TPUs** </br>
    Since most of the researchers and companies using CNNs use 3x3 filter, hardware manufacturers have optimized their GPUs and TPUs to     perform faster operations on these filters. This resulted in more and more people using 3x3 filter more.

## 3. MaxPooling :
**Max pooling** is a sample-based discretization process. The objective is to down-sample an input representation, reducing its dimensionality and allowing for assumptions to be made about features contained in the sub-regions binned. Max pooling is done by applying a max filter to (usually) non-overlapping subregions of the initial representation.

It is common to periodically insert a max Pooling layer in-between successive Conv layers in a ConvNet architecture. Its function is to progressively reduce the spatial size of the representation to reduce the amount of parameters and computation in the network, and hence to also control overfitting. The Pooling Layer operates independently on every depth slice of the input and resizes it spatially, using the MAX operation. The most common form is a pooling layer with filters of size 2x2 applied with a stride of 2 downsamples every depth slice in the input by 2 along both width and height, discarding 75% of the activations. 

![alt+text](https://github.com/rinazbelhaj/EVA/blob/master/Images/image.png?raw=true "3x3 Convolution")

### Position of MaxPooling layer:
Since max pooling layer does a discretization to abstract the image, it is very important to decide on the position of pooling layer in the network so that we are loosing minimum information.

1.  **Distance of the pooling layer from input**</br>
The max pooling layer should not be used close to the input layer. This is because the receptive field in initial few layers will be very small to generate any edges or gradients and therefore max pooling won't be able to pick out any relevant features from those layers.

Eg: In MNIST data, the first three layers with 3x3 kernels will have a receptive field of 7x7. Before the third layer, the receptive field will be too low for kernel to identify any edges or gradients. Hence, there is no point of applying max pooling before the third layer.

2.  **Distance of the pooling layer from output**</br>
The max pooling layer should not be used close to the output layer. This is because towards the last few layers, the kernels would have learned important features or parts of objects that will be essential for the classification process. Max pooling here will distort those features by throwing off some information and thereby affect the discriminating power of the network.

## 4. When to stop convolutions and go ahead with a larger kernel or some other alternative
If the receptive field requirement is satisfied even before the output size nears 1x1, then there is no point of convolving further. In such cases, we usually use global average pooling to reduce the output size to 1x1. We could have used a larger kernel to achieve the same, but then it would add huge number of parameters to the network.

## 5. How many layers:
Once we have finalized on the receptive field requirement, convolution size to be used & padding strategy, the number of layers required is a straight forward calculation. We need to have as many layer that are required to reach the receptive field with the given convolution size.

## 6. How many kernels:
Number of kernels should be decided based on the following factors

1.  Complexity of the task:
    Kernels  are feature extractors of CNN. So we need to use as many kernels as the number of features we expect to extract from an image. For simple problems like MNIST, we might not need many features, hence we can go with lesser number of kernels. But for complex problems like image net challenge or similar tasks with naturaal images, we might need much more number of kernels to extract those many features.
2.  Accuracy requirement:
    A network with a larger number of kernels will be easily able to generate features that are required to correctly classify an image. Because of this reason, for the same deep learning task, training accuracy of the model with larger number of kernel will always be high. So use cases where high accuracy is required, we might have to go ahead with larger number of kernels
3.  Parameter constrains:
    The number of parameters and the size of the network will increase exponentially with increase in number of kernels. Hence for use cases where real time inferencing is required, we need to reduce the number of kernels so that frames per second on inferencing will be higher.

## 7. SoftMax Function:
The final layer of the network maps the features to the output label. So we need to convert this output into a probabilty like function which predicts the label for a given input. This is achieved through softmax function.

## 8. Model Optimizer:
Model optimizer decides the variation of gradient descent algorithm to be used in the training process. Depending on the optimizer used, the model can give different results

## 9. Batch Size:
Deciding on the batch size is very critical to achieve good results from our model. Batch size defines the number of training samples the algorithm sees at a time before updating the weights. Larger batch size will result in faster training but it also leads to poor generalization error. On the flip side, smaller batch size will take longer to train but the model will perform better on test data. The reason for better generalization is vaguely attributed to the existence to noise in small batch size training. Because neural network systems are extremely prone to overfitting, the idea is that seeing many small batch size, each batch being a noisy representation of the entire dataset, will cause a sort of “tug-and-pull” dynamic. This “tug-and-pull” dynamic prevents the neural network from overfitting on the training set and hence results in better performance on the test set. The minimum batch size should be always more than the number of distinct output labels. Batch size should always be in the power of 2 to fully utilize the hardware infrastructure.

## 10. Learning Rate:

## 11. Number of Epochs:
**Epoch** is an important hyperparameter in deep learning. One epoch is when an entire training dataset is passed forward and backward through the neural network only once. Training a neural network is an iterative process involving multiple epochs. Weights in network gets updated with each epoch if we use the entire dataset as a batch. Each epoch can contain more than one batch if the training data is too big to fit in memory. The number of epochs are usually decided by looking at the **learning curve** and the identifying the point of divergence of training and test error.
![alt+text](https://github.com/rinazbelhaj/EIP/blob/master/Epoch.png?raw=true "Epochs")
<img src="https://github.com/rinazbelhaj/EIP/blob/master/Epoch.png?raw=true" style="zoom:30%" />

## 12. Validation Checks:
How do we know our network is not going well, comparatively, very early

## 13. 1x1 Convolutions:
1x1 Convolution** is a type of **filter/kernel** used in **deep CNNs**. It is basically a matrix of size **1x1xinput_channel** convolved over the previous input layers. These filters are used for dimensionality reduction of channels by using **feature pooling** technique without affecting image dimension. The number of output channels can be controlled by using that number of 1x1 filters. 1x1 filters reduces the dimension by selecting only important features across the channels. Sometimes, 1x1 filter are used to introduce extra non-linearity without reducing the dimension, eg: google inception module.

![alt+text](https://github.com/rinazbelhaj/EIP/blob/master/1x1%20Convolution.png?raw=true "1x1 Convolution")
## 14. Image Normalization:

## 15. DropOut:

## 16. Batch Normalization:
The distance of Batch Normalization from Prediction,

## 17. LR Schedule: 
and concept behind it
