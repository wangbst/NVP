# NVP: Neural Vitality based Dynamic Sparsity Pruning in Convolutional Neural Networks
This is the official implementation for NVP: Neural Vitality based Dynamic Sparsity Pruning in Convolutional Neural Networks.

# Overview
As AI models grow in complexity, mitigating computational inefficiency while preserving performance is crucial for sustainable deep learning. We propose a novel framework for analyzing and optimizing neural networks, particularly convolutional neural networks (CNNs), that centers on the phenomenon of dying neurons—units exhibiting low activation variability and reduced expressiveness. To quantify this behavior, we introduce the concept of neuron entropy, a metric that captures the variability in neuron activations in response to diverse inputs, thereby reflecting the network's overall expressiveness. Building upon this insight, we define the Neural Vitality Index (NVI), which synergistically integrates neuron activation entropy and weight norm to assess the vitality of individual neurons. Motivated by the need to mitigate the adverse effects of dying neurons, we further propose Neural Vitality Pruning (NVP), a dynamic pruning method that adaptively sparsifies the network during training. Extensive experiments on CNN architectures across multiple SOTA methods demonstrate that NVP not only enhances model sparsity but also improves performance, providing a principled and effective approach to efficient neural network optimization.

# Results

![Layer-wise pruning rate distribution](https://github.com/wangbst/NVP/blob/main/Figure/Layer-wise%20pruning%20rate%20distribution.png) 

# Dependencies
```shell
conda create -n myenv python=3.7
conda activate myenv
conda install -c pytorch pytorch==1.9.0 torchvision==0.10.0
pip install scipy
```

# Datasets
Please download the Imagenet Dataset. 

# ResNet18 and Leaky ReLU
All used ResNet18 and Leaky ReLU models can be downloaded from here. Please put them in ResNet18().

# Run dying neurons accumulation for a ResNet-18 trained on CIFAR-10.
 ```shell
$ python Resnet18.py
$ python Leaky ReLU.py
```
- In Leaky ReLU.py, replace activation functions ReLU with LeakyReLU.

# Run Neural sparsity for ResNet-18 on CIFAR-10.
 ```shell
$ python Resnet18.py
$ python Leaky ReLU.py
```

 # Run Weight sparsity for ResNet-18 on CIFAR-10.
 ```shell
$ python Resnet18.py
$ python Leaky ReLU.py
```
