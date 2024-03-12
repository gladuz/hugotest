+++
title = "Few-shot image classification using Prototypical Networks"
date= "2023-05-26"
description= "Basic implementation exampleof prototypical networks in Pytorch"
tags = [
    "deep-learning",
    "paper-notes"
]
+++
In few-shot image classification problem, $K$ examples of $N$ types of images are given as an "support" set and the task is to classify the new images by comparing them to those images.
{% include figure.html path="assets/img/protoypical-network-training-example.png" title="example image" class="img-fluid rounded z-depth-1" %}
<div class="caption">https://cs330.stanford.edu/lecture_slides/cs330_transfer_meta_learning.pdf</div>

One of the simplest methods is [Prototypical Networks](https://arxiv.org/abs/1703.05175). During training we feed both "support" and "query" images to the CNN encoder model. Then we take the center of each image class in "support set" and compare the distance of "query" (test) images. We assign the label of the closest center point (prototype) to each "query" image. 
{% include figure.html path="assets/img/prototypical-network-clustering-example.png" title="example image" class="img-fluid rounded z-depth-1" %}
<div class="caption">https://arxiv.org/abs/1703.05175</div>

Nice property of this method is it acts the same in training and test time. It also don't require any additional parameters for meta-training. 

Basic inner training procedure in Pytorch can be implemented follows: [Jupyter notebook in Github](https://github.com/gladuz/Basic-Protonet-Implementation-Pytorch/blob/master/Working.ipynb)
```python
#.... training loop

# We have N types of images with K examples each for training and 1 for testing
#For the image above we have K=5, N=3
logits = model(inputs) #BxK+1xNxD
support_logits = logits[:, :-1, :, :] # BxKxNxD
query_logits = logits[:, -1, :, :] # Bx1xNxD 
query_labels = torch.arange(N).expand(B, -1)

prototypes = support_logits.mean(dim=1) #BxNxD
query_dist = -torch.cdist(query_logits, prototypes) #Negative distances (closer better)
loss = F.cross_entropy(query_dist, query_labels)

#.... Backprop, accuracy etc.
```