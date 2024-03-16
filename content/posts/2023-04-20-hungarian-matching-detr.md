---
title: Hungarian matching algorithm in DETR
date: '2023-04-20'
description: Explaining Hungarian matching algorithm used in DETR with a small example
tags:
  - deep-learning
  - paper-notes
---

# Introduction

In the [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872) paper, they directly predict $N$ number of prediction boxes and treat them as set. To find the matching predicted boxes with the target boxes they use *Hungarian matching* algorithm. There is a [great blogpost by Lei Mao](https://leimao.github.io/blog/Hungarian-Matching-Algorithm/) explaining the basic concepts of Hungarian matching algorithm. 

# Short summary of the problem
In the case of DETR, we predict 100 boxes which is more than maximum number of boxes almost any image. Our task is to find closest* predicted box for each target box. Meaning, we will select $n$ best prediction boxes among the $m$ outputs. 

To do this, we form a cost matrix $C$ with the size $m \times n$ , where $m$ is the number of predictions and $n$ is the number of targets where $m > n$ . $C_{i,j}$ would be the matching cost of prediction $i$ and ground truth box $j$ . 

# Matching cost
Matching cost of the element $C_{ij}$ is given by:

$$
\begin{equation}
C_{ij} = \mathcal{L}_{iou}(b_i, \hat{b}_j) + ||b_i - \hat{b}_j||_1 - \hat{p}_j(c_i)
\end{equation}
$$

where $\hat{p}_j(c_i)$ is the probability of the target class. 
After calculating the cost matrix $C$, we can use [linar sum assignment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html#scipy.optimize.linear_sum_assignment) function from the SciPy package. It returns the *row_ids* and *column_ids* which corresponds to the (matched_predictions_ids, target_ids).

# Code
Following is the simplified version of the Hungarian matching algorithm used in the [source code of DETR](https://github.com/facebookresearch/detr/blob/main/models/matcher.py). 
```python
import torch
from scipy.optimize import linear_sum_assignment

target_labels = [1,2] # two target labels
target_bboxes = torch.rand(2,4) # two target boxes

pred_logits = torch.rand(10, 3) # 10 predictions for 3 labels
pred_bboxes = torch.rand(10, 4)  # 4 boxes for 10 predictions

class_cost = -pred_logits[:, target_labels] # 10x2
# We can use torch.cdist which returns the norm distance matrix 10x2
l1_cost = torch.cdist(pred_bboxes, target_bboxes, p=1) # 10x2
# To simplify we omit the IoU calculation. Look at https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/models/matcher.py#L74
iou_cost = torch.randn(10,2) # 10x2

cost_matrix = class_cost + l1_cost + iou_cost
match_preds, match_targets = linear_sum_assignment(cost_matrix)
print(match_preds, match_targets)
#[2 9] [0 1]
```
In this example, predictions $2,9$ matched with $0,1$ target boxes, respecitvely : (2<->0), (9<->1)