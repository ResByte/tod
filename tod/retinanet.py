import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from typing import List,Dict

__all__ = ["RetinaNet"]

class RetinaNet(nn.Module):
    """RetinaNet model for object detection 
    As described in : https://arxiv.org/abs/1708.02002
    """
    def __init__(self):
        super(RetinaNet,self).__init__()

        # target classes 
        
        # input_features 

        # backbone 

        # head 

        # anchor_generator 




    

    def forward(self, x:torch.Tensor):
        """
        Args:
            x: list of preprocessed input images as tensor, 
            shape:(B,C,H,W)
        Returns:
            if training: 
                losses for the input batch
            if inference:
                List of elements each containing, bbox and category 
        """
        # TODO: 
        # 1. if the input is not pre-processed, process it first 

        # 2. pass through backbone to get list of features 
        #    in case of FPN, extract features at multiple levels 

        # 3. via heads, get box classes and box deltas 

        # 4. get anchors from anchor generator
 
        raise NotImplementedError


    def losses(self):
        """Takes in groundtruth values for classes and bbox,
        as well as predicted categories and bbox 
        and returns both classess and bbox losses
        """
        # compute all kinds of losses 

        # 1. Logits losses for classification 

        # 2. regression loss for bbox 

        return classification_loss, bbox_reg_loss 

    def inference(self):
        """Takes in input tensor or batch of tensor
        and produces bbox, classes for each box.
        """
        raise NotImplementedError


