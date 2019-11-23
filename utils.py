import torch
import numpy as np
import scipy
import pdb

def sampling_bernoulli(probs):
    #pdb.set_trace()
    return probs - torch.rand(probs.size()).cuda()


def sampling_gaussian(probs):
    return probs+ torch.rand(probs.size()).cuda()
