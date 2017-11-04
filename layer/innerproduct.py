# -*- coding: utf-8 -*-

import numpy as np

class Innerproduct:
    def __init__(self, unit_in, unit_out):
        self.unit_in = unit_in
        self.unit_out = unit_out
        self.weight = np.random.rand(unit_out, unit_in)
        self.bias = 0.0
        return
        
    def forward(self, x):
        u = x.dot(self.weight.T) + self.bias
        return u
        
    def backward(self, delta):
        return
        
    def update(self, lr):
        return
