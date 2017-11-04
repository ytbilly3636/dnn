# -*- coding: utf-8 -*-

import numpy as np
import sys

class Convolution:
    def __init__(self, ch_in, ch_out, kernel, stride=1, pad=None):
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.kernel = kernel
        self.stride = stride
        self.pad = pad
        self.filter = np.random.rand(ch_out, ch_in, kernel, kernel)
        self.bias = 0.0
        return
        
    def forward(self, x):
        # x should be a 4ch numpy array (batch, ch, height, width)
        if not ((len(x.shape) == 4) and (x[0] > 0) and (x[1] == self.ch_in) and (x[2] > 0) and (x[3] > 0)):
            print 'x should be a 4ch numpy array (batch, ch, height, width)'
            sys.exit()
        
        # zero padding
        if not pad == None:
            padded_x = np.zeros((x.shape[0], x.shape[1], x.shape[2] + 2*self.pad, x.shape[3] + 2*self.pad), x.dtype)
            padded_x[:, :, self.pad:self.pad+x.shape[2], self.pad:self.pad+x.shape[3]] = x
            x = padded_x
        
        # define size of outputs
        out_h = 1 + (x.shape[2] - self.kernel) / self.stride
        out_w = 1 + (x.shape[3] - self.kernel) / self.stride
        
        # generate inputs
        self.x_buf = []
        for h in xrange(out_h):
            for w in xrange(out_w):
                self.x_buf.append(x[:, :, h*self.stride:h*self.stride+self.kernel, w*self.stride:w*self.stride+self.kernel])
        self.x_buf = np.asarray(self.x_buf)
        self.x_buf = np.rollaxis(self.x_buf, 0, 2)
        
        # convolution
        y = np.tensordot(self.x_buf, self.filter, ((2, 3, 4), (1, 2, 3))) + self.bias
        y = np.rollaxis(y, 1, 3)
        y = y.reshape(y.shape[0], y.shape[1], out_h, out_w)
        return y
        
    def backward(self, delta):
        return
        
    def update(self, lr):
        return
