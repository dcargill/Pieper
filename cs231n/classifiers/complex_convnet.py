import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class WavePropConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  {conv - [batchnorm] - relu - [dropout] - conv - [batchnorm] - relu - [dropout] - 2x2 max pool}*2
   - affine - relu - affine - l2
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(1,128,128), num_filters=32, filter_size=3, 
               hidden_dim=1000, num_classes=(2*121*121), weight_scale=1e-3, reg=0.0,
               dtype=np.complex64, dropout = .5, seed=None):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_dropout = dropout > 0
    self.params = {}
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.use_batchnorm = True
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
                             
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    F = num_filters
    C,H, W = input_dim
    
    W1 = weight_scale * np.random.randn(F, C, filter_size, filter_size)
    b1 = np.zeros(F)
    W2 = weight_scale * np.random.randn(2*F, F, filter_size, filter_size)
    b2 = np.zeros(2*F)
    
    W3 = weight_scale * np.random.randn(4*F, 2*F, filter_size, filter_size)
    b3 = np.zeros(4*F)
    W4 = weight_scale * np.random.randn(4*F, 4*F, filter_size, filter_size)
    b4 = np.zeros(4*F)
    
    pool_height = 2
    pool_width = 2
    stride_pool = 2
    Hp1 = (H - pool_height) / stride_pool + 1
    Wp1 = (W - pool_width) / stride_pool + 1
    Hp2 = (Hp1 - pool_height) / stride_pool + 1
    Wp2 = (Wp1 - pool_width) / stride_pool + 1
    
    Hh = hidden_dim
    W5 = weight_scale * np.random.randn(4*F * Hp2 * Wp2, Hh)
    b5 = np.zeros(Hh)
    
    Hc = num_classes
    W6 = weight_scale * np.random.randn(Hh, Hc)
    b6 = np.zeros(Hc)
    
    self.params.update({'W1': W1,
                        'W2': W2,
                        'W3': W3,
                        'W4': W4,
                        'W5': W5,
                        'W6': W6,
                        'b1': b1,
                        'b2': b2,
                        'b3': b3,
                        'b4': b4,
                        'b5': b5,
                        'b6': b6})
    
    self.bn_params = []
    self.bn_params = {'bn_param1': {'mode': 'train', 'running_mean': np.zeros((32)), 'running_var': np.zeros((32))},
                     'bn_param2': {'mode': 'train', 'running_mean': np.zeros((64)), 'running_var': np.zeros((64))},
                     'bn_param3': {'mode': 'train', 'running_mean': np.zeros((128)), 'running_var': np.zeros((128))},
                     'bn_param4': {'mode': 'train', 'running_mean': np.zeros((128)), 'running_var': np.zeros((128))},
                     'bn_param5': {'mode': 'train', 'running_mean': np.zeros((Hh)), 'running_var': np.zeros((Hh))}
                      }
    gammas = {'gamma1': np.ones((32)),# 128, 128)),
             'gamma2': np.ones((64)),#, 64, 64)),
             'gamma3': np.ones((128)),#, 64, 64)),
             'gamma4': np.ones((128)),#, 32, 32)),
             'gamma5': np.ones((Hh)),#, 128, 128)),
             }
    betas = {'beta1': np.zeros((32)),#, 128, 128)),
              'beta2': np.zeros((64)),#, 64, 64)),
              'beta3': np.zeros((128)),#, 64, 64)),
              'beta4': np.zeros((128)),#, 32, 32)),
              'beta5': np.zeros((Hh)),#, 128, 128)),
             }
    self.params.update(betas)
    self.params.update(gammas)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
        self.dropout_param['mode'] = mode
    if self.use_batchnorm:
        for key, bn_param in self.bn_params.iteritems():
            bn_param[mode] = mode
    
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    W5, b5 = self.params['W5'], self.params['b5']
    W6, b6 = self.params['W6'], self.params['b6']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    gamma = self.params['gamma1']
    beta = self.params['beta1']
    bn_param = self.bn_params['bn_param1']
    #o1,c1 = conv_relu_forward(X, W1, b1, conv_param)
    o1,c1 = conv_spatial_batchnorm_relu_forward(X, W1, b1, gamma, beta, bn_param, conv_param)        
    
    gamma = self.params['gamma2']
    beta = self.params['beta2']
    bn_param = self.bn_params['bn_param2']
    #o2,c2 = conv_relu_pool_forward(o1, W2, b2, conv_param, pool_param)
    o2,c2 = conv_spatial_batchnorm_relu_pool_forward(o1, W2, b2, gamma, beta, bn_param, conv_param, pool_param)
    
    gamma = self.params['gamma3']
    beta = self.params['beta3']
    bn_param = self.bn_params['bn_param3']
    #o3,c3 = conv_relu_forward(o2, W3, b3, conv_param)
    o3,c3 = conv_spatial_batchnorm_relu_forward(o2, W3, b3, gamma, beta, bn_param, conv_param)
    
    gamma = self.params['gamma4']
    beta = self.params['beta4']
    bn_param = self.bn_params['bn_param4']
    #o4,c4 = conv_relu_pool_forward(o3, W4, b4, conv_param, pool_param)
    o4,c4 = conv_spatial_batchnorm_relu_pool_forward(o3, W4, b4, gamma, beta, bn_param, conv_param, pool_param)
    
    gamma = self.params['gamma5']
    beta = self.params['beta5']
    bn_param = self.bn_params['bn_param5']
    #o5,c5 = affine_relu_forward(o4, W5, b5)
    o5,c5 = affine_batchnorm_relu_forward(o4, W5, b5, gamma, beta, bn_param)
    
    scores, c6 = affine_forward(o5, W6, b6)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    data_loss, dscores = l2_loss(scores, y)
    reg_loss = 0.5 * self.reg * np.sum(W1**2)
    reg_loss += 0.5 * self.reg * np.sum(W2**2)
    reg_loss += 0.5 * self.reg * np.sum(W3**2)
    reg_loss = 0.5 * self.reg * np.sum(W4**2)
    reg_loss += 0.5 * self.reg * np.sum(W5**2)
    reg_loss += 0.5 * self.reg * np.sum(W6**2)
    loss = data_loss + reg_loss
    
    #dx6, dW6, db6 = affine_backward(dscores, c6)
    dx6, dW6, db6 = affine_backward(dscores, c6)
    dW6 += self.reg * W6
    
    #dx5, dW5, db5 = affine_relu_backward(dx6, c5)
    dx5, dW5, db5, dgamma5, dbeta5 = affine_batchnorm_relu_backward(dx6, c5)
    dW5 += self.reg * W5
    
    #dx4, dW4, db4 = conv_relu_pool_backward(dx5, c4)
    dx4, dW4, db4, dgamma4, dbeta4 = conv_spatial_batchnorm_relu_pool_backward(dx5, c4)
    dW4 += self.reg * W4
    
    #dx3, dW3, db3 = conv_relu_backward(dx4, c3)
    dx3, dW3, db3, dgamma3, dbeta3 = conv_spatial_batchnorm_relu_backward(dx4, c3)
    dW3 += self.reg * W3
    
    #dx2, dW2, db2 = conv_relu_pool_backward(dx3, c2)
    dx2, dW2, db2, dgamma2, dbeta2 = conv_spatial_batchnorm_relu_pool_backward(dx3, c2)
    dW2 += self.reg * W2
    
    #dx1, dW1, db1 = conv_relu_backward(dx2, c1)
    dx1, dW1, db1, dgamma1, dbeta1 = conv_spatial_batchnorm_relu_backward(dx2, c1)
    dW1 += self.reg * W1
    
    grads.update({'W1': dW1,'b1': db1,'gamma1':dgamma1,'beta1':dbeta1,
                  'W2': dW2,'b2': db2,'gamma2':dgamma2,'beta2':dbeta2,
                  'W3': dW3,'b3': db3,'gamma3':dgamma3,'beta3':dbeta3,
                  'W4': dW4,'b4': db4,'gamma4':dgamma4,'beta4':dbeta4,
                  'W5': dW5,'b5': db5,'gamma5':dgamma5,'beta5':dbeta5,
                  'W6': dW6,'b6': db6})
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return loss, grads

def affine_batchnorm_relu_forward(x, w, b, gamma, beta, bn_param):
    a, fc_cache = affine_forward(x, w, b)
    batch_out, batch_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(batch_out)
    cache = (fc_cache, batch_cache, relu_cache)
    
    return out, cache

def affine_batchnorm_relu_backward(dout,cache):
    fc_cache, batch_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dgamma, dbeta = batchnorm_backward_alt(da, batch_cache)
    dx, dw, db = affine_backward(dx, fc_cache)
    
    return dx, dw, db, dgamma, dbeta

def conv_spatial_batchnorm_relu_forward(x, w, b, gamma, beta, bn_param, conv_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    batch_out, batch_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(batch_out)
    cache = (conv_cache, batch_cache, relu_cache)
    
    return out, cache

def conv_spatial_batchnorm_relu_backward(dout,cache):
    conv_cache, batch_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dgamma, dbeta = spatial_batchnorm_backward(da, batch_cache)
    dx, dw, db = conv_backward_fast(dx, conv_cache)
    
    return dx, dw, db, dgamma, dbeta

def conv_spatial_batchnorm_relu_pool_forward(x, w, b, gamma, beta, bn_param, conv_param, pool_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    batch_out, batch_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    relu_out, relu_cache = relu_forward(batch_out)
    out, pool_cache = max_pool_forward_fast(relu_out, pool_param)
    cache = (conv_cache, batch_cache, relu_cache, pool_cache)
    
    return out, cache
        
def conv_spatial_batchnorm_relu_pool_backward(dout,cache):
    conv_cache, batch_cache, relu_cache, pool_cache = cache
    dpool_out = max_pool_backward_fast(dout, pool_cache)
    drelu_out = relu_backward(dpool_out, relu_cache)
    dbatch_out, dgamma, dbeta = spatial_batchnorm_backward(drelu_out, batch_cache)
    dx, dw, db = conv_backward_fast(dbatch_out, conv_cache)
    
    return dx, dw, db, dgamma, dbeta
  