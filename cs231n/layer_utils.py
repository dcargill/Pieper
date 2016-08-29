from cs231n.layers import *
from cs231n.fast_layers import *


def affine_relu_forward(x, w, b):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache


def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db

def affine_tanh_forward(x, w, b):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, tanh_cache = tanh_forward(a)
  cache = (fc_cache, tanh_cache)
  return out, cache


def affine_tanh_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, tanh_cache = cache
  da = tanh_backward(dout, tanh_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db

def affine_threshold_forward(x,w, b,t):
  N = x.shape[0]
  D = np.prod(x.shape[1:])
  new_x = np.reshape(x, (N, D))
  xw = np.dot(new_x, w)
  xw = xw - t
  #out = np.maximum(0,xw)
  out = np.maximum(0,xw+b)
  mask = np.ones_like(out)
  mask[out <= 0] = 0
  #out += mask*b
  cache = (x,w,b,t, mask)
  return out, cache

def affine_threshold_backward(dout, cache):
  N,M = dout.shape
  x,w,b,t, mask = cache
    
  
  dx = dout.dot(w.T) 
  dx = np.reshape(dx,x.shape)
  
  yes_pass = np.ones_like(mask)
  yes_pass[mask<=0] = 0
  no_pass = np.ones_like(mask)
  no_pass[mask>=0] = 0
  dout_pos = np.ones_like(dout)
  dout_pos[dout<=0] = 0
  dout_neg = np.ones_like(dout)
  dout_neg[dout>=0] = 0
  dtemp1 = dout_pos*yes_pass*dout
  detmp2 = dout_neg*no_pass*dout
  dtemp = -(dtemp1 + detmp2)
  #dt = np.sum(dtemp, axis=0, keepdims=True).reshape(dtemp.shape[1],)# + .5*t
  
  x = np.reshape(x,(x.shape[0],w.shape[0]))
  dw = x.T.dot(dout)
  db = np.sum(dout, axis=0, keepdims=True).reshape(dw.shape[1],)
  dt = np.sign(np.sum(dtemp*dout, axis=0, keepdims=True).reshape(dw.shape[1],)/dw.shape[0])
  #dt = np.zeros_like(dt)
  return dx, dw, db, dt

def conv_relu_forward(x, w, b, conv_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache


def conv_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache)
  return out, cache


def conv_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db

