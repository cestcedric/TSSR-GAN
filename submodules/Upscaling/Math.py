#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 2 - 12

import numpy
from   typing import Generator

def to_list(x, repeat=1):
  """convert x to list object

    Args:
       x: any object to convert
       repeat: if x is to make as [x], repeat `repeat` elements in the list
  """
  if isinstance(x, (Generator, tuple, set)):
    return list(x)
  elif isinstance(x, list):
    return x
  elif isinstance(x, dict):
    return list(x.values())
  elif x is not None:
    return [x] * repeat
  else:
    return []


def gaussian_kernel(kernel_size: (int, tuple, list), width: float):
  """generate a gaussian kernel

  Args:
      kernel_size: the size of generated gaussian kernel. If is a scalar, the
                   kernel is a square matrix, or it's a kernel of HxW.
      width: the standard deviation of gaussian kernel. If width is 0, the
             kernel is identity, if width reaches to +inf, the kernel becomes
             averaging kernel.
  """

  kernel_size = numpy.asarray(to_list(kernel_size, 2), numpy.float)
  half_ksize = (kernel_size - 1) / 2.0
  x, y = numpy.mgrid[-half_ksize[0]:half_ksize[0] + 1,
         -half_ksize[1]:half_ksize[1] + 1]
  kernel = numpy.exp(-(x ** 2 + y ** 2) / (2 * width ** 2))
  return kernel / (kernel.sum() + 1e-8)


def anisotropic_gaussian_kernel(kernel_size: (int, tuple, list), theta: float,
                                l1: float, l2: float):
  """generate anisotropic gaussian kernel

  Args:
      kernel_size: the size of generated gaussian kernel. If is a scalar, the
                   kernel is a square matrix, or it's a kernel of HxW.
      theta: rotation angle (rad) of the kernel. [0, pi]
      l1: scaling of eigen values on base 0. [0.1, 10]
      l2: scaling of eigen values on base 1. [0.1, L1]
  """

  def gmdistribution(mu, sigma):
    half_k = (kernel_size - 1) / 2.0
    x, y = numpy.mgrid[-half_k[0]:half_k[0] + 1, -half_k[1]:half_k[1] + 1]
    X = numpy.expand_dims(numpy.stack([y, x], axis=-1), axis=-2)
    L = numpy.linalg.cholesky(sigma).transpose()
    diag_l = numpy.diag(L)
    log_det_sigma = 2 * numpy.log(diag_l).sum()
    log_1h = numpy.sum(numpy.matmul((X - mu), numpy.linalg.inv(L)) ** 2, axis=(-1, -2))
    log_1h = -0.5 * (log_1h + log_det_sigma)
    log_1h -= numpy.log(2 * numpy.pi)
    y = numpy.exp(log_1h)
    return y

  kernel_size = numpy.array(to_list(kernel_size, 2), numpy.int)
  theta = numpy.clip(theta, 0, numpy.pi)
  l1 = numpy.clip(l1, 0.1, 10)
  l2 = numpy.clip(l2, 0.1, l1)
  mat_v = numpy.array([
    [numpy.cos(theta), numpy.sin(theta)],
    [numpy.sin(theta), -numpy.cos(theta)]
  ], numpy.float)
  mat_d = numpy.array([[l1, 0], [0, l2]], numpy.float)
  sigma = numpy.matmul(numpy.matmul(mat_v, mat_d), numpy.linalg.inv(mat_v))
  kernel = gmdistribution(0, sigma)
  return kernel / (kernel.sum() + 1e-8)


def list_rshift(l, s):
  for _ in range(s):
    l.insert(0, l.pop(-1))
  return l


def bicubic_filter(x, a=-0.5):
  # https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
  if x < 0:
    x = -x
  if x < 1:
    return ((a + 2.0) * x - (a + 3.0)) * x * x + 1
  if x < 2:
    return (((x - 5) * x + 8) * x - 4) * a
  return 0


def weights_downsample(scale_factor):
  if scale_factor < 1:
    ss = int(1 / scale_factor + 0.5)
  else:
    ss = int(scale_factor + 0.5)
  support = 2 * ss
  ksize = support * 2 + 1
  weights = []
  for lambd in range(ksize):
    dist = -2 + (2 * lambd + 1) / support
    weights.append(bicubic_filter(dist))
  h = numpy.array([weights])
  h /= h.sum()
  v = h.transpose()
  kernel = numpy.matmul(v, h)
  assert kernel.shape == (ksize, ksize), f"{kernel.shape} != [{ksize}]"
  return kernel, ss


def weights_upsample(scale_factor):
  if scale_factor < 1:
    ss = int(1 / scale_factor + 0.5)
  else:
    ss = int(scale_factor + 0.5)
  support = 2
  ksize = support * 2 + 1
  weights = [[] for _ in range(ss)]
  for i in range(ss):
    for lambd in range(ksize):
      dist = int((1 + ss + 2 * i) / 2 / ss) + lambd - 1.5 - (2 * i + 1) / 2 / ss
      weights[i].append(bicubic_filter(dist))
  w = [numpy.array([i]) / numpy.sum(i) for i in weights]
  w = list_rshift(w, ss - ss // 2)
  kernels = []
  for i in range(len(w)):
    for j in range(len(w)):
      kernels.append(numpy.matmul(w[i].transpose(), w[j]))
  return kernels, ss


def nd_meshgrid(*size, permute=None):
  _error_msg = ("Permute index must match mesh dimensions, "
                "should have {} indexes but got {}")
  size = numpy.array(size)
  ranges = []
  for x in size:
    ranges.append(numpy.linspace(-1, 1, x))
  mesh = numpy.stack(numpy.meshgrid(*ranges, indexing='ij'))
  if permute is not None:
    if len(permute) != len(size):
      raise ValueError(_error_msg.format(len(size), len(permute)))
    mesh = mesh[permute]
  return mesh.transpose(*range(1, mesh.ndim), 0)


def camera_response_function(inputs, crf_table, max_val=1):
  """Estimated CRF, transform irradiance L to RGB image. If `crf_table` is
    inverted, transform RGB image to irradiance L.

  Args:
      inputs: A 3-D or 4-D tensor, representing irradiance.
      crf_table: CRF lookup table shape (1024,).
      max_val: specify the range of inputs: in (0, max_val)
  Return:
      RGB images (or L) with the same shape as inputs, in range [0, max_val].
  """

  inputs_norm = numpy.clip(inputs / max_val, 0, 1)
  quant = crf_table.shape[0] - 1
  inputs_index = (inputs_norm * quant).astype('int32')
  ret = []
  for i in inputs_index.flatten():
    ret.append(crf_table[i])
  return numpy.reshape(ret, inputs.shape)


def gen_pca_mat(dim=15, kernel_size=15, samples=10000):
  kernels = []
  for i in range(samples):
    theta = numpy.random.uniform(0, numpy.pi)
    l1 = numpy.random.uniform(0.1, 10)
    l2 = numpy.random.uniform(0.1, l1)
    kernels.append(anisotropic_gaussian_kernel(kernel_size, theta, l1, l2))
  kernels = numpy.stack(kernels).reshape([samples, -1]).transpose()
  mat_c = numpy.matmul(kernels, kernels.transpose())
  _, mat_v = numpy.linalg.eigh(mat_c, 'U')
  return mat_v[..., -dim:].transpose()
