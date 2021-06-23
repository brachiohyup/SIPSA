from __future__ import division  # needed
import tensorflow as tf
import numpy as np
import math


def gaussblur2b(x, fsig):
    """Apply gaussian blur to input

    Args:
        x: 4-D input
        fsig: STD for Gaussian filter
    """
    # fsig = (szf+1)/6  # n = 6*sig-1

    szf = 6*fsig-1
    szf = math.floor(szf/2)*2+1
    szf = int(szf)
    szf = max(szf, 3)

    y = tf.identity(x)
    szy = y.shape.as_list()

    bf = _tf_fspecial_gauss(szf, fsig)
    bf = tf.tile(bf, [1, 1, szy[3], 1])

    pp = int((szf-1)/2)
    y = tf.pad(y, [[0, 0], [pp, pp], [pp, pp], [0, 0]], "REFLECT")
    y = tf.nn.depthwise_conv2d(y, bf, strides=[1, 1, 1, 1], padding='VALID')

    return y


def satdwsize(x, sc):
    """ Downscale an image with a scale factor of sc: degradation model for satellite. Args: x: input, sc: scale ratio"""

    sz = x.shape.as_list()
    y = x + 0
    y = tf.split(y, sz[3], axis=3)
    y = tf.concat(y, axis=0)
    y = gaussblur2b(y, 1.0/sc)
    y = tf.space_to_depth(y, sc)
    y = tf.reduce_mean(y, axis=3, keepdims=True)
    y = tf.split(y, sz[3], axis=0)
    y = tf.concat(y, axis=3)
    return y


def QNR(ps, pan, ms, sc=4, isloss=False, getAll=False):
    """QNR metric. Args: ps: 4D network output, pan: 4D PAN input, ms: 4D low-resolution MS input, sc: scale ratio, isloss: can be used as a loss, getAll: outputs 3 metric scores"""
    DL = qnrL(ps, ms)
    DS = qnrS(ps, pan, ms)
    if isloss:
        q = DL + DS
    else:
        q = tf.abs(1.-DL)*tf.abs(1.-DS)

    if getAll:
        q = tf.stack([q, tf.abs(1.-DL), tf.abs(1.-DS)])
    return q


# QNR subfunction metric - D_gamma
def qnrL(y, x):  # y=PS, x=LRMS
    """Args: y: 4D network output, x: 4D low-res MS input"""
    sz = y.shape.as_list()
    qy = qindex(tf.tile(y, [1, 1, 1, sz[3]]), my_tfrepeat(y, sz[3], 3))
    qx = qindex(tf.tile(x, [1, 1, 1, sz[3]]), my_tfrepeat(x, sz[3], 3))
    q = tf.abs(qy-qx)
    q = tf.reduce_sum(q, axis=3, keepdims=True)/(sz[3]*(sz[3]-1))
    q = tf.reduce_mean(q)
    return q


# QNR subfunction metric - D_s
def qnrS(ps, pan, ms, sc=4):  # ms=LRMS
    """Args: ps: 4D network output, pan: 4D PAN input, ms: 4D low-res MS input, sc: scale ratio"""
    sz = ps.shape.as_list()
    pw = satdwsize(pan, sc)
    qy = qindex(ps, tf.tile(pan, [1, 1, 1, sz[3]]))
    qx = qindex(ms, tf.tile(pw, [1, 1, 1, sz[3]]))
    q = tf.abs(qy-qx)
    q = tf.reduce_mean(q)
    return q


# Q index metric
def qindex(y, x, N=32, eps=1e-16, keepdims0=True):
    """Args: y: 4D network outputs, 4D x: target images"""
    szy = y.shape.as_list()
    y = tf.split(y, szy[3], axis=3)
    y = tf.concat(y, axis=0)

    x = tf.split(x, szy[3], axis=3)
    x = tf.concat(x, axis=0)

    y = pad_sr(y, N)
    x = pad_sr(x, N)

    y = tf.space_to_depth(y, N)
    x = tf.space_to_depth(x, N)

    ym = tf.reduce_mean(y, axis=3, keepdims=True)
    xm = tf.reduce_mean(x, axis=3, keepdims=True)

    y = y - ym
    x = x - xm

    syx = tf.reduce_mean(x*y, axis=3, keepdims=True)
    sxx = tf.reduce_mean(x*x, axis=3, keepdims=True)
    syy = tf.reduce_mean(y*y, axis=3, keepdims=True)

    q = 4*syx*xm*ym / ((sxx+syy)*(xm*xm+ym*ym) + eps)

    q = tf.split(q, szy[3], axis=0)
    q = tf.concat(q, axis=3)

    q = tf.reduce_mean(q, [1, 2], keepdims=True)

    if not keepdims0:
        q = tf.reduce_mean(q)

    return q


def pad_sr(x0, N):  # e.g. N = 2**x
    """ Pad an image to have size that is integer multiple of N """
    """Args: x0: 4D image, N: integer multiple e.g. 2**x"""
    tsz1 = tf.shape(x0)
    tsz0 = tsz1[1:3]
    tsz = tf.cast(tsz0, tf.float32)
    tsz = tsz/(N)
    tsz = tf.ceil(tsz)
    tsz = tsz*(N)
    tsz = tf.cast(tsz, tf.int32)
    pp1 = tsz-tsz0
    pp1 = tf.cast(pp1, tf.float32)
    pp1 = pp1/2
    pp1 = tf.ceil(pp1)
    pp1 = tf.cast(pp1, tf.int32)
    pp2 = tsz-tsz0-pp1

    x0 = tf.pad(x0, [[0, 0], [pp1[0], pp2[0]], [pp1[1], pp2[1]], [0, 0]], "REFLECT")

    return x0


def my_resize_nearest(x, sc):
    """ Nearest-neighbor interpolation """
    y = x + 0
    y = tf.tile(y, [1, 1, 1, int(sc**2)])
    y = tf.depth_to_space(y, sc)
    return y


def lrelu(x, leak=0.1):
    """ Leaky ReLU """
    return tf.maximum(x, leak * x)


def guidedfilter_sharp(I, p, r, eps):  # I: 구조정보, p: 색정보
    """ Apply guided filtering """
    I2 = tf.concat([I, p, I * p, I * I], axis=3)
    I2 = boxfilter(I2, r)
    meanI, meanP, meanIp, meanII = tf.split(I2, 4, axis=3)

    covIp = meanIp - meanI * meanP
    varI = meanII - meanI * meanI
    a = covIp / (tf.abs(varI) + eps)
    b = meanP - a * meanI
    q = a * I + b
    return q


def boxfilter(x, szf):  # szf = 1,3,5...
    """ Blurs an image using the box filter """
    y = tf.identity(x)
    szy = y.shape.as_list()

    bf = tf.ones([szf, szf, 1, 1], tf.float32) / (szf**2)
    bf = tf.tile(bf, [1, 1, szy[3], 1])

    pp = int((szf-1)/2)
    pp2 = int(szf-1-pp)
    y = tf.pad(y, [[0, 0], [pp, pp2], [pp, pp2], [0, 0]], "REFLECT")
    y = tf.nn.depthwise_conv2d(y, bf, strides=[1, 1, 1, 1], padding='VALID')
    return y


def l_SP2conv2(x, chout, sc, name, opts):
    """ Convolution and upscale by depth to space """
    y = x + 0
    sz = y.shape.as_list()
    conv_w2, conv_b2 = conv_same_sc(name, [opts['fs'], opts['fs'], sz[3], chout], sc, opts['ty'], opts['ini'],
                                    opts['bias'])
    y = tf.nn.conv2d(y, conv_w2, strides=[1, 1, 1, 1], padding='SAME') + conv_b2
    y = tf.depth_to_space(y, sc)
    return y


def conv_same_sc(name, sz, sc, ty='npan', ini=1, bias=True):
    """ Convolution at same scale """
    if ty == 'npan':
        n = (np.prod(sz[:3]) + np.prod(sz[:2]) * sz[3] * (sc ** 2)) / 2  # fan_avg
        stddev = np.sqrt(ini / n)
        w_ini_np = np.random.normal(scale=stddev, size=sz)
        w_ini_np = np.clip(w_ini_np, -2.0 * stddev, 2.0 * stddev)
    elif ty == 'npau':
        n = (np.prod(sz[:3]) + np.prod(sz[:2]) * sz[3] * (sc ** 2)) / 2  # fan_avg
        limit = np.sqrt(3 * ini / n)
        w_ini_np = np.random.uniform(-limit, limit, size=sz)
    else:
        raise NameError('undefined ini type')

    w_ini_np = np.tile(w_ini_np, (1, 1, 1, int(sc ** 2)))
    sz2 = np.shape(w_ini_np)
    w = tf.get_variable("w_" + name, sz2, initializer=tf.constant_initializer(w_ini_np))

    if bias:
        b = tf.get_variable("b_" + name, sz2[3], initializer=tf.constant_initializer(0.0))
        out = [w, b]
    else:
        out = [w, 0]

    return out


def l_conv(x, chout, name, opts):
    """ Convolution layer with initialization """
    y = x + 0
    sz = y.shape.as_list()

    conv_w2, conv_b2 = ini_conv_old(name, [opts['fs'], opts['fs'], sz[3], chout], opts['ty'], opts['ini'], opts['bias'])
    y = tf.nn.conv2d(y, conv_w2, strides=[1, 1, 1, 1], padding='SAME') + conv_b2

    return y

def encode_l_conv(x, chout, name, opts):
    """ Convolution layer with initialization """
    y = x + 0
    sz = y.shape.as_list()

    conv_w2, conv_b2 = ini_conv_old(name, [opts['fs'], opts['fs'], sz[3], chout], opts['ty'], opts['ini'], opts['bias'])
    y = tf.nn.conv2d(y, conv_w2, strides=[1, 2, 2, 1], padding='SAME') + conv_b2

    return y



def pixel_wise_convolution(x, chout, name, opts):
    """ Convolution layer with initialization """
    y = x + 0
    sz = y.shape.as_list()

    conv_w2, conv_b2 = ini_conv_old(name, [1, 1, sz[3], chout], opts['ty'], opts['ini'], opts['bias'])
    y = tf.nn.conv2d(y, conv_w2, strides=[1, 1, 1, 1], padding='SAME') + conv_b2

    return y

def D_offset(x, chout, name, opts):
    """ Convolution layer with initialization """
    y = x + 0
    sz = y.shape.as_list()

    conv_w2, conv_b2 = ini_conv_old(name, [opts['fs'], opts['fs'], sz[3], 9], opts['ty'], opts['ini'], opts['bias'])
    y = tf.nn.conv2d(y, conv_w2, strides=[1, 1, 1, 1], padding='SAME') + conv_b2

    return x, y

def D_conv(x, chout, name, opts):
    """ Convolution layer with initialization """
    y = x + 0
    sz = y.shape.as_list()

    conv_w2, conv_b2 = ini_conv_old(name, [opts['fs'], opts['fs'], sz[3], chout], opts['ty'], opts['ini'], opts['bias'])
    y = tf.nn.conv2d(y, conv_w2, strides=[1, 1, 1, 1], padding='SAME') + conv_b2

    return x, y

def ini_conv_old(name, sz, ty='xav', ini=1, bias=True):
    """ Initialization of weights and biases """
    if ty == 'uni':
        stddev = np.sqrt(ini / (np.prod(sz[:3])))
        w = tf.get_variable("w_" + name, sz,
                            initializer=tf.random_uniform_initializer(minval=-stddev, maxval=stddev))
    elif ty == 'uni2':
        stddev = np.sqrt(ini / (np.prod(sz[:3]) + np.prod(sz[0:2]) * sz[3]))
        w = tf.get_variable("w_" + name, sz,
                            initializer=tf.random_uniform_initializer(minval=-stddev, maxval=stddev))
    elif ty == 'xav':
        stddev = np.sqrt(ini / (np.prod(sz[:3])))
        w = tf.get_variable("w_" + name, sz,
                            initializer=tf.random_normal_initializer(stddev=stddev))
    elif ty == 'xavT':
        stddev = np.sqrt(ini / (np.prod(sz[0:2]) * sz[3]))
        w = tf.get_variable("w_" + name, sz,
                            initializer=tf.random_normal_initializer(stddev=stddev))
    elif ty == 'xav2t':
        stddev = np.sqrt(ini / (np.prod(sz[:3]) + np.prod(sz[0:2]) * sz[3]))
        w = tf.get_variable("w_" + name, sz,
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
    elif ty == 'npan':
        n = (np.prod(sz[:3]) + np.prod(sz[:2]) * sz[3]) / 2  # fan_avg
        stddev = np.sqrt(ini / n)
        w_ini_np = np.random.normal(scale=stddev, size=sz)
        w_ini_np = np.clip(w_ini_np, -2.0 * stddev, 2.0 * stddev)
        w = tf.get_variable("w_" + name, sz,
                            initializer=tf.constant_initializer(w_ini_np))
    elif ty == 'npau':
        n = (np.prod(sz[:3]) + np.prod(sz[:2]) * sz[3]) / 2  # fan_avg
        limit = np.sqrt(3 * ini / n)
        w_ini_np = np.random.uniform(-limit, limit, size=sz)
        w = tf.get_variable("w_" + name, sz,
                            initializer=tf.constant_initializer(w_ini_np))

    else:
        raise NameError('undefined ini type')

    if bias:
        b = tf.get_variable("b_" + name, sz[3], initializer=tf.constant_initializer(0.0))
        out = [w, b]
    else:
        out = [w, 0]

    return out


def my_tfrepeat(x, nr, di):
    """ Repeat input 'nr' times through dimension 'di'. """
    ylist = tf.unstack(x, axis=di)
    y = tf.stack(ylist, axis=0)
    y = tf.expand_dims(y, int(di + 1))
    sz1 = y.shape.as_list()
    n1 = np.ones((len(sz1),), dtype=np.int)
    n1[int(di + 1)] = nr
    y = tf.tile(y, n1)
    ylist = tf.unstack(y, axis=0)
    y = tf.concat(ylist, axis=di)
    return y

def sobelfilter2XY(x, split=False):
    """ Apply sobel filter to input """
    y = tf.identity(x)
    szy = y.shape.as_list()

    bf = tf.constant([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=tf.float32, shape=[3, 3, 1, 1])
    bf2 = tf.transpose(bf, perm=[1, 0, 2, 3])
    bf = tf.concat([bf, bf2], axis=2)
    bf = tf.tile(bf, [1, 1, szy[3], 1])

    y = my_tfrepeat(y, 2, 3)

    pp = int(1)
    y = tf.pad(y, [[0, 0], [pp, pp], [pp, pp], [0, 0]], "REFLECT")
    y = tf.nn.depthwise_conv2d(y, bf, strides=[1, 1, 1, 1], padding='VALID')
    if split:
        y = tf.stack([y[..., ::2], y[..., 1::2]], axis=4)
    return y


def _tf_fspecial_gauss(size, sigma):
    """ Function to mimic the 'fspecial' gaussian MATLAB function """
    x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / tf.reduce_sum(g)


def gaussblur2(x, szf, rr=1):
    """ Apply gaussian blur to input, szf: filter size. rr: dilation rate """
    fsig = (szf + 1) / 6  # n = 6*sig-1
    y = tf.identity(x)
    szy = y.shape.as_list()

    bf = _tf_fspecial_gauss(szf, fsig)
    bf = tf.tile(bf, [1, 1, szy[3], 1])

    pp = int((szf + (szf - 1) * (rr - 1) - 1) / 2)
    y = tf.pad(y, [[0, 0], [pp, pp], [pp, pp], [0, 0]], "REFLECT")
    y = tf.nn.depthwise_conv2d(y, bf, strides=[1, 1, 1, 1], padding='VALID', rate=[rr, rr])

    return y


def satNorm_all(x, dname, denorm=False):
    """ Normalizes an input using an average statistic of certain satellite. Args: denorm: if true, denormalize. """
    y = x + 0

    if dname == 'wv3ms':
        m = tf.constant([0.158, 0.221, 0.170], shape=[3], dtype=tf.float32)
        s = tf.constant([0.111, 0.129, 0.0854], shape=[3], dtype=tf.float32)
    elif dname == 'wv3pan':
        m = tf.constant([0.222], shape=[1], dtype=tf.float32)
        s = tf.constant([0.131], shape=[1], dtype=tf.float32)
    elif dname == 'kak3ams4':
        m = tf.constant([0.19, 0.22, 0.20, 0.31], dtype=tf.float32)
        s = tf.constant([0.12, 0.12, 0.090, 0.17], dtype=tf.float32)
    elif dname == 'kak3apan':
        m = tf.constant([0.11], dtype=tf.float32)
        s = tf.constant([0.059], dtype=tf.float32)
    else:
        raise NameError('undefined dname')

    if not denorm:
        y = (y - m) / s
    else:
        y = y * s + m

    return y


def tf_aug(xl):
    """ Augmentation method for training """
    i_rot = tf.random.uniform([], 0, 4)
    i_rot = tf.floor(i_rot)
    i_rot = tf.cast(i_rot, tf.int32)
    i_flip = tf.random.uniform([], 0, 2)
    i_flip = tf.floor(i_flip)
    i_flip = tf.cast(i_flip, tf.int32)
    yl = []
    for x in xl:
        y = x + 0
        y = tf.cond(i_flip > 0, lambda: tf.image.flip_left_right(y), lambda: y)
        y = tf.cond(i_rot > 0, lambda: tf.image.rot90(y, i_rot), lambda: y)
        yl.append(y)
    if len(yl) == 1:
        yl = yl[0]
    return yl


def tf_rgb2ycbcr(rgb):
    # rgb = tf.clip_by_value(rgb, 0, 1)
    M = tf.constant([65.481, -37.797, 112, 128.553, -74.203, -93.786, 24.966, 112, -18.214], shape=[3, 3],
                    dtype=tf.float32)
    M = M / 255
    b = tf.constant([16, 128, 128], shape=[1, 1, 1, 3], dtype=tf.float32)
    b = b / 255
    ycbcr = tf.tensordot(rgb, M, axes=1)
    ycbcr = ycbcr + b

    return ycbcr


def tf_ycbcr2rgb(ycbcr):
    # rgb = tf.clip_by_value(rgb, 0, 1)
    M = tf.constant([0.00456621, 0.00456621, 0.00456621, 0.0, -0.00153632, 0.00791071, 0.00625893, -0.00318811, 0.0],
                    shape=[3, 3], dtype=tf.float32)
    M = M * 255
    b = tf.constant([16, 128, 128], shape=[1, 1, 1, 3], dtype=tf.float32)
    b = b / 255
    rgb = tf.tensordot(ycbcr - b, M, axes=1)

    return rgb


def tf_rgb2xyz(rgb):  # range 0~1
    indxyz = tf.cast(rgb > 0.04045, tf.float32)
    xyz = indxyz * tf.maximum((rgb + 0.055) / 1.055, 0) ** 2.4 + (1 - indxyz) * (rgb / 12.92)
    M = tf.constant([0.4124, 0.2126, 0.0193, 0.3576, 0.7152, 0.1192, 0.1805, 0.0722, 0.9505], shape=[3, 3],
                    dtype=tf.float32)
    xyz = tf.tensordot(xyz, M, axes=1)

    return xyz


def tf_xyz2lab(xyz):
    S = tf.constant([100 / 95.047, 100 / 100.000, 100 / 108.883], shape=[1, 3], dtype=tf.float32)
    S = tf.expand_dims(S, axis=0)
    S = tf.expand_dims(S, axis=0)

    xyz_t = xyz * S
    indlab = tf.cast(xyz_t > 0.008856, tf.float32)
    xyz_t = indlab * tf.maximum(xyz_t, 0) ** (1 / 3) + (1 - indlab) * (7.787 * xyz_t + 16 / 116)

    x = xyz_t[..., :1]
    y = xyz_t[..., 1:2]
    z = xyz_t[..., 2:]

    l = 116 * y - 16
    a = 500 * (x - y)
    b = 200 * (y - z)

    lab = tf.concat([l, a, b], axis=3)
    lab = lab / 100

    return lab


def metric_ergas(x1, x2, sc, eps=1e-10):  # x2 = ms input, sc = integer
    """ Compute ERGAS metric between x1 and x2. sc: scale ratio. eps: very small value """
    y1 = x1 + 0
    y2 = x2 + 0

    d = tf.square(y2 - y1)
    dm = tf.reduce_mean(d, axis=[1, 2], keepdims=True)
    y2m = tf.reduce_mean(y2, axis=[1, 2], keepdims=True)
    y = dm/(tf.square(y2m)+eps)
    y = tf.reduce_mean(y, axis=[1, 2, 3])
    y = (100./sc)*tf.sqrt(y+eps)
    y = tf.reduce_mean(y)

    return y


def metric_n_ergas(x1, x2, sc, eps=1e-10, ks=13):  # x2 = ms input, sc = integer
    """ Compute ERGAS metric between x1 and x2. sc: scale ratio. eps: very small value """
    x1 = tf.transpose(x1, perm=[3, 1, 2, 0])
    x2 = tf.transpose(x2, perm=[3, 1, 2, 0])

    sz = tf.shape(x1)
    print(sz)

    y1 = tf.extract_image_patches(x1, ksizes=[1, sz[1], sz[2], 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='reflect')
    y2 = tf.tile(x2, [1, 1, 1, ks*ks])

    d = tf.square(y2 - y1)
    dm = tf.reduce_mean(d, axis=[1, 2], keepdims=True)
    y2m = tf.reduce_mean(y2, axis=[1, 2], keepdims=True)
    y = dm/(tf.square(y2m)+eps)
    y = tf.reduce_mean(y, axis=[1, 2, 3])
    y = (100./sc)*tf.sqrt(y+eps)
    y = tf.minimum(y)
    y = tf.reduce_mean(y)

    return y


def sobelfilterMetric(x):
    """ Apply sobel filter to the input """
    y = tf.identity(x)
    szy = y.shape.as_list()

    bf = tf.constant([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=tf.float32, shape=[3, 3, 1, 1])
    # bf = tf.transpose(bf, perm=[1, 0, 2, 3])
    bf = tf.tile(bf, [1, 1, szy[3], 1])

    pp = int(1)
    y = tf.pad(y, [[0, 0], [pp, pp], [pp, pp], [0, 0]])
    y = tf.nn.depthwise_conv2d(y, bf, strides=[1, 1, 1, 1], padding='VALID')

    return y


def metric_scc(x1, x2, eps=1e-10):  # x2 = pan input
    """ SCC metric between x1 and x2"""
    y1 = x1 + 0
    y2 = x2 + 0

    y2 = tf.tile(y2, [1, 1, 1, y1.shape.as_list()[3]])

    sz1 = y1.shape.as_list()[3]
    sz2 = y2.shape.as_list()[3]
    sz3 = max(sz1, sz2)

    y = tf.concat([y1, y2], axis=3)
    y = sobelfilterMetric(y)
    y1, y2 = tf.split(y, [sz1, sz2], axis=3)

    # listcat = [y1*y1, y2*y2, y1*y2]
    # listcat = tf.stack(listcat, axis=4)
    # # listcat = gaussblur(listcat, fs, (fs+1)/6)
    # listcat = tf.reduce_sum(listcat, axis=[0, 1, 2, 3])
    # m11, m22, m12 = tf.unstack(listcat, axis=0)
    m11 = tf.reduce_sum(y1*y1, axis=[1, 2, 3])
    m22 = tf.reduce_sum(y2*y2, axis=[1, 2, 3])
    m12 = tf.reduce_sum(y1*y2, axis=[1, 2, 3])

    cov12 = m12
    v1 = tf.abs(m11) + eps
    v2 = tf.abs(m22) + eps
    v1 = tf.sqrt(v1)
    v2 = tf.sqrt(v2)

    corr12 = cov12/v2/v1

    corr12 = tf.reduce_mean(corr12)

    return corr12

def metric_scc2(x1, x2, eps=1e-10):  # x2 = pan input
    """ SCC metric between x1 and x2"""
    y1 = x1 + 0
    y2 = x2 + 0
    y1 = tf.reduce_mean(y1,3,True)
    y2 = tf.tile(y2, [1, 1, 1, y1.shape.as_list()[3]])

    sz1 = y1.shape.as_list()[3]
    sz2 = y2.shape.as_list()[3]
    sz3 = max(sz1, sz2)

    y = tf.concat([y1, y2], axis=3)
    y = sobelfilterMetric(y)
    y1, y2 = tf.split(y, [sz1, sz2], axis=3)

    # listcat = [y1*y1, y2*y2, y1*y2]
    # listcat = tf.stack(listcat, axis=4)
    # # listcat = gaussblur(listcat, fs, (fs+1)/6)
    # listcat = tf.reduce_sum(listcat, axis=[0, 1, 2, 3])
    # m11, m22, m12 = tf.unstack(listcat, axis=0)
    m11 = tf.reduce_sum(y1*y1, axis=[1, 2, 3])
    m22 = tf.reduce_sum(y2*y2, axis=[1, 2, 3])
    m12 = tf.reduce_sum(y1*y2, axis=[1, 2, 3])

    cov12 = m12
    v1 = tf.abs(m11) + eps
    v2 = tf.abs(m22) + eps
    v1 = tf.sqrt(v1)
    v2 = tf.sqrt(v2)

    corr12 = cov12/v2/v1

    corr12 = tf.reduce_mean(corr12)

    return corr12


def makeBasedOnGaussSCCF2Fast_arg(y, x2, y2, sc, sp=1, ccfs=[25]):  # 2=arg, f=feat, s=search, x=output_blur, y=ms, x2=pan_blur, y2=gray_ms, sp=7, ccfs=27
    """ Creates aligned PAN-res MS images. Args: y: MS input, x2: PAN input, y2: grayed MS input, sc: scale ratio, sp: patch search range, ccfs: patch size"""

    N = int(sp**2)  # 49
    psp = int((sp-1)/2)  # padding, 3

    # channel 방향으로 search range를 쌓기 위한 사전작업
    ys = tf.pad(y, [[0, 0], [psp, psp], [psp, psp], [0, 0]], "REFLECT")
    ys = tf.extract_image_patches(ys, ksizes=[1, sp, sp, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')  # search range
    ys = my_resize_nearest(ys, sc)
    ys = tf.split(ys, N, axis=3)
    ys = tf.stack(ys, axis=-1)

    d = tf.stop_gradient(ys)

    x2f = tf.space_to_depth(x2, sc)
    x2f = sobelfilter2XY(x2f)
    x2f = tf.depth_to_space(x2f, sc)
    # x2f = sobelfilter2XY(x2)
    y2f = sobelfilter2XY(y2)

    y2f = tf.pad(y2f, [[0, 0], [psp, psp], [psp, psp], [0, 0]], "REFLECT")
    y2fs = tf.extract_image_patches(y2f, ksizes=[1, sp, sp, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')  # search range
    y2fs = my_resize_nearest(y2fs, sc)
    y2fs = tf.split(y2fs, N, axis=3)
    y2fs = tf.concat(y2fs, axis=0)
    x2fs = tf.tile(x2f, (N, 1, 1, 1))

    x2fs = tf.space_to_depth(x2fs, sc)
    y2fs = tf.space_to_depth(y2fs, sc)
    d2list = []
    for i in ccfs:
        d2list.append(CC_GaussFast(x2fs, y2fs, fs=i))
    d2 = tf.add_n(d2list)
    # d2 = CC_Gauss(x2fs, y2fs, fs=ccfs)
    # d2 = d2/(tf.maximum(boxfilterSum(d2, [sp, sp]), 1e-8))
    # d2 = gaussblur2(d2, 21)
    d2 = -d2
    d2 = tf.depth_to_space(d2, sc)
    d2 = tf.reduce_mean(d2, axis=3, keepdims=True)
    d2 = tf.split(d2, N, axis=0)
    d2 = tf.stack(d2, axis=-1)

    d2min = tf.reduce_min(d2, axis=4, keepdims=True)
    d2min = tf.cast(d2<=d2min, tf.float32)  # 이게 2개 이상일수도 그래서 sum을 나눔
    d2minsum = tf.reduce_sum(d2min, axis=4, keepdims=True)
    d2min = d2min/d2minsum  # important
    d3 = tf.stop_gradient(d2min)*d
    d3 = tf.reduce_sum(d3, axis=4)

    return d3


def makeBasedOnGaussSCCF2Fast_arg_ms_scale(ms, pan, ms_gray, sc, sp=1, ccfs=[25]):  # 2=arg, f=feat, s=search, x=output_blur, y=ms, x2=pan_blur, y2=gray_ms, sp=7, ccfs=27
    """ Creates aligned PAN-res MS images. Args: y: MS input, x2: PAN input, y2: grayed MS input, sc: scale ratio, sp: patch search range, ccfs: patch size"""

    N = int(sp**2)  # 49
    psp = int((sp-1)/2)  # padding, 3


    # channel 방향으로 search range를 쌓기 위한 사전작업
    ms = tf.pad(ms, [[0, 0], [psp, psp], [psp, psp], [0, 0]], "REFLECT")
    ms = tf.extract_image_patches(ms, ksizes=[1, sp, sp, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')  # search range
    ms = tf.split(ms, N, axis=3)
    ms = tf.stack(ms, axis=-1)

    d = tf.stop_gradient(ms)

    pan = sobelfilter2XY(pan)
    ms_gray = sobelfilter2XY(ms_gray)

    ms_gray = tf.pad(ms_gray, [[0, 0], [psp, psp], [psp, psp], [0, 0]], "REFLECT")
    ms_gray = tf.extract_image_patches(ms_gray, ksizes=[1, sp, sp, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')  # search range
    ms_gray = tf.split(ms_gray, N, axis=3)
    ms_gray = tf.concat(ms_gray, axis=0)
    pan = tf.tile(pan, (N, 1, 1, 1))

    d2list = []
    for i in ccfs:
        d2list.append(CC_GaussFast(pan, ms_gray, fs=i))
    d2 = tf.add_n(d2list)
    # d2 = CC_Gauss(x2fs, y2fs, fs=ccfs)
    # d2 = d2/(tf.maximum(boxfilterSum(d2, [sp, sp]), 1e-8))
    # d2 = gaussblur2(d2, 21)
    d2 = -d2
    d2 = tf.reduce_mean(d2, axis=3, keepdims=True)
    d2 = tf.split(d2, N, axis=0)
    d2 = tf.stack(d2, axis=-1)

    d2min = tf.reduce_min(d2, axis=4, keepdims=True)
    d2min = tf.cast(d2<=d2min, tf.float32)  # 이게 2개 이상일수도 그래서 sum을 나눔
    d2minsum = tf.reduce_sum(d2min, axis=4, keepdims=True)
    d2min = d2min/d2minsum  # important
    d3 = tf.stop_gradient(d2min)*d
    d3 = tf.reduce_sum(d3, axis=4)

    return d3


def CC_GaussFast(x1, x2, fs=5, eps=1e-10):
    """ Calculates correlation between two inputs. Args: x1: input 1, x2: input 2, fs: correlation filter size, eps: very small value to avoid zero division"""
    y1 = x1 + 0
    y2 = x2 + 0

    sz1 = y1.shape.as_list()[3]
    sz2 = y2.shape.as_list()[3]
    sz3 = max(sz1, sz2)

    listcat = [y1, y2, y1*y1, y2*y2, y1*y2]
    listcat = tf.concat(listcat, axis=3)
    listcat = gaussblur_valid(listcat, fs, (fs+1)/6)
    pfs = int((fs-1)/2)
    listcat = pad(listcat, [[pfs, pfs], [pfs, pfs]])
    # listcat = boxfilter(listcat, fs)
    m1, m2, m11, m22, m12 = tf.split(listcat, [sz1, sz2, sz1, sz2, sz3], axis=3)

    cov12 = m12 - m1*m2
    v1 = tf.abs(m11 - m1*m1) + eps
    v2 = tf.abs(m22 - m2*m2) + eps
    v1 = tf.sqrt(v1)
    v2 = tf.sqrt(v2)

    corr12 = cov12/v2/v1

    return corr12


def pad(x, pp):
    """ Put padding to the input. Args: pp: padding amount e.g. [[pre_h, post_h], [pre_w, post_w]]"""
    y = tf.identity(x)
    if pp[0][0]!=0:
        y2 = y[:, :1, :, :]
        y2 = tf.tile(y2, [1, pp[0][0], 1, 1])
        y = tf.concat([y2, y], axis=1)

    if pp[0][1]!=0:
        y2 = y[:, -1:, :, :]
        y2 = tf.tile(y2, [1, pp[0][1], 1, 1])
        y = tf.concat([y, y2], axis=1)

    if pp[1][0]!=0:
        y2 = y[:, :, :1, :]
        y2 = tf.tile(y2, [1, 1, pp[1][0], 1])
        y = tf.concat([y2, y], axis=2)

    if pp[1][1]!=0:
        y2 = y[:, :, -1:, :]
        y2 = tf.tile(y2, [1, 1, pp[1][1], 1])
        y = tf.concat([y, y2], axis=2)

    return y


def gaussblur_valid(x, szf, fsig):
    """ Apply gaussian blur to the input (only valid region, without pre-padding). Args: szf: gaussian filter size. fsig: gaussian std"""
    y = tf.identity(x)
    szy = y.shape.as_list()

    bf = _tf_fspecial_gauss(szf, fsig)
    bf = tf.tile(bf, [1, 1, szy[3], 1])

    # pp = int((szf-1)/2)
    # y = tf.pad(y, [[0, 0], [pp, pp], [pp, pp], [0, 0]], "REFLECT")
    y = tf.nn.depthwise_conv2d(y, bf, strides=[1, 1, 1, 1], padding='VALID')
    return y


def tf_sharpen(im, amount=0.8, thr=0):
    """ Image sharpening based on gaussian filter """
    im_hf = im - gaussblur2(im, 5)  # edge map
    im_hf_abs = tf.abs(im_hf)
    max = tf.reduce_max(im_hf_abs, axis=[1, 2], keepdims=True) * thr
    thr_map = tf.cast((im_hf_abs > max), tf.float32)  # threshold map
    im_hf = im_hf * thr_map  # apply threshold
    im_hf = im_hf * amount  # apply amount of boosting
    im_sharp = im + im_hf  # apply boosting

    return im_sharp


def tf_sharpen_clip(im, amount=0.8, thr=0):
    """ Image sharpening based on gaussian filter """
    im_hf = im - gaussblur2(im, 5)  # edge map
    im_hf_abs = tf.abs(im_hf)
    max = tf.reduce_max(im_hf_abs, axis=[1, 2], keepdims=True) * thr
    thr_map = tf.cast((im_hf_abs > max), tf.float32)  # threshold map
    im_hf = im_hf * thr_map  # apply threshold
    im_hf = im_hf * amount  # apply amount of boosting
    im_sharp = im + im_hf  # apply boosting

    ws = 5  # window size
    N = int(ws**2)
    ps = int((ws-1)/2)  # padding

    y = tf.pad(im_sharp, [[0, 0], [ps, ps], [ps, ps], [0, 0]], "REFLECT")
    y = tf.extract_image_patches(y, ksizes=[1, ws, ws, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')  # (1, H, W, 3*25)
    y = tf.split(y, N, axis=-1)
    y = tf.stack(y, axis=-1)
    y_min = tf.reduce_min(y, axis=-1)
    y_max = tf.reduce_max(y, axis=-1)

    im_sharp_clip = tf.clip_by_value(im_sharp, y_min, y_max)

    return im_sharp_clip


def tf_clip(im, ref):
    """ Image sharpening based on gaussian filter """

    ws = 5  # window size
    N = int(ws**2)
    ps = int((ws-1)/2)  # padding

    y = tf.pad(ref, [[0, 0], [ps, ps], [ps, ps], [0, 0]], "REFLECT")
    y = tf.extract_image_patches(y, ksizes=[1, ws, ws, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')  # (1, H, W, 3*25)
    y = tf.split(y, N, axis=-1)
    y = tf.stack(y, axis=-1)
    y_min = tf.reduce_min(y, axis=-1)
    y_max = tf.reduce_max(y, axis=-1)

    im_sharp_clip = tf.clip_by_value(im, y_min, y_max)

    return im_sharp_clip
