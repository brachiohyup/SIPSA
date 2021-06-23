import tensorflow as tf
import helperTF

def tf_flatten(a):
    """Flatten tensor"""
    return tf.reshape(a, [-1])

def tf_repeat(a, repeats, axis=0):
    a = tf.expand_dims(a, -1)
    a = tf.tile(a, [1, repeats])
    a = tf_flatten(a)
    return a

def tf_map_coordinates(input, coords):
    """
    :param input: tf.Tensor. shape=(h, w)
    :param coords: tf.Tensor. shape = (n_points, 2)
    :return:
    """
    coords_tl = tf.cast(tf.floor(coords), tf.int32)
    coords_br = tf.cast(tf.ceil(coords), tf.int32)
    coords_bl = tf.stack([coords_br[:, 0], coords_tl[:, 1]], axis=1)
    coords_tr = tf.stack([coords_tl[:, 0], coords_br[:, 1]], axis=1)
    vals_tl = tf.gather_nd(input, coords_tl)
    vals_br = tf.gather_nd(input, coords_br)
    vals_bl = tf.gather_nd(input, coords_bl)
    vals_tr = tf.gather_nd(input, coords_tr)
    coords_offset_tl = coords - tf.cast(coords_tl, tf.float32)
    vals_t = vals_tl + (vals_tr - vals_tl) * coords_offset_tl[:, 1]
    vals_b = vals_bl + (vals_br - vals_bl) * coords_offset_tl[:, 1]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_tl[:, 0]
    return mapped_vals

def tf_batch_map_coordinates(input, coords):
    """
    Batch version of tf_map_coordinates
    :param input: tf.Tensor. shape = (b, h, w)
    :param coords: tf.Tensor. shape = (b, n_points, 2)
    :return:
    """
    input_shape = tf.shape(input)
    batch_size = input_shape[0]
    input_size_h = input_shape[1]
    input_size_w = input_shape[2]
    n_coords = tf.shape(coords)[1]
    coords_w = tf.clip_by_value(coords[..., 1], 0, tf.cast(input_size_w, tf.float32) - 1)
    coords_h = tf.clip_by_value(coords[..., 0], 0, tf.cast(input_size_h, tf.float32) - 1)
    coords = tf.stack([coords_h, coords_w], axis=-1)
    coords_tl = tf.cast(tf.floor(coords), tf.int32)
    coords_br = tf.cast(tf.ceil(coords), tf.int32)
    coords_bl = tf.stack([coords_br[..., 0], coords_tl[..., 1]], axis=-1)
    coords_tr = tf.stack([coords_tl[..., 0], coords_br[..., 1]], axis=-1)
    idx = tf_repeat(tf.range(batch_size), n_coords)
    def _get_vals_by_coords(input, coords):
        indices = tf.stack([idx, tf_flatten(coords[..., 0]), tf_flatten(coords[..., 1])], axis=-1)
        vals = tf.gather_nd(input, indices)
        vals = tf.reshape(vals, (batch_size, n_coords))
        return vals

    vals_tl = _get_vals_by_coords(input, coords_tl)
    vals_br = _get_vals_by_coords(input, coords_br)
    vals_bl = _get_vals_by_coords(input, coords_bl)
    vals_tr = _get_vals_by_coords(input, coords_tr)

    coords_offset_tl = coords - tf.cast(coords_tl, 'float32')
    vals_t = vals_tl + (vals_tr - vals_tl) * coords_offset_tl[..., 1]
    vals_b = vals_bl + (vals_br - vals_bl) * coords_offset_tl[..., 1]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_tl[..., 0]
    return mapped_vals

def tf_batch_map_offsets(input, offsets):
    """
    :param input: tf.Tensor, shape=(b, h, w)
    :param offsets: tf.Tensor, shape=(b, h, w, 2)
    :return:
    """
    input_shape = tf.shape(input)
    batch_size = input_shape[0]
    input_size_h = input_shape[1]
    input_size_w = input_shape[2]
    offsets = tf.reshape(offsets, (batch_size, -1, 2))
    grid_x, grid_y = tf.meshgrid(tf.range(input_size_w), tf.range(input_size_h))
    grid = tf.stack([grid_y, grid_x], axis=-1)
    grid = tf.cast(grid, tf.float32)
    grid = tf.reshape(grid, (-1, 2))
    grid = tf.expand_dims(grid, axis=0)
    grid = tf.tile(grid, multiples=[batch_size, 1, 1])
    coords = offsets + grid
    mapped_vals = tf_batch_map_coordinates(input, coords)
    return mapped_vals

def to_bc_h_w_2(x, x_shape):
    """(b, h, w, 2c) -> (b*c, h, w, 2)"""
    x = tf.transpose(x, [0, 3, 1, 2])
    x = tf.reshape(x, [x_shape[0], x_shape[3], 2, x_shape[1], x_shape[2]])
    x = tf.transpose(x, [0, 1, 3, 4, 2])
    x = tf.reshape(x, [-1, x_shape[1], x_shape[2], 2])
    return x

def to_bc_h_w(x, x_shape):
    """(b, h, w, c) -> (b*c, h, w)"""
    x = tf.transpose(x, [0, 3, 1, 2])
    x = tf.reshape(x, [-1, x_shape[1], x_shape[2]])
    return x

def to_b_h_w_c(x, x_shape):
    """(b*c, h, w) -> (b, h, w, c)"""
    x = tf.reshape(x, (-1, x_shape[3], x_shape[1], x_shape[2]))
    x = tf.transpose(x, [0, 2, 3, 1])
    return x

def deformable_convolution(input,lno,opts):

    x=input
    conv_result = helperTF.l_conv(x, 64*3, "conv%02d" % lno, opts)

    offsets = conv_result[:, :, :, 0: 64 * 2-1]
    weights = tf.nn.sigmoid(conv_result[:, :, :, 64*2: 64*3-1])
    x_shape = tf.shape(x)
    x_shape_list = x.get_shape().as_list()
    x = to_bc_h_w(x, x_shape)
    offsets = to_bc_h_w_2(offsets, x_shape)
    weights = to_bc_h_w(weights, x_shape)
    x_offset = tf_batch_map_offsets(x, offsets)
    weights = tf.expand_dims(weights, axis=1)
    weights = to_b_h_w_c(weights, x_shape)
    x_offset = to_b_h_w_c(x_offset, x_shape)
    x_offset = tf.multiply(x_offset, weights)
    x_offset.set_shape(x_shape_list)
    return x_offset