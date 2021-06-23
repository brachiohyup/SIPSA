import tensorflow as tf
import helperTF, deformable
import tensorflow as tf



def local_conv(img, kernel_2d, channels, kernel_size):
    # A new version local convolution implementation. Faster than the original one.
    # img (B, H, W, channels)
    # kernel_2d (B, H, W, k*k)
    # k == kernel, c == channel
    img = tf.image.extract_image_patches(img, ksizes=(1, kernel_size, kernel_size, 1), strides=(1, 1, 1, 1),
                                         rates=(1, 1, 1, 1), padding="SAME")  # Output [H, W, k*k*c]
    # Output [H, W, k*k*c]
    img = tf.split(img, kernel_size * kernel_size, axis=-1)  # k*k tensors of [H, W, c]
    img = tf.stack(img, axis=3)  # [H, W, k*k, c]

    ### kernel_2d shape : [B, H, W, kk]
    kernel_2d = tf.expand_dims(kernel_2d, -1)  # [H, W, k*k, 1]

    kernel_2d = tf.tile(kernel_2d, [1, 1, 1, 1, channels])  # [H, W, k*k, c]
    result = tf.multiply(img, kernel_2d)  # Elementwise multiplication. Resulting [H, W, k*k, c]
    result = tf.reduce_sum(result, axis=3)  # (H, W, c)
    return result


def LEEHYUBII_please_with_softmax_81(x, bic, n, sc, name, reuse=False):  # x: MS, bic: PAN, x2: nir
    """ Our unsupervised net with aligned HR MS targets - 190826"""
    """Alignment and Colorization End-to-End learning"""
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        opts = {'ini': 1 / 10,  # used for initializing weights
                'ty': 'npau',  # initialize type
                'nch': 64,  # number of channels
                'nlr': 14,  # number of layers
                'leak': 0.1,  # negative slope of leaky ReLU
                'fs': 3,  # filter size
                'sc': sc,  # scale
                'bias': True,
                }
        opts2 = {'ini': 1 / 10,  # used for initializing weights
                 'ty': 'npau',  # initialize type
                 'nch': 32,  # number of channels
                 'nlr': 28,  # number of layers
                 'leak': 0.1,  # negative slope of leaky ReLU
                 'fs': 9,  # filter size
                 'sc': sc,  # scale
                 'bias': True,
                 }
        lno = 0  # layer count
        ch = opts['nch']
        chout = 3  # ch of output (RGB)
        blur_x = helperTF.my_resize_nearest(x, 4)
        # blur_n = helperTF.my_resize_nearest(n, 4)

        # blur_x = helperTF.gaussblur2(blur_x, 5, 1)


        # blur_n = helperTF.gaussblur2(blur_n, 5, 1)
        # n1, n2, n3 = tf.split(blur_n, 3, 3)
        # blur_n = n2
        bicdw = tf.space_to_depth(bic, sc)  # space_to_depth PAN
        y0 = tf.concat([bic, blur_x], axis=3)  # concat space_to_depth PAN and MS
        # y0 = tf.concat([bicdw, blur_x], axis=3)  # concat space_to_depth PAN and MS
        y = y0 + 0
        # PAN 1/16 + NIR, RGB

        y = tf.concat([bicdw,x],3)
        lno += 1
        y = helperTF.l_conv(y, 32, "conv%02d" % lno, opts2)
        y = helperTF.lrelu(y, opts['leak'])
        lno += 1
        y = helperTF.l_conv(y, 32, "conv%02d" % lno, opts2)
        for i in range(0, 1, 1):
            yi = y + 0

            y = helperTF.lrelu(y, opts['leak'])
            lno += 1
            y = helperTF.l_conv(y, 32, "conv%02d" % lno, opts)

            y = y + yi
        y = helperTF.lrelu(y, opts['leak'])

        lno += 1
        y = helperTF.l_conv(y, 81, "conv%02d" % lno, opts)
        skip_256 = y

        for i in range(0, 7, 1):
            yi = y + 0

            y = helperTF.lrelu(y, opts['leak'])
            lno += 1
            y = helperTF.l_conv(y, 81, "conv%02d" % lno, opts)

            y = y + yi
        y = helperTF.lrelu(y, opts['leak'])

        lno += 1

        filter = y

        filter = tf.nn.softmax(filter,3)

        ch = opts['nch']
        chout = 3  # ch of output (RGB)
        blur_x = helperTF.gaussblur2(x, 5, 1)
        blur_x = x
        blur_n = helperTF.gaussblur2(n, 5, 1)
        n1, n2, n3 = tf.split(blur_n, 3, 3)

        bicdw = tf.space_to_depth(bic, sc)  # space_to_depth PAN

        y0 = tf.concat([bicdw, blur_x], axis=3)  # concat space_to_depth PAN and MS
        # y0 = bicdw

        x2 = helperTF.my_resize_nearest(x, sc)  # upscaled MS -> used for residual learning
        xp2 = tf.concat([bic, bic, bic], 3)
        xp2 = tf.concat([xp2, xp2], 3)
        # y = y0 + 0
        # y1 = tf.concat([blur_x, n1], axis=3)  #
        y1 = x
        y1 = y1 + 0



        lno += 1

        y1 = helperTF.l_conv(y1, 64, "conv%02d" % lno, opts)
        for i in range(0, 3, 1):
            yi = y1 + 0

            y1 = helperTF.lrelu(y1, opts['leak'])
            lno += 1
            y1 = helperTF.l_conv(y1, 64, "conv%02d" % lno, opts)

            y1 = y1 + yi

        #
        #
        y1 = local_conv(y1, filter, 64, 9)
        ############################
        lno += 1
        pan_latter = bicdw
        y1 = helperTF.pixel_wise_convolution(y1, 4, "conv%02d" % lno, opts)
        aligned = y1
        lno += 1

        y = tf.concat([pan_latter, y1], 3)

        lno += 1
        y = helperTF.l_conv(y, 64, "conv%02d" % lno, opts)
        # y = helperTF.lrelu(y, opts['leak'])

        # stack convolution layers
        for i in range(0, 5, 1):
            yi = y + 0

            y = helperTF.lrelu(y, opts['leak'])
            lno += 1
            y = helperTF.l_conv(y, 64, "conv%02d" % lno, opts)

            y = y + yi
        lno += 1
        y = helperTF.l_SP2conv2(y, 4, sc, "conv%02d" % lno, opts)
        lno += 1
        y = helperTF.l_conv(y, 64, "conv%02d" % lno, opts)
        for i in range(0, 2, 1):
            yi = y + 0
            y = helperTF.lrelu(y, opts['leak'])
            lno += 1
            y = helperTF.l_conv(y, 64, "conv%02d" % lno, opts)
            y = y + yi

        edge_bic = tf.abs(helperTF.sobelfilter2XY(bic))
        y = tf.concat([y, edge_bic], 3)
        lno += 1
        y = helperTF.pixel_wise_convolution(y, 4, "conv%02d" % lno, opts)  # #
        median = helperTF.my_resize_nearest(x, 2)  # upscaled MS -> used for residual learning

    return y, median, aligned

