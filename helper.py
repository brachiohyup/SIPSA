import numpy as np
import tifffile
import random
from PIL import Image
import os
from scipy.ndimage import gaussian_filter
import os.path
import sys

# Nearest-neighbor interpolation
def imresizenn(imh, sc):
    imh = np.array(imh)
    sz = np.shape(imh)
    sz = np.array(sz)
    if np.size(sz) == 2:  # for 1 channel inputs
        sc = np.array(sc)
        if sc < 1:
            sz = np.dot(np.floor(np.dot(sz, 1.0 / sc)), sc)
        sz = sz.astype(int)
        imh = imh[:sz[0], :sz[1]]
        sz2 = np.dot(sz, sc).astype(int)
        imh = Image.fromarray(imh)
        iml = imh.resize((sz2[1], sz2[0]), resample=Image.NEAREST)
    else:  # for multi-channel inputs (e.g. RGB)
        ch = sz[2]
        sz = sz[:2]
        sc = np.array(sc)
        if sc < 1:
            sz = np.dot(np.floor(np.dot(sz, 1.0 / sc)), sc)
        sz = sz.astype(int)
        imh = imh[:sz[0], :sz[1], :]
        sz2 = np.dot(sz, sc).astype(int)
        if ch == 3:
            imh = Image.fromarray(imh)
            iml = imh.resize((sz2[1], sz2[0]), resample=Image.NEAREST)
        else:
            iml = np.stack(
                [np.array(Image.fromarray(imh[:, :, i]).resize((sz2[1], sz2[0]), resample=Image.NEAREST)) for i in
                 range(ch)], 2)

    iml = np.array(iml)
    iml = np.round(iml)
    iml = np.clip(iml, 0, 255)

    return iml


# Linear stretch
def im_percent_norm(x, ms, p=(2, 98), eps=1 / (2 ** 10)):
    pv = np.percentile(ms, p, axis=(0, 1))
    y = x.astype(np.float32)
    pmin = pv[0, ...]
    pmax = pv[1, ...]
    y = np.clip(y, pmin, pmax)
    y = (y - pmin) / np.maximum((pmax - pmin), eps)

    return y



# Rearranges blocks of spatial data, into depth.
def space_to_depth(x, block_size):
    x = np.asarray(x)
    batch, height, width, depth = x.shape
    reduced_height = height // block_size
    reduced_width = width // block_size
    y = x.reshape(batch, reduced_height, block_size,
                  reduced_width, block_size, depth)
    z = np.swapaxes(y, 2, 3).reshape(batch, reduced_height, reduced_width, -1)
    return z


# Downscale an image with a scale factor of sc: degradation model for satellite
def satdwsize(x, sc):
    y = x.astype(np.float32)
    if np.size(y.shape) == 2:
        y = y[:, :, np.newaxis]
    vblur = 1 / sc
    y = gaussian_filter(y, (vblur, vblur, 0.0))

    y = np.transpose(y, (2, 0, 1))
    y = y[:, :, :, np.newaxis]
    y = space_to_depth(y, sc)
    y = np.mean(y, axis=3)
    y = np.transpose(y, (1, 2, 0))

    return y


# Crop an image to have a size that is integer multiple of sc
def cropn(imh, sc):  # sc>1
    imh = np.array(imh)
    sz = np.shape(imh)
    if np.size(sz) == 2:
        sc = np.array(sc)
        sz = np.dot(np.floor(np.dot(sz, 1.0 / sc)), sc)
        sz = sz.astype(int)
        imh = imh[:sz[0], :sz[1]]
    else:
        sz = sz[:2]
        sc = np.array(sc)
        sz = np.dot(np.floor(np.dot(sz, 1.0 / sc)), sc)
        sz = sz.astype(int)
        imh = imh[:sz[0], :sz[1], :]

    return imh


# Get files with certain extension in a folder including all sub-folders
def getpath_subdirs_any(dir1, extlist=[".npy"]):
    dir_list = list()
    dir_list.append(dir1)
    for dirname, dirnames, filenames in os.walk(dir1):
        for subdirname in dirnames:
            dir_path = os.path.join(dirname, subdirname)
            dir_list.append(dir_path)

    def is_fun(filename):
        return any(filename.endswith(extension) for extension in extlist)

    def mysort(x):
        return sorted(x, key=lambda item: (len(item), item))

    imlist = list()
    for image_dir in mysort(dir_list):
        for x in mysort(os.listdir(image_dir)):
            if is_fun(x):
                imlist.append(os.path.join(image_dir, x))

    return imlist


# Extract subimages from each training image (number of subimages per image = NUM_SUB_PER_IM)
def im2subim(NUM_SUB_PER_IM, SUBIM_SIZE_LR, imlistPAN, imlistMS, imlistnir, imlist_mask, SCALE):
    # iml = imh = MS, imMS2 = MS2, imb = PAN

    cropListPAN = list()
    cropListMS = list()
    cropListMS2 = list()
    cropListmask = list()
    for imIndex in range(len(imlistPAN)):
        im_pan = tifffile.imread(imlistPAN[imIndex])
        im_pan = im_pan[:, :, np.newaxis]
        im_ms = tifffile.imread(imlistMS[imIndex])
        im_ms2 = tifffile.imread(imlistnir[imIndex])
        im_mask = Image.open(imlist_mask[imIndex])

        im_pan = im_pan.astype(np.float32)
        im_ms = im_ms.astype(np.float32)
        im_ms2 = im_ms2.astype(np.float32)
        # im_mask = im_mask.astype(np.float32)
# mask = Image.open(testmask[itest])
# mask = np.array(mask, dtype=int)
        im_mask = np.array(im_mask, dtype=int)
        im_mask = np.expand_dims(im_mask,2)

        sz_pan = im_pan.shape
        sz_pan = np.array(sz_pan, dtype=np.float32)
        sz_ms = np.floor(sz_pan / SCALE)
        sz_pan = sz_ms * SCALE
        sz_pan = sz_pan.astype(np.int)
        sz_ms = sz_ms.astype(np.int)

        im_pan = im_pan[:sz_pan[0], :sz_pan[1], :]
        im_ms = im_ms[:sz_ms[0], :sz_ms[1], :]
        im_ms2 = im_ms2[:sz_ms[0], :sz_ms[1], :]
        im_mask = im_mask[:sz_ms[0], :sz_ms[1], :]


        sz0 = SUBIM_SIZE_LR

        ih, iw, _ = im_ms.shape
        nw, nh = iw - sz0, ih - sz0

        for subImIndex in range(NUM_SUB_PER_IM):
            if nw == 0:
                indw = 0
            else:
                indw = random.randint(0, nw)
            if nh == 0:
                indh = 0
            else:
                indh = random.randint(0, nh)

            im_panC = im_pan[indh * SCALE:(indh + sz0) * SCALE, indw * SCALE:(indw + sz0) * SCALE, :]  # PAN scale
            im_msC = im_ms[indh:(indh + sz0), indw:(indw + sz0), :]  # MS scale
            im_ms2C = im_ms2[indh :(indh + sz0)  , indw  :(indw + sz0)  , :]  # PAN scale
            im_maskC = im_mask[indh :(indh + sz0) , indw :(indw + sz0) , :]  # PAN scale

            cropListPAN.append(im_panC)
            cropListMS.append(im_msC)
            cropListMS2.append(im_ms2C)
            cropListmask.append(im_maskC)

    batch_temp = [[cropListMS[cropImIndex], cropListPAN[cropImIndex], cropListMS2[cropImIndex], cropListmask[cropImIndex]] for cropImIndex in
                  range(len(cropListMS))]

    im_ms, im_pan, im_ms2, im_mmask = zip(*batch_temp)

    return im_ms, im_pan, im_ms2, im_mmask



# Extract subimages from each training image (number of subimages per image = NUM_SUB_PER_IM)
def im2subim_align_ms_scale(NUM_SUB_PER_IM, SUBIM_SIZE_LR, imlistPAN, imlistMS, imlistMS2, SCALE):
    # iml = imh = MS, imMS2 = MS2, imb = PAN

    cropListPAN = list()
    cropListMS = list()
    cropListMS2 = list()
    for imIndex in range(len(imlistPAN)):
        im_pan = tifffile.imread(imlistPAN[imIndex])
        im_pan = im_pan[:, :, np.newaxis]
        im_ms = tifffile.imread(imlistMS[imIndex])
        im_ms2 = tifffile.imread(imlistMS2[imIndex])

        im_pan = im_pan.astype(np.float32)
        im_ms = im_ms.astype(np.float32)
        im_ms2 = im_ms2.astype(np.float32)

        sz_pan = im_pan.shape
        sz_pan = np.array(sz_pan, dtype=np.float32)
        sz_ms = np.floor(sz_pan / SCALE)
        sz_pan = sz_ms * SCALE
        sz_pan = sz_pan.astype(np.int)
        sz_ms = sz_ms.astype(np.int)

        im_pan = im_pan[:sz_pan[0], :sz_pan[1], :]
        im_ms = im_ms[:sz_ms[0], :sz_ms[1], :]
        im_ms2 = im_ms2[:sz_ms[0], :sz_ms[1], :]

        sz0 = SUBIM_SIZE_LR

        ih, iw, _ = im_ms.shape
        nw, nh = iw - sz0, ih - sz0

        for subImIndex in range(NUM_SUB_PER_IM):
            if nw == 0:
                indw = 0
            else:
                indw = random.randint(0, nw)
            if nh == 0:
                indh = 0
            else:
                indh = random.randint(0, nh)

            im_panC = im_pan[indh * SCALE:(indh + sz0) * SCALE, indw * SCALE:(indw + sz0) * SCALE, :]  # PAN scale
            im_msC = im_ms[indh:(indh + sz0), indw:(indw + sz0), :]  # MS scale
            im_ms2C = im_ms2[indh:(indh + sz0), indw:(indw + sz0), :]  # MS scale

            cropListPAN.append(im_panC)
            cropListMS.append(im_msC)
            cropListMS2.append(im_ms2C)

    batch_temp = [[cropListMS[cropImIndex], cropListPAN[cropImIndex], cropListMS2[cropImIndex]] for cropImIndex in
                  range(len(cropListMS))]

    im_ms, im_pan, im_ms2 = zip(*batch_temp)

    return im_ms, im_pan, im_ms2


class saveData():
    def __init__(self, type,timestamp):
        self.save_dir = os.path.join('./log_' + str(type),timestamp)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        log_name = '/' + type + '_log.txt'
        if os.path.exists(self.save_dir + log_name):
            self.logFile = open(self.save_dir + log_name, 'a')
        else:
            self.logFile = open(self.save_dir + log_name, 'w')

    def save_log(self, log):
        print(log)
        sys.stdout.flush()
        self.logFile.write(log + '\n')
        self.logFile.flush()
