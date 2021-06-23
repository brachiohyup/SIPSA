import os

os.environ["Path"] += os.pathsep + os.pathsep.join(
    ["C:\Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0_v7.6.1.34/bin",
     "C:\Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0_v7.6.1.34/libnvvp"])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
GPU_NUM = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_NUM)  # gpu index, cpu: ""

import numpy as np
from PIL import Image
import tensorflow as tf
import time, tifffile
import net
import helper, helperTF
import datetime
import zipfile
import socket

N_CPU = 6  # number of cpus to include
BATCH_SIZE = 4  # batch size
EPOCH_TOTAL = int(1e1)  # total number of epoch
LEARNING_RATE_BASE = 1e-4  # initial learning rate
WEIGHT_DECAY_BASE = 1e-7  # initial weight decay rate
ITER_TOTAL = 1e6/10  # total number of iterations

ITER_PER_EPOCH = int(ITER_TOTAL / EPOCH_TOTAL)  # number of iterations per epoch
SUBIM_SIZE_LR = int(128)  # training patch size of MS
SCALE = 4  # scale factor between PAN and MS image
SUBIM_SIZE_HR = int(SUBIM_SIZE_LR * SCALE)  # training patch size of PAN
CHANNEL_MS = 3  # number of channels in MS: RGB (4)
CHANNEL_P = 1  # number of channels in PAN
IMG_DIV = (2 ** 11) - 1  # WV3: 11 bit
NUM_SUB_PER_IM = int(30)  # number of sub-patch per image

TEST_STEP = 1  # run test every "TEST_STEP" epoch
NETWORK_NAME = "LEEHYUBII_please_with_softmax_81"

if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%m-%d-%H_%M")
    SAVE_DIR_PATH = os.path.join('./log_train', timestamp)
    os.makedirs(SAVE_DIR_PATH, exist_ok=True)
    print("Results will be saved in  =>  " + SAVE_DIR_PATH)
    save = helper.saveData('train', timestamp)
    save.save_log("IP Adress:" + str([l for l in (
    [ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")][:1], [
        [(s.connect(('8.8.8.8', 53)), s.getsockname()[0], s.close()) for s in
         [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) if l][0][0]))
    save.save_log("Proposed_10_16_0")

    save.save_log("GPU #: " + str(GPU_NUM))
    save.save_log("CHANNEL #: " + str(CHANNEL_MS))
    save.save_log("NETWORK: " + str(NETWORK_NAME) + ' x' + str(SCALE) + '\n')
    print("Please____ " )

    IMG_SAVE_DIR_PATH = SAVE_DIR_PATH + "/Imgs"
    os.makedirs(IMG_SAVE_DIR_PATH, exist_ok=True)

    IMG_TEMP_SAVE_DIR_PATH = IMG_SAVE_DIR_PATH + '/Temp'
    os.makedirs(IMG_TEMP_SAVE_DIR_PATH, exist_ok=True)

    PARAM_PATH = SAVE_DIR_PATH + '/Params_' + str(CHANNEL_MS) + 'ch_' + str(NETWORK_NAME)
    os.makedirs(PARAM_PATH, exist_ok=True)

    DATA_PATH = r'E:\WV3'

    EXT_LIST = [".tif"]
    list2_train_MS = helper.getpath_subdirs_any(DATA_PATH + '\Trgbn_TRAIN/RGB', EXT_LIST)
    list2_train_PAN = helper.getpath_subdirs_any(DATA_PATH + '\Trgbn_TRAIN/PAN', EXT_LIST)
    # list2_train_MS2 = helper.getpath_subdirs_any(DATA_PATH + '\Trgbn_TRAIN/NIR', EXT_LIST)
    list2_train_NIR = helper.getpath_subdirs_any(DATA_PATH + '\Trgbn_TRAIN/NIR', EXT_LIST)
    list2_train_mask = helper.getpath_subdirs_any(DATA_PATH + '\Trgbn_TRAIN/mask', EXT_LIST)

    list2_train_mask2 = helper.getpath_subdirs_any(DATA_PATH + '\Trgbn_TRAIN/mask2', EXT_LIST)
    list2_train_MS2 = helper.getpath_subdirs_any(DATA_PATH + '\Trgbn_TRAIN/RGB2', EXT_LIST)
    list2_train_PAN2 = helper.getpath_subdirs_any(DATA_PATH + '\Trgbn_TRAIN/PAN2', EXT_LIST)
    list2_train_NIR2 = helper.getpath_subdirs_any(DATA_PATH + '\Trgbn_TRAIN/NIR2', EXT_LIST)
    list2_train_mask3 = helper.getpath_subdirs_any(DATA_PATH + '\Trgbn_TRAIN/mask2', EXT_LIST)
    list2_train_MS3 = helper.getpath_subdirs_any(DATA_PATH + '\Trgbn_TRAIN/RGB2', EXT_LIST)
    list2_train_PAN3 = helper.getpath_subdirs_any(DATA_PATH + '\Trgbn_TRAIN/PAN2', EXT_LIST)
    list2_train_NIR3 = helper.getpath_subdirs_any(DATA_PATH + '\Trgbn_TRAIN/NIR2', EXT_LIST)

    list_train_MS = list2_train_MS + list2_train_MS2 + list2_train_MS3
    list_train_PAN = list2_train_PAN + list2_train_PAN2 + list2_train_PAN3
    list_train_NIR = list2_train_NIR + list2_train_NIR2 + list2_train_NIR3
    list_train_mask = list2_train_mask + list2_train_mask2 +  list2_train_mask3
    print("Training MS image number : %d " % len(list_train_MS))
    print("Training PAN image number : %d " % len(list_train_PAN))
    print("Test NIR image number : %d " % len(list_train_NIR))
    print("Test HOLE image number : %d " % len(list_train_mask))

    print("road Training MS image number : %d " % len(list2_train_MS2))
    print("road Training PAN image number : %d " % len(list2_train_PAN2))
    print("road Test NIR image number : %d " % len(list2_train_NIR2))
    print("road Test HOLE image number : %d " % len(list2_train_mask2))

    TEST_FOLDER = 'Test_Rand100_03-01-02_08'

    TEST_PATH = os.path.join(DATA_PATH, TEST_FOLDER)
    list_test_MS = helper.getpath_subdirs_any(TEST_PATH + '/RGB', EXT_LIST)
    list_test_PAN = helper.getpath_subdirs_any(TEST_PATH + '/PAN', EXT_LIST)
    list_test_MS2 = helper.getpath_subdirs_any(TEST_PATH + '/NIR', EXT_LIST)
    save.save_log("\nTest Set: " + str(TEST_FOLDER) + '\n')

    # Learning rate / Weight decay rate
    global_step = tf.Variable(0, name='global_step', trainable=False)

    global_rate = tf.compat.v1.train.piecewise_constant(global_step, [int(ITER_PER_EPOCH * EPOCH_TOTAL / 2 - 1)],
                                              [1e0, 1e-1])
    learning_rate = LEARNING_RATE_BASE * global_rate
    wd = WEIGHT_DECAY_BASE * global_rate

    def ds_gen():
        while True:
            indrand = np.random.randint(len(list_train_PAN))
            filename_pan = list_train_PAN[indrand]
            filename_ms = list_train_MS[indrand]
            filename_nir = list_train_NIR[indrand]
            filename_mask = list_train_mask[indrand]

            imlistPAN = [filename_pan]
            imlistMS = [filename_ms]
            imlistNIR = [filename_nir]
            imlistmask= [filename_mask]

            ms_bat, pan_bat, ms2_bat, mask_bat = helper.im2subim(NUM_SUB_PER_IM, SUBIM_SIZE_LR, imlistPAN, imlistMS, imlistNIR,imlistmask, SCALE)
            ms_bat = np.array(ms_bat, dtype=np.float32) / IMG_DIV
            pan_bat = np.array(pan_bat, dtype=np.float32) / IMG_DIV
            ms2_bat = np.array(ms2_bat, dtype=np.float32) / IMG_DIV
            mask_bat = np.array(mask_bat, dtype=np.int)

            yield ms_bat, pan_bat, ms2_bat, mask_bat

    def tf_ds_gen():
        dataset0 = (tf.data.Dataset.from_generator(ds_gen, (tf.float32, tf.float32, tf.float32, tf.float32)))
        return dataset0

    dataset = (tf.data.Dataset.range(100).repeat()
               .apply(
        tf.data.experimental.parallel_interleave(lambda filename: tf_ds_gen(), cycle_length=int(N_CPU), sloppy=False))
               .apply(tf.data.experimental.unbatch())
               .shuffle(buffer_size=int(NUM_SUB_PER_IM * 10))
               .batch(int(BATCH_SIZE))
               )

    iterator = dataset.make_one_shot_iterator()

    ms_bat, pan_bat, ms2_bat, mask_bat = iterator.get_next()  # get next batch

    ms_bat.set_shape((BATCH_SIZE, SUBIM_SIZE_LR, SUBIM_SIZE_LR, CHANNEL_MS))
    pan_bat.set_shape((BATCH_SIZE, SUBIM_SIZE_HR, SUBIM_SIZE_HR, CHANNEL_P))
    ms2_bat.set_shape((BATCH_SIZE, SUBIM_SIZE_LR, SUBIM_SIZE_LR, CHANNEL_MS))
    mask_bat.set_shape((BATCH_SIZE, SUBIM_SIZE_LR, SUBIM_SIZE_LR, CHANNEL_P))

    ms_bat, pan_bat, ms2_bat, mask_bat = helperTF.tf_aug([ms_bat, pan_bat, ms2_bat, mask_bat])  # data augmentation

    ms_test = tf.placeholder(tf.float32, (1, None, None, CHANNEL_MS))
    pan_test = tf.placeholder(tf.float32, (1, None, None, CHANNEL_P))
    ms2_test = tf.placeholder(tf.float32, (1, None, None, CHANNEL_MS))
    mask_test = tf.placeholder(tf.float32, (1, None, None, CHANNEL_P))
    # ms2_test = tf.placeholder(tf.float32, (1, None, None, CHANNEL_MS))

    # Function to run the network
    def op_net0(ms, pan, ms2, mask, isTraining=True):  # yl_t: MS, yb_t: PAN, t_ms2: Aligned MS
        if isTraining:
            ms = ms
            pan = pan
            ms2 = ms2
        else:
            ms = ms
            pan = pan
            ms2 = ms2

        # Get mean and standard deviation of Input
        def getMeanStd(x):
            fs = 9  # window size
            eps = 1e-4
            cat = [x, x * x]
            cat = tf.concat(cat, axis=3)
            cat = helperTF.gaussblur2(cat, fs)
            ma, maa = tf.split(cat, 2, axis=3)
            stda = maa - ma * ma
            stda = tf.sqrt(tf.abs(stda) + eps)
            return ma, stda

        # Resize by nearest and apply gaussian blur
        def resize_blur(x, SCALE, fs=5):
            y = helperTF.my_resize_nearest(x, SCALE)
            y = helperTF.gaussblur2(y, fs)
            return y
        def resize_nn(x, SCALE, fs=5):
            y = helperTF.my_resize_nearest(x, SCALE)
            return y
        # Get mean and standard deviation
        Mm, Ms = getMeanStd(ms)
        Mm2, Ms2 = getMeanStd(ms2)
        Pm, Ps = getMeanStd(helperTF.satdwsize(pan, SCALE))  # Down-scale PAN image and get normalization factor (for speed and efficiency)
        Pm = resize_blur(Pm, SCALE)  # Up-scale back to original size
        Ps = resize_blur(Ps, SCALE)

        # Apply normalization
        ms = (ms-Mm)/Ms
        pan = (pan-Pm)/Ps
        ms2 = (ms2-Mm2)/Ms2
        Mm = resize_blur(Mm,SCALE)
        Ms = resize_blur(Ms,SCALE)

        # Run the network
        output, median_temp, aligned_temp= getattr(net, NETWORK_NAME)(ms, pan, ms2, SCALE, "G")

        # Loss functions
        if isTraining:

            # 512
            # Edge loss
            output_temp1, output_temp2, output_temp3, output_temp4 = tf.split(output,4,3)
            output_grey = tf.concat([output_temp1,output_temp2,output_temp3],3)
            # output_gray = tf.reduce_mean(output, axis=3, keepdims=True)
            output_gray = tf.reduce_mean(output_grey, axis=3, keepdims=True)

            out_gray_dx = helperTF.sobelfilter2XY(output_gray)
            pan_dx = helperTF.sobelfilter2XY(pan)
            out_gray_dx = tf.abs(out_gray_dx)
            pan_dx = tf.abs(pan_dx)
            lossE_og = tf.reduce_mean(tf.abs(out_gray_dx - pan_dx))

            # Color loss
            output_rgb = tf.concat([output_temp1,output_temp2,output_temp3],3)
            output_nir = tf.concat([output_temp4,output_temp4,output_temp4],3)
            ms1_up = resize_blur(ms,SCALE)
            ms2_up = resize_blur(ms2,SCALE)
            ms2_up1,ms2_up2,ms2_up3 = tf.split(ms2_up,3,3)
            ms2_up = tf.concat([ms2_up2,ms2_up2,ms2_up2],3)
            small_mask =mask
            median_mask = resize_nn(mask,2)
            mask = resize_nn(mask,SCALE)



            # You can implement this loss function, shifted MS image generation with Extract_patch.
            # Implementation can be done very easily.
            # For your understanding, I change the main loss function code for intuitive understanding.
            gt_big_ms1_x1 = tf.roll(ms1_up, 4, 1)
            gt_big_ms1_x1_y_1 = tf.roll(gt_big_ms1_x1, -4, 2)
            gt_big_ms1_x1_y_2 = tf.roll(gt_big_ms1_x1, 8, 2)
            gt_big_ms1_x1_y_3 = tf.roll(gt_big_ms1_x1, -8, 2)

            gt_big_ms1_y1 = tf.roll(ms1_up, 4, 2)
            gt_big_ms1_x1_1 = tf.roll(gt_big_ms1_x1, 4, 2)
            gt_big_ms1_x2 = tf.roll(ms1_up, 8, 1)
            gt_big_ms1_x2_y_2 = tf.roll(gt_big_ms1_x2, -8, 2)

            gt_big_ms1_y2 = tf.roll(ms1_up, 8, 2)
            gt_big_ms1_x2_1 = tf.roll(gt_big_ms1_x2, 8, 2)
            gt_big_ms1_x2_2 = tf.roll(gt_big_ms1_x2, -8, 2)
            gt_big_ms1_x2_3 = tf.roll(gt_big_ms1_x2, 4, 2)
            gt_big_ms1_x2_4 = tf.roll(gt_big_ms1_x2, -4, 2)

            gt_big_ms1_x3 = tf.roll(ms1_up, 16, 1)
            gt_big_ms1_y3 = tf.roll(ms1_up, 16, 2)
            gt_big_ms1_x3_1 = tf.roll(gt_big_ms1_x3, 16, 2)
            gt_big_ms1_x3_3 = tf.roll(gt_big_ms1_x3, 4, 2)
            gt_big_ms1_x3_4 = tf.roll(gt_big_ms1_x3, -4, 2)
            gt_big_ms1_x3_5 = tf.roll(gt_big_ms1_x3, 8, 2)
            gt_big_ms1_x3_6 = tf.roll(gt_big_ms1_x3, -8, 2)
            gt_big_ms1_x3_7 = tf.roll(gt_big_ms1_x3, 12, 2)
            gt_big_ms1_x3_8 = tf.roll(gt_big_ms1_x3, -12, 2)

            gt_big_ms1_x6_9 = tf.roll(gt_big_ms1_y3, 4, 1)
            gt_big_ms1_x6_10 = tf.roll(gt_big_ms1_y3, 8, 1)
            gt_big_ms1_x6_11 = tf.roll(gt_big_ms1_y3, -4, 1)
            gt_big_ms1_x6_12 = tf.roll(gt_big_ms1_y3, -8, 1)

            gt_16_1 = tf.concat([gt_big_ms1_x1_y_1,gt_big_ms1_x1_y_2,gt_big_ms1_x1_y_3,gt_big_ms1_x2_y_2,
                                 gt_big_ms1_x2_2,gt_big_ms1_x2_3,gt_big_ms1_x2_4,gt_big_ms1_x3_3,gt_big_ms1_x3_4,
                                 gt_big_ms1_x3_5,gt_big_ms1_x3_6,gt_big_ms1_x3_7,gt_big_ms1_x3_8,gt_big_ms1_x3_3,
                                 gt_big_ms1_x6_9, gt_big_ms1_x6_10,gt_big_ms1_x6_11,gt_big_ms1_x6_12],0)
            gt_big_ms1_x31 = tf.roll(ms1_up, 12, 1)
            gt_big_ms1_y31 = tf.roll(ms1_up, 12, 2)

            gt_big_ms1_x3_11 = tf.roll(gt_big_ms1_x31, 12, 2)
            gt_big_ms1_x3_11_1 = tf.roll(gt_big_ms1_x31, -12, 2)
            gt_big_ms1_x3_11_2 = tf.roll(gt_big_ms1_x31, -8, 2)
            gt_big_ms1_x3_11_3 = tf.roll(gt_big_ms1_x31, 8, 2)
            gt_big_ms1_x3_11_4 = tf.roll(gt_big_ms1_x31, 4, 2)
            gt_big_ms1_x3_11_5 = tf.roll(gt_big_ms1_x31, -4, 2)
            gt_big_ms1_x3_11_6 = tf.roll(gt_big_ms1_y31, -4, 1)
            gt_big_ms1_x3_11_7 = tf.roll(gt_big_ms1_y31, 4, 1)
            gt_big_ms1_x3_11_8 = tf.roll(gt_big_ms1_y31, -8, 1)
            gt_big_ms1_x3_11_9 = tf.roll(gt_big_ms1_y31, 8, 1)

            gt_big_ms1_x311 = tf.roll(ms1_up, -12, 1)
            gt_big_ms1_y311 = tf.roll(ms1_up, -12, 2)
            gt_big_ms1_x3_111 = tf.roll(gt_big_ms1_x311, -12, 2)
            gt_big_ms1_x3_111_1 = tf.roll(gt_big_ms1_x311, 12, 2)
            gt_big_ms1_x3_111_2 = tf.roll(gt_big_ms1_x311, 4, 2)
            gt_big_ms1_x3_111_3 = tf.roll(gt_big_ms1_x311, -4, 2)
            gt_big_ms1_x3_111_4 = tf.roll(gt_big_ms1_x311, 8, 2)
            gt_big_ms1_x3_111_5 = tf.roll(gt_big_ms1_x311, -8, 2)
            gt_big_ms1_x3_111_6 = tf.roll(gt_big_ms1_y311, 4, 1)
            gt_big_ms1_x3_111_7 = tf.roll(gt_big_ms1_y311, -4, 1)
            gt_big_ms1_x3_111_8 = tf.roll(gt_big_ms1_y311, 8, 1)
            gt_big_ms1_x3_111_9 = tf.roll(gt_big_ms1_y311, -8, 1)
            gt_12 = tf.concat([gt_big_ms1_x3_111_1,gt_big_ms1_x3_111_2,gt_big_ms1_x3_111_3,gt_big_ms1_x3_111_4,gt_big_ms1_x3_111_5,
                               gt_big_ms1_x3_111_6,gt_big_ms1_x3_111_7,gt_big_ms1_x3_111_8,gt_big_ms1_x3_111_9, gt_big_ms1_x3_11_1,
                               gt_big_ms1_x3_11_2,gt_big_ms1_x3_11_3,gt_big_ms1_x3_11_4,gt_big_ms1_x3_11_5,gt_big_ms1_x3_11_6,
                               gt_big_ms1_x3_11_7,gt_big_ms1_x3_11_8,gt_big_ms1_x3_11_9],0)

            gt_big_ms1_x4 = tf.roll(ms1_up, -4, 1)
            gt_big_ms1_x4_y_1 = tf.roll(gt_big_ms1_x4, 4, 2)
            gt_big_ms1_x4_y_2 = tf.roll(gt_big_ms1_x4, 8, 2)
            gt_big_ms1_x4_y_3 = tf.roll(gt_big_ms1_x4, -8, 2)


            gt_big_ms1_y4 = tf.roll(ms1_up, -4, 2)
            gt_big_ms1_x4_1 = tf.roll(gt_big_ms1_x4, -4, 2)

            gt_big_ms1_x5 = tf.roll(ms1_up, -8, 1)
            gt_big_ms1_x5_y2 = tf.roll(gt_big_ms1_x5, 8, 2)
            gt_big_ms1_x5_y3 = tf.roll(gt_big_ms1_x5, 4, 2)
            gt_big_ms1_x5_y4 = tf.roll(gt_big_ms1_x5, -4, 2)
            gt_4 = tf.concat([gt_big_ms1_x5_y2,gt_big_ms1_x5_y3,gt_big_ms1_x5_y4, gt_big_ms1_x4_y_1, gt_big_ms1_x4_y_2,
                              gt_big_ms1_x4_y_3],0)

            gt_big_ms1_y5 = tf.roll(ms1_up, -8, 2)
            gt_big_ms1_x5_1 = tf.roll(gt_big_ms1_x5, -8, 2)

            gt_big_ms1_x6 = tf.roll(ms1_up, -16, 1)
            gt_big_ms1_y6 = tf.roll(ms1_up, -16, 2)
            gt_big_ms1_x6_1 = tf.roll(gt_big_ms1_x6, -16, 2)
            gt_big_ms1_x6_2 = tf.roll(gt_big_ms1_x6, -4, 2)
            gt_big_ms1_x6_3 = tf.roll(gt_big_ms1_x6, -8, 2)
            gt_big_ms1_x6_4 = tf.roll(gt_big_ms1_x6, -12, 2)
            gt_big_ms1_x6_5 = tf.roll(gt_big_ms1_x6, 4, 2)
            gt_big_ms1_x6_6 = tf.roll(gt_big_ms1_x6, 8, 2)
            gt_big_ms1_x6_7 = tf.roll(gt_big_ms1_x6, 12, 2)
            gt_big_ms1_x6_8 = tf.roll(gt_big_ms1_x6, 16, 2)
            gt_big_ms1_x6_9 = tf.roll(gt_big_ms1_y6, 4, 1)
            gt_big_ms1_x6_10 = tf.roll(gt_big_ms1_y6, 8, 1)
            gt_big_ms1_x6_11 = tf.roll(gt_big_ms1_y6, -4, 1)
            gt_big_ms1_x6_12 = tf.roll(gt_big_ms1_y6, -8, 1)

            gt_16 = tf.concat([gt_big_ms1_x6_2,gt_big_ms1_x6_3,gt_big_ms1_x6_4,gt_big_ms1_x6_5,gt_big_ms1_x6_6,
                               gt_big_ms1_x6_7,gt_big_ms1_x6_8,gt_big_ms1_x6_9,gt_big_ms1_x6_10,gt_big_ms1_x6_11,
                               gt_big_ms1_x6_12],0)

            gt_big_ms1_x7 = tf.roll(ms1_up, -24, 1)
            gt_big_ms1_y7 = tf.roll(ms1_up, -24, 2)
            gt_big_ms1_x7_1 = tf.roll(gt_big_ms1_x7, -24, 2)
            gt_big_ms1_x8 = tf.roll(ms1_up, -32, 1)
            gt_big_ms1_y8 = tf.roll(ms1_up, -32, 2)
            gt_big_ms1_x8_1 = tf.roll(gt_big_ms1_x8, -32, 2)
            gt_big_ms1_x9 = tf.roll(ms1_up, 24, 1)
            gt_big_ms1_y9 = tf.roll(ms1_up, 24, 2)
            gt_big_ms1_x9_1 = tf.roll(gt_big_ms1_x9, 24, 2)

            gt_big_ms1_x10 = tf.roll(ms1_up, 32, 1)
            gt_big_ms1_y10 = tf.roll(ms1_up, 32, 2)
            gt_big_ms1_x10_1 = tf.roll(gt_big_ms1_x10, 32, 2)
            gt_big_ms1_x10_2 = tf.roll(gt_big_ms1_x10, -32, 2)
            gt_big_ms1_x10_3 = tf.roll(gt_big_ms1_y9, -24, 2)
            gt_big_ms1_x10_4 = tf.roll(gt_big_ms1_x7, 24, 2)
            gt_big_ms1_x10_5 = tf.roll(gt_big_ms1_x7, 24, 2)
            gt_big_ms1_x10_6 = tf.roll(gt_big_ms1_x7, 32, 2)
            gt_big_ms1_x10_7 = tf.roll(gt_big_ms1_x8, 32, 2)
            gt_big_ms1_x10_8 = tf.roll(gt_big_ms1_x8, 24, 2)
            gt_big_ms1_x10_9 = tf.roll(gt_big_ms1_x8, -24, 2)

            gt = tf.concat([gt_16,gt_16_1,gt_4,gt_12,gt_big_ms1_x10_2,gt_big_ms1_x10_3,
                            gt_big_ms1_x10_4,gt_big_ms1_x10_5,gt_big_ms1_x10_6,gt_big_ms1_x10_7,
                            gt_big_ms1_x10_8,gt_big_ms1_x10_9],0)

            big_gt_ms = tf.concat(
                [ms1_up,gt_big_ms1_x1, gt_big_ms1_x1_1, gt_big_ms1_x2, gt_big_ms1_x2_1, gt_big_ms1_x3,
                 gt_big_ms1_x3_1, gt_big_ms1_x31, gt_big_ms1_x3_11, gt_big_ms1_y31, gt_big_ms1_x311,
                 gt_big_ms1_y311, gt_big_ms1_x3_111, gt_big_ms1_x4, gt_big_ms1_x4_1, gt_big_ms1_x5, gt_big_ms1_x5_1,
                 gt_big_ms1_x6, gt_big_ms1_x6_1, gt_big_ms1_y1, gt_big_ms1_y2, gt_big_ms1_y3, gt_big_ms1_y4,
                 gt_big_ms1_y5, gt_big_ms1_y6, gt_big_ms1_x7, gt_big_ms1_x7_1, gt_big_ms1_x8, gt_big_ms1_x8_1,
                 gt_big_ms1_x9,gt_big_ms1_x9_1,gt_big_ms1_x10,gt_big_ms1_x10_1,gt_big_ms1_y7, gt_big_ms1_y8,
                 gt_big_ms1_y9, gt_big_ms1_y10, gt], 0)

            output_rgb1 = tf.concat([output_rgb, output_rgb, output_rgb], 0)
            single = output_rgb
            output_rgb = tf.concat(
                [output_rgb1, output_rgb1, output_rgb1, output_rgb1, output_rgb1, output_rgb1, output_rgb], 0)

            output_rgb = tf.concat([output_rgb,output_rgb,output_rgb,output_rgb,output_rgb],0)
            output_rgb = tf.concat([output_rgb,single,single,single],0)

            big_rgb_loss = tf.abs((output_rgb - big_gt_ms))
            big_rgb_loss = (tf.reduce_min(big_rgb_loss,0))

            big_color_loss = tf.reduce_mean(big_rgb_loss)

            # 128 loss function
            # Edge loss
            small_r1, small_g1, small_b1, small_n1 = tf.split(aligned_temp,4,3)
            small_rgb1 = tf.concat([small_r1,small_g1,small_b1],3)

            aligned_grey = tf.reduce_mean(small_rgb1, axis=3, keepdims=True)

            aligned_grey = helperTF.sobelfilter2XY(aligned_grey)
            aligned_grey = tf.abs(aligned_grey)

            small_pan = helperTF.satdwsize(pan,4)
            small_pan = helperTF.sobelfilter2XY(small_pan)
            small_pan = tf.abs(small_pan)

            smalln_edge = tf.reduce_mean(tf.abs(aligned_grey - small_pan))


            # Color loss
            small_r, small_g, small_b, small_n = tf.split(aligned_temp,4,3)
            small_rgb = tf.concat([small_r,small_g,small_b],3)
            small_nir = tf.concat([small_n,small_n,small_n],3)



            # You can implement this loss function, shifted MS image generation with Extract_patch.
            # Implementation can be done very easily.
            # For your understanding, I change the main loss function code for intuitive understanding.
            gt_small_ms2 = helperTF.satdwsize(ms2_up, 4)
            gt_small_ms1 = helperTF.satdwsize(ms1_up, 4)
            gt_small_ms1_x1 = tf.roll(gt_small_ms1, 1, 1)
            gt_small_ms1_x1_y_1 = tf.roll(gt_small_ms1_x1, -1, 2)
            gt_small_ms1_x1_y_2 = tf.roll(gt_small_ms1_x1, 2, 2)
            gt_small_ms1_x1_y_3 = tf.roll(gt_small_ms1_x1, -2, 2)

            gt_small_ms1_y1 = tf.roll(gt_small_ms1, 1, 2)
            gt_small_ms1_x1_1 = tf.roll(gt_small_ms1_x1, 1, 2)
            gt_small_ms1_x2 = tf.roll(gt_small_ms1, 2, 1)
            gt_small_ms1_x2_y_2 = tf.roll(gt_small_ms1_x2, -2, 2)

            gt_small_ms1_y2 = tf.roll(gt_small_ms1, 2, 2)
            gt_small_ms1_x2_1 = tf.roll(gt_small_ms1_x2, 2, 2)
            gt_small_ms1_x2_2 = tf.roll(gt_small_ms1_x2, -2, 2)
            gt_small_ms1_x2_3 = tf.roll(gt_small_ms1_x2, 1, 2)
            gt_small_ms1_x2_4 = tf.roll(gt_small_ms1_x2, -1, 2)

            gt_small_ms1_x3 = tf.roll(gt_small_ms1, 1, 1)
            gt_small_ms1_y3 = tf.roll(gt_small_ms1, 1, 2)
            gt_small_ms1_x3_1 = tf.roll(gt_small_ms1_x3, 1, 2)
            gt_small_ms1_x3_3 = tf.roll(gt_small_ms1_x3, 1, 2)
            gt_small_ms1_x3_4 = tf.roll(gt_small_ms1_x3, -1, 2)
            gt_small_ms1_x3_5 = tf.roll(gt_small_ms1_x3, 2, 2)
            gt_small_ms1_x3_6 = tf.roll(gt_small_ms1_x3, -2, 2)
            gt_small_ms1_x3_7 = tf.roll(gt_small_ms1_x3, 3, 2)
            gt_small_ms1_x3_8 = tf.roll(gt_small_ms1_x3, -3, 2)

            gt_small_ms1_x6_9 = tf.roll(gt_small_ms1_y3, 1, 1)
            gt_small_ms1_x6_10 = tf.roll(gt_small_ms1_y3, 2, 1)
            gt_small_ms1_x6_11 = tf.roll(gt_small_ms1_y3, -1, 1)
            gt_small_ms1_x6_12 = tf.roll(gt_small_ms1_y3, -2, 1)

            gt_small_ms1_x31 = tf.roll(gt_small_ms1, 3, 1)
            gt_small_ms1_y31 = tf.roll(gt_small_ms1, 3, 2)

            gt_small_ms1_x3_11 = tf.roll(gt_small_ms1_x31, 3, 2)
            gt_small_ms1_x3_11_1 = tf.roll(gt_small_ms1_x31, -3, 2)
            gt_small_ms1_x3_11_2 = tf.roll(gt_small_ms1_x31, -2, 2)
            gt_small_ms1_x3_11_3 = tf.roll(gt_small_ms1_x31, 2, 2)
            gt_small_ms1_x3_11_4 = tf.roll(gt_small_ms1_x31, 1, 2)
            gt_small_ms1_x3_11_5 = tf.roll(gt_small_ms1_x31, -1, 2)
            gt_small_ms1_x3_11_6 = tf.roll(gt_small_ms1_y31, -1, 1)
            gt_small_ms1_x3_11_7 = tf.roll(gt_small_ms1_y31, 1, 1)
            gt_small_ms1_x3_11_8 = tf.roll(gt_small_ms1_y31, -2, 1)
            gt_small_ms1_x3_11_9 = tf.roll(gt_small_ms1_y31, 2, 1)

            gt_small_ms1_x311 = tf.roll(gt_small_ms1, -3, 1)
            gt_small_ms1_y311 = tf.roll(gt_small_ms1, -3, 2)
            gt_small_ms1_x3_111 = tf.roll(gt_small_ms1_x311, -3, 2)
            gt_small_ms1_x3_111_1 = tf.roll(gt_small_ms1_x311, 3, 2)
            gt_small_ms1_x3_111_2 = tf.roll(gt_small_ms1_x311, 1, 2)
            gt_small_ms1_x3_111_3 = tf.roll(gt_small_ms1_x311, -1, 2)
            gt_small_ms1_x3_111_4 = tf.roll(gt_small_ms1_x311, 2, 2)
            gt_small_ms1_x3_111_5 = tf.roll(gt_small_ms1_x311, -2, 2)
            gt_small_ms1_x3_111_6 = tf.roll(gt_small_ms1_y311, 1, 1)
            gt_small_ms1_x3_111_7 = tf.roll(gt_small_ms1_y311, -1, 1)
            gt_small_ms1_x3_111_8 = tf.roll(gt_small_ms1_y311, 2, 1)
            gt_small_ms1_x3_111_9 = tf.roll(gt_small_ms1_y311, -2, 1)

            temp = tf.concat([gt_small_ms1_x6_9,gt_small_ms1_x6_10,gt_small_ms1_x6_11,gt_small_ms1_x6_12,gt_small_ms1_x3_11,gt_small_ms1_x3_111],0)
            gt_small_ms1_x4 = tf.roll(gt_small_ms1, -1, 1)
            gt_small_ms1_x4_y_1 = tf.roll(gt_small_ms1_x4, 1, 2)
            gt_small_ms1_x4_y_2 = tf.roll(gt_small_ms1_x4, 2, 2)
            gt_small_ms1_x4_y_3 = tf.roll(gt_small_ms1_x4, -2, 2)


            gt_small_ms1_y4 = tf.roll(gt_small_ms1, -1, 2)
            gt_small_ms1_x4_1 = tf.roll(gt_small_ms1_x4, -1, 2)

            gt_small_ms1_x5 = tf.roll(gt_small_ms1, -2, 1)
            gt_small_ms1_x5_y2 = tf.roll(gt_small_ms1_x5, 2, 2)
            gt_small_ms1_x5_y3 = tf.roll(gt_small_ms1_x5, 1, 2)
            gt_small_ms1_x5_y4 = tf.roll(gt_small_ms1_x5, -1, 2)

            gt_small_ms1_y5 = tf.roll(gt_small_ms1, -2, 2)
            gt_small_ms1_x5_1 = tf.roll(gt_small_ms1_x5, -2, 2)

            gt_small_ms1_x6 = tf.roll(gt_small_ms1, -1, 1)
            gt_small_ms1_y6 = tf.roll(gt_small_ms1, -1, 2)
            gt_small_ms1_x6_1 = tf.roll(gt_small_ms1_x6, -1, 2)
            gt_small_ms1_x6_2 = tf.roll(gt_small_ms1_x6, -1, 2)
            gt_small_ms1_x6_3 = tf.roll(gt_small_ms1_x6, -2, 2)
            gt_small_ms1_x6_4 = tf.roll(gt_small_ms1_x6, -3, 2)
            gt_small_ms1_x6_5 = tf.roll(gt_small_ms1_x6, 1, 2)
            gt_small_ms1_x6_6 = tf.roll(gt_small_ms1_x6, 2, 2)
            gt_small_ms1_x6_7 = tf.roll(gt_small_ms1_x6, 3, 2)
            gt_small_ms1_x6_8 = tf.roll(gt_small_ms1_x6, 1, 2)
            gt_small_ms1_x6_9 = tf.roll(gt_small_ms1_y6, 1, 1)
            gt_small_ms1_x6_10 = tf.roll(gt_small_ms1_y6, 2, 1)
            gt_small_ms1_x6_11 = tf.roll(gt_small_ms1_y6, -1, 1)
            gt_small_ms1_x6_12 = tf.roll(gt_small_ms1_y6, -2, 1)


            gt_small_ms1_x7 = tf.roll(gt_small_ms1, -6, 1)
            gt_small_ms1_y7 = tf.roll(gt_small_ms1, -6, 2)
            gt_small_ms1_x7_1 = tf.roll(gt_small_ms1_x7, -6, 2)
            gt_small_ms1_x8 = tf.roll(gt_small_ms1, -2, 1)
            gt_small_ms1_y8 = tf.roll(gt_small_ms1, -2, 2)
            gt_small_ms1_x8_1 = tf.roll(gt_small_ms1_x8, -2, 2)
            gt_small_ms1_x9 = tf.roll(gt_small_ms1, 6, 1)
            gt_small_ms1_y9 = tf.roll(gt_small_ms1, 6, 2)
            gt_small_ms1_x9_1 = tf.roll(gt_small_ms1_x9, 6, 2)

            gt_small_ms1_x10 = tf.roll(gt_small_ms1, 2, 1)
            gt_small_ms1_y10 = tf.roll(gt_small_ms1, 2, 2)
            gt_small_ms1_x10_1 = tf.roll(gt_small_ms1_x10, 2, 2)
            gt_small_ms1_x10_2 = tf.roll(gt_small_ms1_x10, -2, 2)
            gt_small_ms1_x10_3 = tf.roll(gt_small_ms1_y9, -6, 2)
            gt_small_ms1_x10_4 = tf.roll(gt_small_ms1_x7, 6, 2)
            gt_small_ms1_x10_5 = tf.roll(gt_small_ms1_x7, 6, 2)
            gt_small_ms1_x10_6 = tf.roll(gt_small_ms1_x7, 2, 2)
            gt_small_ms1_x10_7 = tf.roll(gt_small_ms1_x8, 2, 2)
            gt_small_ms1_x10_8 = tf.roll(gt_small_ms1_x8, 6, 2)
            gt_small_ms1_x10_9 = tf.roll(gt_small_ms1_x8, -6, 2)
            gt_ms = tf.concat([gt_small_ms1,gt_small_ms1_x1,gt_small_ms1_x1_1,gt_small_ms1_x2,gt_small_ms1_x2_1,gt_small_ms1_x3,gt_small_ms1_x3_1,gt_small_ms1_x4,gt_small_ms1_x4_1,gt_small_ms1_x5,gt_small_ms1_x5_1,gt_small_ms1_x6,gt_small_ms1_x6_1,gt_small_ms1_y1,gt_small_ms1_y2,gt_small_ms1_y3,gt_small_ms1_y4,gt_small_ms1_y5,gt_small_ms1_y6],0)
            gt_ms = tf.concat([gt_ms,gt_small_ms1_x7,gt_small_ms1_x7_1,gt_small_ms1_x8,gt_small_ms1_x8_1,gt_small_ms1_x9,gt_small_ms1_x9_1,gt_small_ms1_x10,gt_small_ms1_x10_1,gt_small_ms1_y7,gt_small_ms1_y8,gt_small_ms1_y9,gt_small_ms1_y10],0)

            gt_16 = tf.concat(
                [gt_small_ms1_x6_2, gt_small_ms1_x6_3, gt_small_ms1_x6_4, gt_small_ms1_x6_5, gt_small_ms1_x6_6,
                 gt_small_ms1_x6_7, gt_small_ms1_x6_8, gt_small_ms1_x6_9, gt_small_ms1_x6_10, gt_small_ms1_x6_11,
                 gt_small_ms1_x6_12], 0)
            gt_4 = tf.concat(
                [gt_small_ms1_x5_y2, gt_small_ms1_x5_y3, gt_small_ms1_x5_y4, gt_small_ms1_x4_y_1, gt_small_ms1_x4_y_2,
                 gt_small_ms1_x4_y_3], 0)
            gt_12 = tf.concat(
                [gt_small_ms1_x3_111_1, gt_small_ms1_x3_111_2, gt_small_ms1_x3_111_3, gt_small_ms1_x3_111_4,
                 gt_small_ms1_x3_111_5,
                 gt_small_ms1_x3_111_6, gt_small_ms1_x3_111_7, gt_small_ms1_x3_111_8, gt_small_ms1_x3_111_9,
                 gt_small_ms1_x3_11_1,
                 gt_small_ms1_x3_11_2, gt_small_ms1_x3_11_3, gt_small_ms1_x3_11_4, gt_small_ms1_x3_11_5,
                 gt_small_ms1_x3_11_6,
                 gt_small_ms1_x3_11_7, gt_small_ms1_x3_11_8, gt_small_ms1_x3_11_9], 0)
            gt_16_1 = tf.concat([gt_small_ms1_x1_y_1, gt_small_ms1_x1_y_2, gt_small_ms1_x1_y_3, gt_small_ms1_x2_y_2,
                                 gt_small_ms1_x2_2, gt_small_ms1_x2_3, gt_small_ms1_x2_4, gt_small_ms1_x3_3,
                                 gt_small_ms1_x3_4,
                                 gt_small_ms1_x3_5, gt_small_ms1_x3_6, gt_small_ms1_x3_7, gt_small_ms1_x3_8,
                                 gt_small_ms1_x3_3,
                                 gt_small_ms1_x6_9, gt_small_ms1_x6_10, gt_small_ms1_x6_11, gt_small_ms1_x6_12], 0)
            gt_ms = tf.concat([gt_ms, gt_16, gt_16_1, gt_4, gt_12, gt_small_ms1_x10_2, gt_small_ms1_x10_3,
                            gt_small_ms1_x10_4, gt_small_ms1_x10_5, gt_small_ms1_x10_6, gt_small_ms1_x10_7,
                            gt_small_ms1_x10_8, gt_small_ms1_x10_9,temp], 0)


            small_rgb_1 = tf.concat([small_rgb,small_rgb,small_rgb],0)
            small_rgb_1 = tf.concat([small_rgb_1,small_rgb_1,small_rgb_1,small_rgb_1,small_rgb_1,small_rgb_1,small_rgb],0)
            small_rgb_1 = tf.concat([small_rgb_1,small_rgb,small_rgb,small_rgb,small_rgb,small_rgb,small_rgb,small_rgb,small_rgb,small_rgb,small_rgb,small_rgb,small_rgb],0)
            small_rgb_1 = tf.concat([small_rgb_1,small_rgb_1,small_rgb_1,small_rgb,small_rgb,small_rgb,small_rgb,small_rgb],0)
            small_rgb_loss = tf.abs((small_rgb_1-gt_ms))

            small_color_loss = tf.reduce_min(tf.abs(small_rgb_loss),0)
            small_color_loss = tf.reduce_mean((tf.abs(small_color_loss)))

            loss_G = lossE_og * 10  + big_color_loss * 5
            small_loss = smalln_edge * 10 + small_color_loss * 5
            loss_G = tf.reduce_mean(loss_G)*4 + small_loss

        ch_3_output1, ch_3_output2, ch_3_output3,ch_3_output4 = tf.split(output,4,3)
        ch_3_output = tf.concat([ch_3_output1,ch_3_output2,ch_3_output3],3)


        output = ch_3_output*Ms + Mm  # de-normalization

        aligned_temp1, aligned_temp2, aligned_temp3,aligned_temp4 = tf.split(aligned_temp,4,3)
        aligned_temp = tf.concat([aligned_temp1,aligned_temp2,aligned_temp3],3)

        aligned_temp = resize_blur(aligned_temp,4)
        aligned_temp = aligned_temp*Ms + Mm

        if not isTraining:
            loss_G = helperTF.QNR(output, pan, ms, getAll=True)

        return output, median_temp, aligned_temp, loss_G, loss_G*1e3

    name_op = 'op_net'
    op_net = tf.make_template(name_op, op_net0)

    output, median_temp, aligned_temp, loss_G, tTrainLoss = op_net(ms_bat, pan_bat, ms2_bat, mask_bat)  # train network
    output_test, out_median_temp, out_4ch_test, _, _ = op_net(ms_test, pan_test,ms2_test, mask_test, False)  # test network
    output_test = tf.clip_by_value(output_test, 0, 1)
    out_4ch_test = tf.clip_by_value(out_4ch_test, 0, 1)

    # Calculate metrics
    small_output = helperTF.satdwsize(output_test, SCALE)

    align_X_metric = [helperTF.metric_ergas(small_output, ms_test, SCALE),
                helperTF.metric_scc(output_test, pan_test),
                helperTF.qindex(helperTF.satdwsize(output_test, SCALE), ms_test, keepdims0=False),
                helperTF.QNR(output_test, pan_test, ms_test)]

    small_ms2_test = helperTF.satdwsize(ms_test, SCALE)
    align_O_metric = [helperTF.metric_ergas(helperTF.satdwsize(output_test, SCALE), ms_test, SCALE),
                helperTF.metric_scc2(out_4ch_test, pan_test),
                helperTF.qindex(helperTF.satdwsize(output_test, SCALE), ms_test, keepdims0=False),
                helperTF.QNR(output_test, pan_test, ms_test)]

    # optimizer start
    G_tvars = [var for var in tf.trainable_variables() if var.name.startswith(name_op + '/' + 'G')]
    MyAdamW = tf.contrib.opt.extend_with_decoupled_weight_decay(tf.train.AdamOptimizer)
    G_optim = MyAdamW(weight_decay=wd, learning_rate=learning_rate)
    G_grads_and_vars = G_optim.compute_gradients(loss_G, var_list=G_tvars)
    train_op = G_optim.apply_gradients(G_grads_and_vars, global_step=global_step)
    # optimizer end

    var_shapes = [np.prod(var.shape.as_list()) for var in G_tvars]
    var_shapes = np.sum(var_shapes)
    save.save_log('Total_parameters: ' + str(var_shapes))  # total number of parameters

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=EPOCH_TOTAL)
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(init)  # Run variable initialization
        for epcIndex in range(1, EPOCH_TOTAL + 1):
            g_step, train_loss_tot = 0, 0

            start_time = time.time()
            for iterIndex in range(int(ITER_PER_EPOCH)):
                _, g_step, train_loss = sess.run([train_op, global_step, tTrainLoss])  # run the session
                train_loss_tot += train_loss

            time_train = time.time() - start_time
            train_loss_tot = train_loss_tot / ITER_PER_EPOCH

            # Run the test
            test_loss_tot = 0
            start_time = time.time()

            if epcIndex % TEST_STEP == 0 or epcIndex == 1:
                testLoss_tot = 0
                testLoss2_tot = 0
                metricsX_tot = np.zeros((4,), np.float32)
                metricsO_tot = np.zeros((4,), np.float32)
                metrics_N = 0
                for testIndex in range(len(list_test_PAN)):
                    # Load test files
                    imname = os.path.basename(list_test_PAN[testIndex])

                    im_ms = tifffile.imread(list_test_MS[testIndex])
                    im_pan = tifffile.imread(list_test_PAN[testIndex])
                    im_pan = im_pan[:, :, np.newaxis]
                    im_ms2 = tifffile.imread(list_test_MS[testIndex])

                    im_ms = helper.cropn(im_ms, SCALE)
                    im_pan = helper.cropn(im_pan, SCALE * SCALE)
                    im_ms2 = helper.cropn(im_ms2, SCALE)

                    im_ms = im_ms.astype(np.float32)
                    im_pan = im_pan.astype(np.float32)
                    im_ms2 = im_ms2.astype(np.float32)

                    im_ms = np.array(im_ms) / IMG_DIV
                    im_pan = np.array(im_pan) / IMG_DIV
                    im_ms2 = np.array(im_ms2) / IMG_DIV

                    im_ms_net = im_ms[np.newaxis, :, :, :]
                    im_pan_net = im_pan[np.newaxis, :, :, :]
                    im_ms2_net = im_ms2[np.newaxis, :, :, :]

                    # Run test
                    im_ps, aligned_test, metricsX, metricsO = sess.run([output_test, out_4ch_test,  align_X_metric, align_O_metric], feed_dict={ms_test: im_ms_net, pan_test: im_pan_net, ms2_test: im_ms2_net})

                    metricsX_tot += metricsX
                    metricsO_tot += metricsO
                    metrics_N += 1.

                    def postProcImg(x):
                        y = np.squeeze(x)
                        if np.size(y.shape) == 3:
                            y = y[:, :, :3]
                        y = helper.im_percent_norm(y, y)
                        y = np.clip(y, 0, 1) * 255.0
                        y = np.uint8(y)

                        return y

                    im_ps = postProcImg(im_ps)
                    aligned_test = postProcImg(aligned_test)

                    # imh: MS, imb: PAN, imnn: nearest-neighbor interpolated MS, ims: network output (ours)
                    if epcIndex == 1:
                        im_ms = postProcImg(im_ms)
                        im_pan = postProcImg(im_pan)

                        im_msnn = helper.imresizenn(im_ms, SCALE)
                        im_msnn = np.uint8(im_msnn)

                        Image.fromarray(im_ms).save(os.path.join(IMG_SAVE_DIR_PATH, imname + '_00_mslr_' + timestamp + '.png'))
                        Image.fromarray(im_pan).save(os.path.join(IMG_SAVE_DIR_PATH, imname + '_01_pan_' + timestamp + '.png'))
                        Image.fromarray(im_msnn).save(os.path.join(IMG_SAVE_DIR_PATH, imname + '_02_msnn_' + timestamp + '.png'))
                    try:
                        Image.fromarray(im_ps).save(os.path.join(IMG_SAVE_DIR_PATH, imname + '_03_ours_' + timestamp + '.png'))
                        Image.fromarray(aligned_test).save(os.path.join(IMG_SAVE_DIR_PATH, imname + '_02_msnn_aligned' + timestamp + '.png'))

                    except:
                        Image.fromarray(im_ps).save(
                            os.path.join(IMG_TEMP_SAVE_DIR_PATH, 'Epc' + str(epcIndex) + '_' + imname + '_03_ours' + timestamp + '.png'))

                metricsX_tot = metricsX_tot / metrics_N  # average metric value
                metricsO_tot = metricsO_tot / metrics_N  # average metric value
                metricX_print = ("[Align X] AVG ERGAS: %.3f\t SCC: %.3f\t Q: %.3f\t QNR: %.3f #####" % (metricsX_tot[0], metricsX_tot[1], metricsX_tot[2], metricsX_tot[3]))
                metricO_print = ("[4ch SCC] AVG ERGAS: %.3f\t SCC: %.3f\t Q: %.3f\t QNR: %.3f #####" % (metricsO_tot[0], metricsO_tot[1], metricsO_tot[2], metricsO_tot[3]))

                test_loss_tot = test_loss_tot/len(list_test_PAN)

            time_test = time.time() - start_time

            save.save_log('epcIndex {:d} |iter {:d} |train {:0.3f} |time_train {:0.3f} |time_test {:0.3f} |time_tot {:0.3f}'.format(
                epcIndex, g_step, train_loss_tot, (time_train / 60), (time_test / 60), ((time_train + time_test) / 60)))
            save.save_log(metricX_print)
            save.save_log(metricO_print)

            # Save model
            saver.save(sess, PARAM_PATH + '/model-' + str(int(ITER_TOTAL)), write_meta_graph=False,
                       write_state=False)  # save parameter
