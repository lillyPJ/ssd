import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *
import os
import xml.dom.minidom
# %matplotlib inline
from makeNewDir import *

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0, imgName='conv1'):
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data)
    plt.show()
    plt.savefig(imgName)


# Make sure that caffe is on the python path:
# caffe_root = 'your_caffe_root/'  # this file is expected to be in {caffe_root}/examples
caffe_root = '/home/siky/lili/TextBoxes/'
os.chdir(caffe_root)
import sys

sys.path.insert(0, 'python')

import caffe

caffe.set_device(0)
caffe.set_mode_gpu()

model_def = caffe_root + 'models/VGGNet/VGG/longer_conv_300x300/deploy.prototxt'
model_weights = caffe_root + 'models/VGGNet/VGG/longer_conv_300x300/VGG_VGG_longer_conv_300x300_iter_2000.caffemodel'

model_def = caffe_root + 'models/VGGNet/VGG/pre-train/deploy.prototxt'
#model_weights = caffe_root + 'models/VGGNet/VGG/pre-train/VGG_VGG_longer_conv_300x300_iter_50000.caffemodel'

model_def = caffe_root + 'examples/TextBoxes/deploy.prototxt'
#model_weights = caffe_root + 'examples/TextBoxes/TextBoxes_icdar13.caffemodel'

# model_weights = caffe_root + 'models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel'
# model_def = caffe_root + 'models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced_deploy.prototxt'


scales = ((700, 700),)
# IMPORTANT: If use mutliple scales in the paper, you need an extra non-maximum superession for the results
# scales=((300,300),(700,700),(700,500),(700,300),(1600,1600))


net = caffe.Net(model_def,  # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)  # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
print(net.blobs['data'].data.shape)
# dataset_name = 'chineseWHZ'
# img_root = '/home/lili/datasets/taggerwendang/raw/img'
dataset_name = 'ICDAR2013'
img_root = '/home/siky/datasets/ICDAR2013/img/test/'
test_list = [os.path.join(img_root, f) for f in os.listdir(img_root) if f.endswith('.jpg')]

save_bbs_dir = caffe_root + 'data/' + dataset_name + '/test_bb/'
save_fig_dir = caffe_root + 'data/' + dataset_name + '/vis_fig/'
makedirs(save_bbs_dir)
makedirs(save_fig_dir)
# for line in test_list:
line = test_list[219]
line = line.strip()
image_name = '/home/siky/datasets/ICDAR2013/img/test/img_1.jpg'
line = image_name
line = os.path.basename(line)
print image_name

save_detection_path = save_bbs_dir + 'res_' + line[:-3] + 'txt'

image = caffe.io.load_image(image_name)
image_height, image_width, channels = image.shape
detection_result = open(save_detection_path, 'wt')
# replace paras
model_weights2 = caffe_root + 'examples/TextBoxes/TextBoxes_icdar13.caffemodel'
net2 = caffe.Net(model_def,  # defines the structure of the model
                 model_weights2,  # contains the trained weights
                 caffe.TEST)  # use test mode (e.g., don't perform dropout)

# filter_size = [(k, v[0].data.shape) for k, v in net.params.items()]
# for each in filter_size:
#     layer = net.params[each[0]][0].data
#     print '{}:{}'.format(each[0], layer.shape)
#     if(layer.ndim > 1):
#         replaceLayer = each[0]
#         net.params[replaceLayer][0] = net2.params[replaceLayer][0]
# lays = ['conv3_1', 'conv3_2', 'conv3_3','conv4_1','conv4_2', 'conv5_1'] #not much change
# #lays = ['conv4_3', 'conv5_2', 'conv5_3'] # change a lot
#
# # for replaceLayer in lays:
# #     net.params[replaceLayer][0] = net2.params[replaceLayer][0]
#
# lays = ['conv4_3_norm', 'fc7', 'pool6', 'conv6_2'] # no change
# #lays = ['conv7_2', 'conv8_2',] # change a lot
#
# for replaceLayer in lays:
#     net.params[replaceLayer + '_mbox_loc'][0] = net2.params[replaceLayer + '_mbox_loc'][0]
#     net.params[replaceLayer + '_mbox_conf'][0] = net2.params[replaceLayer + '_mbox_conf'][0]


for scale in scales:
    image_resize_height = scale[0]
    image_resize_width = scale[1]
    transformer = caffe.io.Transformer({'data': (1, 3, image_resize_height, image_resize_width)})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

    net.blobs['data'].reshape(1, 3, image_resize_height, image_resize_width)
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image




    # Forward pass.
    result = net.forward()
    detections = net.forward()['detection_out']



    # plot features and filters
    # # 1. filter size
    # filter_size = [(k, v[0].data.shape) for k, v in net.params.items()]
    # print filter_size
    # for ii, each in enumerate(filter_size):
    #     layer = net.params[each[0]][0].data
    #     print '{}:{}'.format(each[0], layer.shape)
    #     if(layer.ndim > 1):
    #         vis_square(layer[0, :36], imgName= 'fig_filters/{}_{}.png'.format(ii, each[0]))
    #
    # # 2. feature map
    # feature_map = [(k, v.data.shape) for k, v in net.blobs.items()]
    # print feature_map
    # for ii, each in enumerate(filter_size):
    #     layer = net.blobs[each[0]].data
    #     print '{}:{}'.format(each[0], layer.shape)
    #     if (layer.ndim > 3):
    #         vis_square(layer[0, :16], imgName= 'fig_blobs/{}_{}.png'.format(ii, each[0]))




    det_label = detections[0, 0, :, 1]
    det_conf = detections[0, 0, :, 2]
    det_xmin = detections[0, 0, :, 3]
    det_ymin = detections[0, 0, :, 4]
    det_xmax = detections[0, 0, :, 5]
    det_ymax = detections[0, 0, :, 6]

    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

    top_conf = det_conf[top_indices]
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    plt.clf()
    plt.imshow(image)
    currentAxis = plt.gca()

    for i in xrange(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * image.shape[1]))
        ymin = int(round(top_ymin[i] * image.shape[0]))
        xmax = int(round(top_xmax[i] * image.shape[1]))
        ymax = int(round(top_ymax[i] * image.shape[0]))
        xmin = max(1, xmin)
        ymin = max(1, ymin)
        xmax = min(image.shape[1] - 1, xmax)
        ymax = min(image.shape[0] - 1, ymax)
        score = top_conf[i]
        result = str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + '\r\n'
        detection_result.write(result)

        name = '%.2f' % (score)
        coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
        color = 'b'
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(xmin, ymin, name, bbox={'facecolor': 'white', 'alpha': 0.5})

detection_result.close()

plt.savefig(save_fig_dir + line)
# test_list.close()
print('success')
