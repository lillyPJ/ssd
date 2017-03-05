import numpy as np
import matplotlib
matplotlib.use('Agg')
import os
import xml.dom.minidom
# %matplotlib inline
from makeNewDir import *
import matplotlib.pyplot as plt
# Make sure that caffe is on the python path:
#caffe_root = 'your_caffe_root/'  # this file is expected to be in {caffe_root}/examples
caffe_root = '/home/lili/codes/ssd/caffe-ssd/'
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

# from google.protobuf import text_format
# from caffe.proto import caffe_pb2


# load PASCAL VOC labels
# labelmap_file = '/home/lili/codes/textBoxes_XiangBai/TextBoxes/examples/TextBoxes/labelmap_voc.prototxt'
# file = open(labelmap_file, 'r')
# labelmap = caffe_pb2.LabelMap()
# text_format.Merge(str(file.read()), labelmap)

#
# def get_labelname(labelmap, labels):
#     num_labels = len(labelmap.item)
#     labelnames = []
#     if type(labels) is not list:
#         labels = [labels]
#     for label in labels:
#         found = False
#         for i in xrange(0, num_labels):
#             if label == labelmap.item[i].label:
#                 found = True
#                 labelnames.append(labelmap.item[i].display_name)
#                 break
#         assert found == True
#     return labelnames


#model_def = 'your_caffe_root/examples/TextBoxes/deploy.prototxt'
#model_weights = 'your_caffe_root/examples/TextBoxes/TextBoxes_icdar13.caffemodel'

model_def = caffe_root + 'models/VGGNet/cocoICDAR13SCUT/SSD_300x300/deploy.prototxt'
model_weights = caffe_root + 'models/VGGNet/cocoICDAR13SCUT/SSD_300x300/VGG_cocoICDAR13SCUT_SSD_300x300_iter_65000.caffemodel'

model_def = '/dataL/Codes/ssd/models/ssd_cocoICDAR13SCUT/2/512*512_new/deploy.prototxt'
model_weights = '/dataL/Codes/ssd/models/ssd_cocoICDAR13SCUT/2/512*512_new/VGG_cocoICDAR13SCUT_SSD_512x512_iter_100000.caffemodel'

#-----512_new4
model_def = '/dataL/Codes/ssd/models/ssd_cocoICDAR13SCUT/3/512_new4/SSD_512x512/deploy.prototxt'
model_weights = '/dataL/Codes/ssd/models/ssd_cocoICDAR13SCUT/3/512_new4/SSD_512x512/VGG_cocoICDAR13SCUT_SSD_512x512_iter_110000.caffemodel'



# scales=((300,300),(700,700),(700,500),(700,300),(1600,1600))
#scales=((512, 512), (1200,1200))
scales=((512,512),(700,700),(700,500),(1200,500))

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
print(net.blobs['data'].data.shape)

#test_list=open('icdar_2013_dataset_root/test_list.txt')
#save_dir='your_caffe_root/data/TextBoxes/test_bb_multi_scale/'
icdar_2013_dataset_root = '/home/lili/datasets/VOC/VOCdevkit/ICDAR2013/'
test_list=open(icdar_2013_dataset_root + 'ImageSets/Main/test.txt')
save_dir=caffe_root + 'data/TextBoxes/test_bb_multi_scale/'
save_fig_dir = caffe_root + 'data/TextBoxes/fig_multi_scale/'
makedirs(save_dir)
makedirs(save_fig_dir)

for i, line in enumerate(test_list.readlines()):
	line=line.strip()
	image_name=line
	#image_path='icdar_2013_dataset_root/test_images/'+line
	#save_detection_path=save_dir+line[0:len(line)-4]+'.txt'
	image_path = icdar_2013_dataset_root + 'JPEGImages/' + line + '.jpg'
	save_detection_path = save_dir + 'res_' + line + '.txt'

	print(image_path)
	print '{}:{}'.format(i, image_name)
	image=caffe.io.load_image(image_path)
	image_height,image_width,channels=image.shape
	# print(max(image_height,image_width))
	plt.clf()

	# imageNew = sci.imresize(image, scale)
	plt.imshow(image)
	currentAxis = plt.gca()

	detection_result=open(save_detection_path,'wt')
	for scale in scales:
		image_resize_height = scale[0]
		image_resize_width = scale[1]
		transformer = caffe.io.Transformer({'data': (1,3,image_resize_height,image_resize_width)})
		transformer.set_transpose('data', (2, 0, 1))
		transformer.set_mean('data', np.array([104,117,123])) # mean pixel
		transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
		transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
		
		net.blobs['data'].reshape(1,3,image_resize_height,image_resize_width)		
		transformed_image = transformer.preprocess('data', image)
		net.blobs['data'].data[...] = transformed_image

		# Forward pass.
		detections = net.forward()['detection_out']

		# Parse the outputs.
		det_label = detections[0,0,:,1]
		det_conf = detections[0,0,:,2]
		det_xmin = detections[0,0,:,3]
		det_ymin = detections[0,0,:,4]
		det_xmax = detections[0,0,:,5]
		det_ymax = detections[0,0,:,6]

		# Get detections with confidence higher than 0.1.
		top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.1]

		top_conf = det_conf[top_indices]
		#top_label_indices = det_label[top_indices].tolist()
		#top_labels = get_labelname(labelmap, top_label_indices)
		top_xmin = det_xmin[top_indices]
		top_ymin = det_ymin[top_indices]
		top_xmax = det_xmax[top_indices]
		top_ymax = det_ymax[top_indices]

		for i in xrange(top_conf.shape[0]):
			xmin = int(round(top_xmin[i] * image.shape[1]))
			ymin = int(round(top_ymin[i] * image.shape[0]))
			xmax = int(round(top_xmax[i] * image.shape[1]))
			ymax = int(round(top_ymax[i] * image.shape[0]))
			xmin = max(1, xmin)
			ymin = max(1, ymin)
			xmax = min(image.shape[1]-1, xmax)
			ymax = min(image.shape[0]-1, ymax)
			score = top_conf[i]
			result=str(xmin)+' '+str(ymin)+' '+str(xmax)+' '+str(ymin)+' '+str(xmax)+' '+str(ymax)+' '+str(xmin)+' '+str(ymax)+' '+str(score)+'\n'
			detection_result.write(result)

			name = '%.2f' % (score)
			coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
			color = 'b'
			currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
			currentAxis.text(xmin, ymin, name, bbox={'facecolor': 'white', 'alpha': 0.5})

	detection_result.close()
	plt.savefig(save_fig_dir + line)
test_list.close()
print('success')

