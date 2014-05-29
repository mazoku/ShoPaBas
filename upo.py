# SOMETHING LIKE UNIDENTIFIED PYTHON OBJECT
# ONLY FOR TESTING, SCRIBBLING ETC.

__author__ = 'tomas'


import numpy as np
import py3DSeedEditor
import matplotlib.pyplot as plt
import skimage.segmentation as skiseg
import tools
import shopabas_suppxls_siblings as sss
import scipy.ndimage.morphology as scindimor


#------------------------------------------------------------------------------------------------
# 33 ... hypodense
# 41 ... small tumor
# 138 ... hyperdense

# slice_idx = 41
#
# labels = np.load('label_im.npy')
# data = np.load('input_data.npy')
# o_data = np.load('input_orig_data.npy')
# mask = np.load('mask.npy')
#
# # py3DSeedEditor.py3DSeedEditor(labels).show()
# data_s = data[slice_idx, :, :]
# o_data_s = o_data[slice_idx, :, :]
# labels_s = labels[slice_idx, :, :]
# mask_s = mask[slice_idx, :, :]
#
# # plt.figure()
# # plt.imshow(data_s, 'gray')
# # plt.show()
#
# if slice_idx == 33:
#     data_det = data[slice_idx, 50:70, 10:30]
#     o_data_det = o_data[slice_idx, 50:70, 10:30]
#     labels_det = labels[slice_idx, 50:70, 10:30]
# elif slice_idx == 41:
#     data_det = data[slice_idx, 45:60, 35:50]
#     o_data_det = o_data[slice_idx, 45:60, 35:50]
#     labels_det = labels[slice_idx, 45:60, 35:50]
# elif slice_idx == 138:
#     data_det = data[slice_idx, 30:60, 35:65]
#     o_data_det = o_data[slice_idx, 30:60, 35:65]
#     labels_det = labels[slice_idx, 30:60, 35:65]
#
# data_bbox, _ = tools.crop_to_bbox(data_s, mask_s)
# o_data_bbox, _ = tools.crop_to_bbox(o_data_s, mask_s)
# labels_bbox, _ = tools.crop_to_bbox(labels_s, mask_s)
#
# tum = labels_det == 3
# tum = scindimor.binary_fill_holes(tum)
# labels_det = np.where(tum, 3, labels_det)
#
# # plt.figure()
# # # plt.subplot(121), plt.imshow(data_bbox, 'gray', interpolation='nearest')
# # plt.subplot(121), plt.imshow(o_data_bbox, 'gray', interpolation='nearest')
# # # plt.subplot(132), plt.imshow(data_bbox, 'gray', interpolation='nearest')
# # plt.subplot(122), plt.imshow(labels_bbox, interpolation='nearest')
# #
# # plt.figure()
# # plt.subplot(121), plt.imshow(o_data_det, 'gray'), plt.title('original')
# # # plt.subplot(132), plt.imshow(data_det, 'gray', interpolation='nearest')
# # plt.subplot(122), plt.imshow(labels_det, interpolation='nearest'), plt.title('segmentation')
#
# plt.figure()
# plt.imshow(o_data_s, 'gray'), plt.title('input')
# plt.figure()
# plt.imshow(o_data_det, 'gray'), plt.title('detail')
# plt.figure()
# plt.imshow(labels_det, interpolation='nearest'), plt.title('segmentation')
# plt.show()


#------------------------------------------------------------------------------------------------
# labels = np.load('label_im.npy')
# n_labels = labels.max() + 1
# print 'Calculating object features...'
# feats = sss.get_features(labels, n_labels)
#
# for i in range(feats.shape[0]):
#     print i, ') ', feats[i, :]
#
# for i in range(n_labels):
#     liver = labels >= 0
#     liver = np.where(labels == i, 2, liver)
#     py3DSeedEditor.py3DSeedEditor(liver).show()

#-------------------------------
import cv2
import matplotlib.pyplot as plt

import numpy as np
import skimage.morphology as skimor
import skimage.filter as skifil
import pygco

# importing data ---------------------------------------------
# 33 ... hypodense
# 41 ... small tumor
# 138 ... hyperdense

slice_idx = 33

# labels = np.load('label_im.npy')
data = np.load('input_data.npy')
o_data = np.load('input_orig_data.npy')
mask = np.load('mask.npy')

data_s = data[slice_idx, :, :]
o_data_s = o_data[slice_idx, :, :]
mask_s = mask[slice_idx, :, :]

data_bbox, _ = tools.crop_to_bbox(data_s, mask_s)
o_data_bbox, _ = tools.crop_to_bbox(o_data_s, mask_s)
mask_bbox, _ = tools.crop_to_bbox(mask_s, mask_s)


plt.figure()
plt.subplot(131), plt.imshow(data_bbox, 'gray')
plt.subplot(132), plt.imshow(o_data_bbox, 'gray')
plt.subplot(133), plt.imshow(mask_bbox, 'gray')
plt.show()
#-------------------------------------------------------------


im = cv2.imread('/home/tomas/Dropbox/images/medicine/hypodense_bad2.png', 0).astype(np.float)

sigma = 100
ims = skifil.gaussian_filter(im, sigma)

#imd = np.absolute(im - ims)
imd = im - ims

imd = np.where(imd < 0, 0, imd)


plt.figure()
plt.subplot(131), plt.imshow(im, 'gray', vmin=0, vmax=255)
# plt.subplot(132), plt.imshow(imd, 'gray', vmin=0)
plt.subplot(132), plt.imshow(imd, 'gray')
plt.subplot(133), plt.imshow(ims, 'gray', vmin=0, vmax=255)

# plt.figure()
# plt.imshow(ims, 'gray')
plt.show()