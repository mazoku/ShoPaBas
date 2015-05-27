__author__ = 'tomas'

import skimage.io as skiio
import skimage.transform as skitra

import numpy as np

import cv2

import shopabas as spb

# dcmdir = '/home/tomas/Dropbox/Work/Data/medical/org-53596059-export_liver.pklz'
# dcmdir = '/home/tomas/Dropbox/Work/Data/medical/org-53009707-export_liver.pklz'
# dcmdir = '/home/tomas/Dropbox/Work/Data/medical/org-52496602-export_liver.pklz'
# dcmdir = '/home/tomas/Dropbox/Work/Data/medical/org-38289898-export1.pklz'
# data = misc.obj_from_file(dcmdir, filetype='pickle')
#
# data3d = data['data3d']
# data3d = tools.windowing(data3d, level=50+(-1000 - data3d.min()), width=300, sliceId=0)
# segmentation = data['segmentation'].astype(np.uint8)
#
# # downscaling
# scale = 0.5
# data3d = tools.resize3D(data3d, scale, sliceId=0).astype(np.uint8)
# segmentation = tools.resize3D(segmentation, scale, sliceId=0).astype(np.uint8)
#
# # rescaling / zooming
# voxel_size = data['voxelsize_mm']
# spacing = voxel_size / 1
#
# print 'spacing for zooming: ', spacing
# print 'shape before zooming: ', data3d.shape
#
# # py3DSeedEditor.py3DSeedEditor(data3d).show()
# data3d = scindiint.zoom(data3d.astype(np.float), spacing)
# segmentation = scindiint.zoom(segmentation, spacing)
#
# print 'shape after zooming: ', data3d.shape
#
# # py3DSeedEditor.py3DSeedEditor(data3d).show()
#
# np.save('input_orig_data.npy', data3d)

# slice = 46
# slice = 12
# slice = 33

scale = 0.2
set = 'man_with_hat'
fname = '/home/tomas/Dropbox/images/Berkeley_Benchmark/set/%s/original.jpg' % set

im_orig = skiio.imread(fname, as_grey=True)
# im = cv2.resize(im_orig, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
im_res = skitra.rescale(im_orig, scale)

# data = im_orig
data = im_res
# using_suppxls = True

f = 5  # factor of maximal distance - max distance in uint8 image (0-255)
# if issubclass(data.dtype.type, np.integer):
if data.dtype == np.int:
    max_d = f  # factor recalculated to integer image type
# elif issubclass(data.dtype.type, np.float):
elif data.dtype == np.float:
    max_d = f * 1./255  # factor recalculated to float image type
# max_d = 10
# method_type = 'interactive'
# method_type = 'automatic'
spb.run(data, params='config.ini')