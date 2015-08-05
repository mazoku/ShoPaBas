__author__ = 'tomas'

import os

import numpy as np
import scipy.io as scio

from skimage import measure, segmentation
import skimage.filters as skifil
import skimage.morphology as skimor
import skimage.transform as skitra
from skimage import img_as_float
import skimage.io as skiio
import skimage.exposure as skiexp
import skimage.segmentation as skiseg

import io3d

import myFigure
import graph_tools as gt
import py3DSeedEditor

import cv2
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
import Tkinter as tk
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import ConfigParser

from myFigure import *

import tools
import json

DATA_DICOM = 0
DATA_IMG = 1

class ShoPaBas:

    def __init__(self, params='config.cfg', data=None, data_type=DATA_IMG, fname=None, mask=None, slice=None):
        self.data_orig = None  # input data in original form
        self.data = None  # working data represents data after smoothing etc.
        self.mask_orig = None  # input mask in original form
        self.mask = None  # working mask represents mask after resizing, bounding boxing etc.
        self.max_d = 0  # maximal allowed distance of a point from the seed in a basin

        self.seeds = None  # list of seed points

        # --  preparing parameters -----
        if isinstance(params, str):
            self.params = load_parameters(params)

        # -- loading data -----
        print 'loading data ...',
        if data is None and fname is not None:
            if data_type == DATA_IMG:
                self.data_orig = skiio.imread(fname, as_grey=True)
            elif data_type == DATA_DICOM:
                dcm_data = io3d.datareader.read(fname)
                self.data_orig = dcm_data[0]
                segmentation = dcm_data[1]['segmentation']
                if segmentation.any():
                    mask = segmentation
        elif data is not None:
            self.data_orig = data
        else:
            raise IOError('No data or filename were specified.')
        print 'ok'

        if issubclass(self.data_orig.dtype.type, np.int):
            self.max_d = self.params['general']['max_diff_factor']  # factor recalculated to integer image type
        elif issubclass(self.data_orig.dtype.type, np.float):
            self.max_d = self.params['general']['max_diff_factor'] * 1./255  # factor recalculated to float image type

        # --  preparing the data -----
        if mask is None:
            self.mask_orig = np.ones(self.data_orig.shape, dtype=np.bool)
        else:
            self.mask_orig = mask

        if (slice is not None) and (self.data_orig.ndim == 3):
            self.data = self.data_orig[slice, :, :]
            self.mask = self.mask_orig[slice, :, :].astype(np.bool)
        else:
            self.data = self.data_orig.copy()
            self.mask = self.mask_orig.copy()

        self.data = img_as_float(self.data)

        if self.params['general']['scale'] != 1:
            self.data = skitra.rescale(self.data, self.params['general']['scale'], mode='nearest', preserve_range=True)
            self.mask = skitra.rescale(self.mask, self.params['general']['scale'], mode='nearest', preserve_range=True)

        self.data, self.mask = tools.crop_to_bbox(self.data, self.mask)

        # --  data smoothing  -----
        if self.params['smoothing']['smooth']:
            self.data = tools.smoothing_tv(self.data, 0.05, sliceId=0)
            # self.data = tools.smoothing_bilateral(self.data, sliceId=0)

        self.data = self.data * self.mask

        self.G = None  # graph of the image

        self.suppxls = None  # superpixels
        self.suppxl_ints_im = None  # image containing intensities of derived superpixels (insteadd of suppxls' labels

        # py3DSeedEditor.py3DSeedEditor(liver_s).show()

    def create_superpixels(self):
        data = self.data.copy()
        mask = self.mask.copy()
        print 'Calculating superpixels ...',
        if data.ndim == 2:
            data_rgb = np.dstack((data, data, data))
            suppxls = skiseg.slic(data_rgb, n_segments=100)
        else:
            suppxls = tools.slics_3D(data, n_segments=100, compactness=10, pseudo_3D=False)
        # py3DSeedEditor.py3DSeedEditor(suppxls, range_per_slice=True).show()
        print 'ok'

        print 'Masking and relabeling superpixels ...',
        # mask superpixels
        suppxls = mask * (suppxls + 1)  # +1 because suppxls starts with 0
        suppxls -= 1  # -1 to shift background to -1 and suppxls first index back to 0
        print 'ok'

        plt.figure()
        data_v = skiexp.rescale_intensity(data, out_range=(0,255)).astype(np.uint8)
        plt.imshow(skiseg.mark_boundaries(data_v, suppxls.astype(np.uint8)))
        plt.axis('off')
        plt.show()

        n_suppxls = len(np.unique(suppxls))
        print '\t# of superpixels: ', n_suppxls

        # relabel them to overcome problems with superpixels cutted with ROI
        suppxls = tools.relabel(suppxls)
        # py3DSeedEditor.py3DSeedEditor(suppxls).show()

        print 'Calculating superpixel intensities ...',
        suppxl_ints_im = tools.suppxl_ints2im(suppxls, im=data)
        print 'ok'

        return suppxls, suppxl_ints_im

    def calc_hom_energy(self, type='mean_bil', normalise=False):
        if self.data.dtype.type == np.float64:
            data = skiexp.rescale_intensity(self.data, in_range='image', out_range=np.uint8).astype(np.uint8)
        else:
            data = self.data.copy()

        selem = skimor.disk(3)
        if type == 'mean':
            hom = skifil.scharr(img_as_float(skifil.rank.mean(data, selem)))
        elif type == 'mean_bil':
            hom = skifil.scharr(img_as_float(skifil.rank.mean_bilateral(data, selem, s0=50, s1=50)))
        elif type == 'median':
            hom = skifil.scharr(img_as_float(skifil.rank.median(data, selem)))
        else:
            hom = skifil.scharr(self.data)

        if normalise:
            hom = skiexp.rescale_intensity(hom, out_range=(0,1))

        # normalize gradient and calculate mean -----
        # hom_1 = skifil.scharr(img_as_float(skifil.rank.mean(data, selem)))
        # hom_2 = skifil.scharr(img_as_float(skifil.rank.mean_bilateral(data, selem, s0=50, s1=50)))
        # hom_3 = skifil.scharr(img_as_float(skifil.rank.median(data, selem)))
        # hom_norm = np.zeros((4, data.shape[0], data.shape[1]))
        # hom_norm[0, :, :] = skiexp.rescale_intensity(hom_1, out_range=(0,1))
        # hom_norm[1, :, :] = skiexp.rescale_intensity(hom_2, out_range=(0,1))
        # hom_norm[2, :, :] = skiexp.rescale_intensity(hom_3, out_range=(0,1))
        # hom_norm[3, :, :] = skiexp.rescale_intensity(hom_4, out_range=(0,1))
        #
        # hom_norm = np.mean(hom_norm, 0)
        #
        # plt.figure()
        # plt.imshow(hom_norm, 'gray', interpolation='nearest')
        # plt.colorbar()
        # -----

        # myFigure.MyFigure((skifil.scharr(self.data),
        #                    skifil.scharr(img_as_float(skifil.rank.mean(data, selem))),
        #                    skifil.scharr(img_as_float(skifil.rank.mean_bilateral(data, selem, s0=50, s1=50))),
        #                    skifil.scharr(img_as_float(skifil.rank.median(data, selem)))),
        #                   title=('scharr', 'mean->scharr', 'mean_bil->scharr', 'median->scharr'),
        #                   int_range=True, colorbar=True)

        return hom

    def calc_seed_energy(self, seeds=None, data=None, ret_en_stack=True):
        if seeds is None:
            seeds = self.seeds.copy()
        if not isinstance(seeds, list):
            seeds = list((seeds,))

        if data is None:
            data = self.data.copy()
        seeds_en = np.zeros(data.shape)

        if ret_en_stack:
            en_stack = np.zeros(np.hstack((len(seeds), data.shape)))

        for i in range(len(seeds)):
            dist_layer, energy_s = gt.get_shopabas(self.G, seeds[i], self.data.shape, self.params['shopabas']['max_diff_factor'])
            if ret_en_stack:
                en_stack[i, :, :] = energy_s
            seeds_en += energy_s

        if ret_en_stack:
            return energy_s, en_stack
        else:
            return energy_s
        
    def calc_energy(self):
        en_hom = self.calc_hom_energy()

        energy = en_hom

        return energy

    def run(self):

        print 'Calculating initial energy ...',
        self.energy = self.calc_energy()
        print 'ok'

        #-----------------------------------
        if self.params['general']['using_superpixels']:
            self.suppxls, self.suppxl_ints_im = self.create_superpixels()

            py3DSeedEditor.py3DSeedEditor(self.suppxl_ints_im).show()

        #-----------------------------------
        if self.params['general']['using_superpixels']:
            self.G, _ = gt.create_graph_from_suppxls(self.data, roi=self.mask, suppxls=self.suppxls,
                                                suppxl_ints=self.suppxl_ints_im, wtype=self.params['graph']['weight_type'])
        else:
            self.G = gt.create_graph(self.data, wtype=self.params['graph']['weight_type'])


def load_parameters(config_name='config.cfg'):
    if config_name != '':
        if os.path.isfile(config_name):
            cf_file = open(config_name, "r")
            lines = ""
            for line in cf_file:
                if "#" not in line:
                    lines += line
            cf_file.close()
            params = json.loads(lines)
            return params
        else:
            raise IOError('%s is invalid file path' % config_name)

#----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # set = 'man_with_hat'
    # fname = '/home/tomas/Dropbox/images/Berkeley_Benchmark/set/%s/original.jpg' % set

    # 2 hypo, 1 on the border --------------------
    slice = 17
    fname = '/home/tomas/Data/liver_segmentation/tryba/data_other/org-exp_183_46324212_venous_5.0_B30f-.pklz'
    # fname = '/home/tomas/Data/liver_segmentation/tryba/data_other/org-exp_183_46324212_arterial_5.0_B30f-.pklz'

    # hypo in venous -----------------------
    # slice = 6
    # arterial - bad
    # fname = '/home/tomas/Data/liver_segmentation_06mm/tryba/data_other/org-exp_186_49290986_venous_0.6_B20f-.pklz'
    # venous - good
    # fname = '/home/tomas/Data/liver_segmentation_06mm/tryba/data_other/org-exp_186_49290986_arterial_0.6_B20f-.pklz'

    # hyper, 1 on the border -------------------
    # arterial 0.6mm - not that bad
    # fname = '/home/tomas/Data/liver_segmentation_06mm/hyperdenzni/org-exp_239_61293268_DE_Art_Abd_0.75_I26f_M_0.5-.pklz'
    # venous 5mm - bad
    # fname = '/home/tomas/Data/liver_segmentation_06mm/hyperdenzni/org-exp_239_61293268_DE_Ven_Abd_0.75_I26f_M_0.5-.pklz'

    # shluk -----------------
    # arterial 5mm
    # fname = '/home/tomas/Data/liver_segmentation/tryba/data_other/org-exp_180_49509315_arterial_5.0_B30f-.pklz'
    # fname = '/home/tomas/Data/liver_segmentation_06mm/tryba/data_other/org-exp_180_49509315_arterial_0.6_B20f-.pklz'

    # targeted -------------
    # arterial 0.6mm - bad
    # fname = '/home/tomas/Data/liver_segmentation_06mm/hyperdenzni/org-exp_238_54280551_Abd_Arterial_0.75_I26f_3-.pklz'
    # venous 0.6mm - bad
    # fname = '/home/tomas/Data/liver_segmentation_06mm/hyperdenzni/org-exp_238_54280551_Abd_Venous_0.75_I26f_3-.pklz'

    # TODO: study ID 25 - 2/2
    # fname = '/home/tomas/Data/liver_segmentation/tryba/data_other/org-exp_185_48441644_venous_5.0_B30f-.pklz'
    # fname = '/home/tomas/Data/liver_segmentation/tryba/data_other/org-exp_185_48441644_arterial_5.0_B30f-.pklz'



    spb = ShoPaBas(fname=fname, data_type=DATA_DICOM, slice=slice)
    spb.run()

    # TODO: 