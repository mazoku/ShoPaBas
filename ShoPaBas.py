__author__ = 'tomas'

import numpy as np
import scipy.io as scio

from skimage import measure, segmentation
import skimage.filters as skifil
import skimage.morphology as skimor
import skimage.transform as skitra
from skimage import img_as_float
import skimage.io as skiio

import io3d

import cv2
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
import Tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import ConfigParser

from myFigure import *

import tools

DATA_DICOM = 0
DATA_IMG = 1

class ShoPaBas:

    def __init__(self, params='config.ini', data=None, data_type=DATA_IMG, fname=None, mask=None, slice=None):
        self.data_orig = None  # input data in original form
        self.data = None  # working data represents data after smoothing etc.
        self.mask_orig = None  # input mask in original form
        self.mask = None  # working mask represents mask after resizing, bounding boxing etc.
        self.max_d = 0  # maximal allowed distance of a point from the seed in a basin

        # --  preparing parameters -----
        if isinstance(params, str):
            self.params = self.load_parameters(params)

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
            self.max_d = self.params['max_diff_factor']  # factor recalculated to integer image type
        elif issubclass(self.data_orig.dtype.type, np.float):
            self.max_d = self.params['max_diff_factor'] * 1./255  # factor recalculated to float image type

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

        if self.params['scale'] != 1:
            self.data = skitra.rescale(self.data, self.params['scale'], mode='nearest', preserve_range=True)
            self.mask = skitra.rescale(self.mask, self.params['scale'], mode='nearest', preserve_range=True)
        # elif issubclass(self.data.dtype.type, np.int):
            # self.data = img_as_float(self.data)

        self.data, self.mask = tools.crop_to_bbox(self.data, self.mask)

        # --  data smoothing  -----
        if self.params['smoothing']:
            self.data = tools.smoothing_tv(self.data, 0.05, sliceId=0)
            # self.data = tools.smoothing_bilateral(self.data, sliceId=0)

        self.data = self.data * self.mask
        # py3DSeedEditor.py3DSeedEditor(liver_s).show()

    def load_parameters(self, config_path):
        config = ConfigParser.ConfigParser()
        config.read(config_path)

        params = dict()

        # an automatic way
        for section in config.sections():
            for option in config.options(section):
                try:
                    params[option] = config.getint(section, option)
                except ValueError:
                    try:
                        params[option] = config.getfloat(section, option)
                    except ValueError:
                        if option == 'voxel_size':
                            str = config.get(section, option)
                            params[option] = np.array(map(int, str.split(', ')))
                        else:
                            params[option] = config.get(section, option)

        return params

    def calc_energy(self):
        e_hom_1 = skifil.scharr(self.data)
        selem = skimor.disk(3)
        # e_hom_2 = abs(self.data - img_as_float(skifil.rank.mean(self.data, selem)))
        e_hom_2 = skifil.scharr(img_as_float(skifil.rank.mean(self.data, selem)))
        # e_hom_3 = abs(self.data - img_as_float(skifil.rank.mean_bilateral(self.data, selem, s0=10, s1=10)))
        e_hom_3 = skifil.scharr(img_as_float(skifil.rank.mean_bilateral(self.data, selem, s0=10, s1=10)))
        # e_hom_4 = abs(self.data - img_as_float(skifil.rank.median(self.data, selem)))
        e_hom_4 = skifil.scharr(img_as_float(skifil.rank.median(self.data, selem)))

        plt.figure()
        plt.subplot(221), plt.imshow(e_hom_1, 'gray')
        plt.subplot(222), plt.imshow(e_hom_2, 'gray')
        plt.subplot(223), plt.imshow(e_hom_3, 'gray')
        plt.subplot(224), plt.imshow(e_hom_4, 'gray')
        plt.show()

    def run(self):
        self.calc_energy()


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