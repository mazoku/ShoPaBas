__author__ = 'tomas'

import numpy as np
import scipy.io as scio

from skimage import measure, segmentation
from skimage.filter import sobel
import skimage.morphology as skmor
import skimage.transform as skitra
from skimage import img_as_float
import skimage.io as skiio

import cv2
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
import Tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import ConfigParser

import tools

class PathBasins:

    def __init__(self, params='config.ini', data=None, fname=None, mask=None):
        self.data_orig = None  # input data in original form
        self.data = None  # working data represents data after smoothing etc.
        self.mask_orig = None  # input mask in original form
        self.mask = None  # working mask represents mask after resizing, bounding boxing etc.
        self.max_d = 0  # maximal allowed distance of a point from the seed in a basin

        # --  preparing parameters -----
        if data is None and fname is not None:
            self.data_orig = skiio.imread(fname, as_grey=True)
        elif data is not None:
            self.data_orig = data
        else:
            raise IOError('No data nor filename were specified.')

        if isinstance(params, str):
            self.params = self.load_parameters(params)

        if issubclass(data.dtype, np.int):
            self.max_d = self.params['max_diff_factor']  # factor recalculated to integer image type
        elif issubclass(data.dtype.type, np.float):
            self.max_d = self.params['max_diff_factor'] * 1./255  # factor recalculated to float image type

        # --  preparing the data -----
        if mask is None:
            self.mask_orig = np.ones(data.shape, dtype=np.bool)
        else:
            self.mask_orig = mask

        if (slice is not None) and (data.ndim == 3):
            self.data = self.data_orig[slice, :, :]
            self.mask = self.mask_orig[slice, :, :].astype(np.bool)
        else:
            self.data = self.data_orig.copy()

        if self.params['scale'] != 1:
            self.data = skitra.rescale(self.data, self.params['scale'], mode='nearest', preserve_range=True)
            self.mask = skitra.rescale(self.mask, self.params['scale'], mode='nearest', preserve_range=True)
        elif issubclass(self.data.dtype.type, np.int):
            self.data = img_as_float(self.data)

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