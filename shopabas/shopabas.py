from __future__ import division

import os
import sys
from collections import defaultdict
import ConfigParser

import numpy as np
import networkx as nx

import skimage.io as skiio
import skimage.transform as skitra
import skimage.exposure as skiexp
import skimage.morphology as skimor
import skimage.filters as skifil
from skimage import img_as_float

import io3d
import graph_tools as gt

if os.path.exists('../../imtools/'):
    # sys.path.append('../imtools/')
    sys.path.insert(0, '../../imtools/')
    from imtools import tools, misc
else:
    print 'You need to import package imtools: https://github.com/mjirik/imtools'
    sys.exit(0)

# defining constants
DATA_DICOM = 0
DATA_IMG = 1


class ShoPaBas:

    def __init__(self, config_path='config.ini', data=None, data_type=DATA_IMG, fname=None, mask=None, slice=None):
        self.data_orig = None  # input data in original form
        self.data = None  # working data represents data after smoothing etc.
        self.mask_orig = None  # input mask in original form
        self.mask = None  # working mask represents mask after resizing, bounding boxing etc.
        self.seeds = None  # list of seed points

        # ---- reading parameters ----
        self.params = self.load_parameters(config_path)
        self.win_level = self.params['general']['win_level']
        self.win_width = self.params['general']['win_width']
        self.max_d = self.params['shopabas']['max_diff_factor']  # fmaximal allowed distance in a shopabas

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

        # converting data (w.r.t. window width and level) to float <0,1>
        self.data = tools.windowing(self.data.copy(), level=self.win_level, width=self.win_width,
                                      sliceId=0, out_range=(0, 1))
        self.max_d *= 1. / 255  # distance recalculated to float image type

        if self.params['general']['scale'] != 1:
            self.data = skitra.rescale(self.data, self.params['general']['scale'], mode='nearest', preserve_range=True)
            self.mask = skitra.rescale(self.mask, self.params['general']['scale'], mode='nearest', preserve_range=True)

        self.data, self.mask = tools.crop_to_bbox(self.data, self.mask)

        # --  data smoothing  -----
        if self.params['smoothing']['smooth']:
            self.data = tools.smoothing_tv(self.data, weight=self.params['smoothing']['tv_weight'],
                                           sliceId=0, return_uint=False)
            # self.data = tools.smoothing_bilateral(self.data, sliceId=0)

        self.data = self.data * self.mask

        self.G = None  # graph of the image

        self.suppxls = None  # superpixels
        self.suppxl_ints_im = None

        self.seeds = []  # list of seed points in [row, column] form
        self.curr_iteration = 0  # current number of iteration

    @staticmethod
    def load_parameters(config_path):
        if os.path.isfile(config_path) and os.path.exists(config_path):
            config = ConfigParser.ConfigParser()
            config.read(config_path)
            params = defaultdict(dict)

            # an automatic way
            for section in config.sections():
                for option in config.options(section):
                    try:
                        params[section][option] = config.getint(section, option)
                    except ValueError:
                        try:
                            params[section][option] = config.getfloat(section, option)
                        except ValueError:
                            params[section][option] = config.get(section, option)

            return params
        else:
            raise IOError('Invalid file path: %s' % config_path)

    def add_seed(self, pt, form='rc'):
        '''
        Add a new seed point to the list.
        :param pt: a new seed point
        :param type: if the point is in [row, column] or [x, y] form
        :return:
        '''
        if form == 'rc':
            self.seeds.append(pt)
        elif form == 'xy':
            self.seeds.append((pt[1], pt[2]))
        else:
            raise AttributeError('Wrong seed form: \'%s\'. Only \'rc\' or \'xy\' are allowed.' % form)

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
            hom = skiexp.rescale_intensity(hom, out_range=(0, 1))

        return hom

    def calc_seed_energy(self, seeds=None, data=None, ret_en_stack=True):
        # if seeds is None:
        #     seeds = self.seeds.copy()
        if not isinstance(seeds, list):
            seeds = list((seeds,))

        if data is None:
            data = self.data.copy()
        seeds_en = np.zeros(data.shape)

        if ret_en_stack:
            en_stack = np.zeros(np.hstack((len(seeds), data.shape)))

        for i in range(len(seeds)):
            dist_layer, energy_s = gt.get_shopabas(self.G, seeds[i], self.data.shape, self.max_d)
            if ret_en_stack:
                en_stack[i, :, :] = energy_s
            seeds_en += energy_s.reshape(seeds_en.shape)

        if ret_en_stack:
            return energy_s, en_stack
        else:
            return energy_s

    def calc_energy(self):
        en_hom = self.calc_hom_energy()

        # if self.seeds is not None:
        #     en_seeds = self.calc_seed_energy(ret_en_stack=False)
        # else:
        #     en_seeds = 0

        # TODO: zkombinovat energie - en_hom, en_seeds
        energy = en_hom

        return energy

    def force(self, source, target):
        d = nx.dijkstra_path_length(self.G, source, target)
        v = np.array(source) - np.array(target)

        # TODO: urcit smer sily a nascalovat

    def update_seed(self, seed):
        dists = np.zeros(len(self.seeds))
        target = np.ravel_multi_index(seed, self.data.shape)
        print 'source: (%i, %i)' % (seed[0], seed[1])
        for i, s in enumerate(self.seeds):
            source = np.ravel_multi_index(s, self.data.shape)
            f = self.force(source, target)
            d = nx.shortest_path_length(self.G, source, target)
            dists[i] = d
            print '\ttarget: (%i, %i)  ->  dist = %.1f' % (s[0], s[1], d)
        # print dists

    def iteration(self):
        self.curr_iteration += 1
        print 'iteration #%i' % self.curr_iteration

        for s in self.seeds:
            self.update_seed(s)

    def run(self, n_classes, n_iters=10):
        print 'Calculating initial energy ...',
        self.energy = self.calc_energy()
        print 'ok'

        # -----------------------------------
        if self.params['general']['using_superpixels']:
            self.suppxls, self.suppxl_ints_im = self.create_superpixels()

            # py3DSeedEditor.py3DSeedEditor(self.suppxl_ints_im).show()

        # -----------------------------------
        if self.params['general']['using_superpixels']:
            self.G, _ = gt.create_graph_from_suppxls(self.data, roi=self.mask, suppxls=self.suppxls,
                                                     suppxl_ints=self.suppxl_ints_im,
                                                     wtype=self.params['graph']['weight_type'])
        else:
            self.G = gt.create_graph(self.data, wtype=self.params['graph']['weight_type'])

        self.curr_iteration = 0
        for i in range(n_iters):
            self.iteration()


# ---------------------------------------------------------------
if __name__ == '__main__':
    data = np.zeros((100, 100), dtype=np.uint8)
    seeds = [(50, 10), (45, 40), (50, 60)]
    n_classes = 3
    n_iters = 1

    spb = ShoPaBas(data=data)
    for s in seeds:
        spb.add_seed(s)
    spb.run(n_classes, n_iters=n_iters)