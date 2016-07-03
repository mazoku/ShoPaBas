from __future__ import division

import os
import sys
from collections import defaultdict
import ConfigParser

import numpy as np
import scipy.linalg as scilin
import networkx as nx

import cv2
import matplotlib.pyplot as plt

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

    def __init__(self, config_path='config.ini', data=None, data_type=DATA_IMG, fname=None, mask=None, slice=None,
                 debug=False):
        self.data_orig = None  # input data in original form
        self.data = None  # working data represents data after smoothing etc.
        self.mask_orig = None  # input mask in original form
        self.mask = None  # working mask represents mask after resizing, bounding boxing etc.
        self.seeds = None  # list of seed points
        self.debug = debug

        # ---- reading parameters ----
        self.params = self.load_parameters(config_path)
        self.win_level = self.params['general']['win_level']
        self.win_width = self.params['general']['win_width']
        self.max_d = self.params['shopabas']['max_diff_factor']  # fmaximal allowed distance in a shopabas
        self.learning_rate = self.params['shopabas']['learning_rate']
        self.lr_decay = self.params['shopabas']['lr_decay']

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
        # self.data = tools.windowing(self.data.copy(), level=self.win_level, width=self.win_width,
        #                               sliceId=0, out_range=(0, 1))
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
        self.seed_hom_str = []  # list of seeds' homogenous strength
        self.curr_iteration = 0  # current number of iteration
        self.repellor_weight = self.params['shopabas']['repellor_weight']
        self.seeds_evo = []  # evolution of individual seeds; saved for visualization

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
            s = pt
        elif form == 'xy':
            s = (pt[1], pt[0])
        else:
            raise AttributeError('Wrong seed form: \'%s\'. Only \'rc\' or \'xy\' are allowed.' % form)
        self.seeds.append(pt)
        self.seed_hom_str.append(self.seed_hom_strength(pt))

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
        # # smer pusobici sily je jednotokvy vektor ukazujici smerem source -> target
        # v = np.array(target) - np.array(source)
        # v = v.astype(np.float) / scilin.norm(v)
        #
        # # zakladem pusobici sily je vzajemna vzdalenost seedu
        # source_lin = np.ravel_multi_index(source, self.data.shape)
        # target_lin = np.ravel_multi_index(target, self.data.shape)
        # d = nx.shortest_path_length(self.G, source_lin, target_lin)
        # v /= d  # cim vzdalenejsi, tim mensi sila
        v, d = self.dist_force(source, target)

        # sila seedu je vztazena k homogenite jeho okoli -> std je mala
        hom_source = self.seed_hom_strength(source)
        hom_target = self.seed_hom_strength(target)
        hom = hom_source / hom_target
        v *= hom  # cim homogenejsi okoli, tim vetsi sila seedu

        # prevazim pomoci learning rate
        v *= self.learning_rate
        # print 'd={}, v={}'.format(d, v)

        return v, d, hom

    def dist_force(self, source, target):
        # smer pusobici sily je jednotokvy vektor ukazujici smerem source -> target
        v = np.array(target) - np.array(source)
        v = v.astype(np.float) / scilin.norm(v)

        # zakladem pusobici sily je vzajemna vzdalenost seedu
        source_lin = np.ravel_multi_index(source, self.data.shape)
        target_lin = np.ravel_multi_index(target, self.data.shape)
        d = nx.shortest_path_length(self.G, source_lin, target_lin)
        v /= d  # cim vzdalenejsi, tim mensi sila

        return v, d


    def seed_hom_strength(self, seed, r=7):
        # plt.figure()
        # plt.imshow(self.data, 'gray', interpolation='nearest')
        # while True:
        #     x = plt.ginput(1, timeout=0)
        #     if x:
        #         x = x[0]
        #     else:
        #         break
        #
        #     seed = (int(x[1]), int(x[0]))
        mask = np.zeros_like(self.data)
        try:
            cv2.circle(mask, (int(seed[1]), int(seed[0])), r, 255, -1)
        except OverflowError:
            pass
        ints = self.data[np.nonzero(mask)]
        str = ints.std()
        str = min(10. / (str + 0.001), 2)
        # print '{}: {}'.format(str, ints)
        # plt.figure()
        # plt.subplot(121), plt.imshow(self.data, 'gray', interpolation='nearest')
        # plt.subplot(122), plt.imshow(mask, 'gray', interpolation='nearest')
        # plt.show()
        return str

    def update_seed(self, target, show=False):
        target_lin = np.ravel_multi_index(target, self.data.shape)
        # print 'source: {}'.format(target)
        forces = []
        homs = []
        for i, source in enumerate(self.seeds):
            source_lin = np.ravel_multi_index(source, self.data.shape)
            # source_f = np.array((len(self.seeds) - 1, 2))
            if source_lin != target_lin:
                f, d, hom = self.force(source, target)
                homs.append(hom)
                forces.append(f)
                # print '\ttarget: {0}  ->  dist = {1:.1f}'.format(source, d)
            else:
                forces.append((0, 0))

        # border repeller
        # td = target[0]
        # rd = self.data.shape[1] - target[1]
        # bd = self.data.shape[0] - target[0]
        # ld = target[1]
        tr = (0, target[1])
        rr = (target[0], self.data.shape[1] - 1)
        br = (self.data.shape[0] - 1, target[1])
        lr = (target[0], 0)
        repellors = (tr, rr, br, lr)
        rep_forces = []
        for rep in repellors:
            df, _ = self.dist_force(rep, target)
            df *= self.repellor_weight
            if np.isnan(df).any():
                raise ValueError('Repellor error: rep={}, seed={}'.format(rep, target))
            rep_forces.append(df)

        # print 'reps: {}'.format(repellors)
        # print 'rep forces: {}'.format(rep_forces)

        force = np.array(forces).sum(0)
        rep_force = np.array(rep_forces).sum(0)
        updated_seed = np.array(target) + force + rep_force
        # print 'pos: {0}, f={1} (hom={3}) -> new:{2}'.format(target, force, updated_seed, homs)

        if self.debug and show:
            # if im is None:
            #     s = np.array(self.seeds)
            #     # cmin, rmin = s.min(0)
            #     cmax, rmax = s.max(0)
            #     im = np.zeros((rmax + 20, cmax + 20))
            im = cv2.cvtColor(self.data.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            for i, s in enumerate(self.seeds):
                if np.ravel_multi_index(s, self.data.shape) == target_lin:
                    c = (0, 255, 0)
                    pt1 = (s[1], s[0])
                    pt2 = (int(s[1] + force[1]), int(s[0] + force[0]))
                    cv2.line(im, pt1, pt2, c, 1)
                else:
                    c = (0, 0, 255)
                    pt1 = (s[1], s[0])
                    pt2 = (int(s[1] + forces[i][1]), int(s[0] + forces[i][0]))
                    cv2.line(im, pt1, pt2, c, 1)
                cv2.circle(im,  (s[1], s[0]), 1, c, -1)
            # cv2.imshow('update', cv2.resize(im, (0, 0), fx=8, fy=8))
            cv2.imshow('update', im)
            cv2.waitKey(-1)

        return updated_seed, force, rep_force

    def iteration(self):
        self.curr_iteration += 1
        print '\niteration #%i ------------' % self.curr_iteration

        updated_seeds = self.seeds[:]
        updated_seed_hom_str = self.seed_hom_str[:]
        for i, s in enumerate(self.seeds):
            updated, force, rep_force = self.update_seed(s, show=False)
            updated = np.round(updated).astype(np.int)
            updated_seeds[i] = updated
            updated_hom = self.seed_hom_strength(updated)
            updated_seed_hom_str[i] = updated_hom
            # TODO: tobogan z gradientu
            print 'position: {0} -> {1}, hom: {2:.2f} -> {3:.2f}, force:{6}, s_force: {4}, rep_force: {5}'.format(s, updated,
            self.seed_hom_str[i], updated_hom, force, rep_force, force + rep_force)
        if (np.array(self.seeds) == np.array(updated_seeds)).all() or\
            abs(np.array(self.seeds) - np.array(updated_seeds)).sum() < 2:
            return -1
        else:
            self.seeds = updated_seeds
            self.seeds_evo.append(updated_seeds[:])
            self.seed_hom_str = updated_seed_hom_str
            return 1

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
            ret = self.iteration()
            if ret == -1:
                break

        # seeds path visualization
        cv2.namedWindow('vis', cv2.WINDOW_NORMAL)
        seeds = np.array(self.seeds_evo)  # (n_iter+1, n_seeds, 2)
        imv = cv2.cvtColor(self.data, cv2.COLOR_GRAY2BGR)
        colors = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0], [255, 0, 255], [0, 255, 255]])
        for i, it in enumerate(seeds):
            for j, s in enumerate(it):
                color = ((i+1) / seeds.shape[0] * colors[j, :]).astype(np.int64)
                # cv2.circle(imv, (s[1], s[0]), 3, (color[0], color[1], color[2]), -1)
                cv2.circle(imv, (s[1], s[0]), 1, color, -1)
        cv2.imshow('vis', imv)
        cv2.waitKey(-1)
        cv2.destroyAllWindows()
        pass


# ---------------------------------------------------------------
if __name__ == '__main__':
    data_fname = '/home/tomas/Data/medical/liver_segmentation/org-exp_183_46324212_venous_5.0_B30f-.pklz'
    data, mask, voxel_size = tools.load_pickle_data(data_fname)

    slice_ind = 17
    data_s = data[slice_ind, :, :]
    data_s = tools.windowing(data_s)
    mask_s = mask[slice_ind, :, :]

    data_s, mask_s = tools.crop_to_bbox(data_s, mask_s)

    data_s *= mask_s.astype(data_s.dtype)

    # plt.figure()
    # plt.imshow(data_s, 'gray')
    # pts = plt.ginput(3)
    # seeds = [(int(x[1]), int(x[0])) for x in pts]

    # data = np.zeros((100, 100), dtype=np.uint8)
    # seeds = [(50, 10), (45, 40), (50, 60)]
    seeds = [(147, 23), (167, 68), (74, 79)]
    n_classes = 3
    n_iters = 40
    debug = True

    spb = ShoPaBas(data=data_s, debug=debug)
    for s in seeds:
        spb.add_seed(s)
    spb.seeds_evo.append([np.array(x) for x in spb.seeds])
    spb.run(n_classes, n_iters=n_iters)