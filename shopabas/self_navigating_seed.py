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


def calc_hom_energy(img, type='mean_bil', normalise=False):
    if img.dtype.type == np.float64:
        data = skiexp.rescale_intensity(img, in_range='image', out_range=np.uint8).astype(np.uint8)
    else:
        data = img.copy()

    selem = skimor.disk(3)
    if type == 'mean':
        hom = skifil.scharr(img_as_float(skifil.rank.mean(data, selem)))
    elif type == 'mean_bil':
        hom = skifil.scharr(img_as_float(skifil.rank.mean_bilateral(data, selem, s0=50, s1=50)))
    elif type == 'median':
        hom = skifil.scharr(img_as_float(skifil.rank.median(data, selem)))
    else:
        hom = skifil.scharr(img)

    if normalise:
        hom = skiexp.rescale_intensity(hom, out_range=(0, 1))

    return hom


def calc_seed_energy(img, seeds, G, ret_en_stack=True, max_d=50):
    # if seeds is None:
    #     seeds = self.seeds.copy()
    if not isinstance(seeds, list):
        seeds = list((seeds,))

    seeds_en = np.zeros(img.shape)

    if ret_en_stack:
        en_stack = np.zeros(np.hstack((len(seeds), img.shape)))

    for i in range(len(seeds)):
        dist_layer, energy_s = gt.get_shopabas(G, seeds[i], img.shape, max_d)
        if ret_en_stack:
            en_stack[i, :, :] = energy_s
        seeds_en += energy_s.reshape(seeds_en.shape)

    if ret_en_stack:
        return energy_s, en_stack
    else:
        return energy_s


def calc_energy(img, G):
    # offset = 20
    # imgd = np.where(img < (img[np.nonzero(img)].mean() - offset), img, 0)
    # imgb = np.where(img > (img[np.nonzero(img)].mean() + offset), img, 0)
    #
    # end_hom = calc_hom_energy(imgd)
    # enb_hom = calc_hom_energy(imgb)
    #
    # plt.figure()
    # plt.subplot(231), plt.imshow(img, 'gray')
    # plt.subplot(232), plt.imshow(imgd, 'gray')
    # plt.subplot(233), plt.imshow(imgb, 'gray')
    # plt.subplot(234), plt.imshow(en_hom, 'gray')
    # plt.subplot(235), plt.imshow(end_hom, 'gray')
    # plt.subplot(236), plt.imshow(enb_hom, 'gray')
    #
    # plt.show()

    # if self.seeds is not None:
    #     en_seeds = self.calc_seed_energy(img, seeds, G, ret_en_stack=False)
    # else:
    #     en_seeds = 0

    en_hom = calc_hom_energy(img)

    # TODO: zkombinovat energie - en_hom, en_seeds
    energy = en_hom

    energy = skiexp.rescale_intensity(energy, in_range=(0, 0.2), out_range=(0, 1))

    return energy


def update_seed(img, s, mag, ori, r):#, mag_c=1):
    s = np.array(s)
    # vymaskovat okolni body
    mask = np.zeros_like(img)
    cv2.circle(mask, (s[1], s[0]), r, 255, -1)

    pts = np.nonzero(mask)
    mags = mag[pts]
    # mag_c = 1 / len(mag[pts] > 0)
    # mags = mag_c * mag[pts]
    oris = ori[pts]
    pts = [(x, y) for x,y in zip(pts[0], pts[1])]

    forces = []
    for pt, m, o in zip(pts, mags, oris):
        u = m * np.array([np.cos(o), np.sin(o)])
        u_rc = [-u[1], u[0]]
        forces.append(u_rc)

    force = np.array(forces).sum(0)

    s = s.astype(force.dtype) + force
    return np.round(s).astype(np.int)


def run(img, seeds, n_iters=10, r=10):
    G = gt.create_graph(img, wtype='exp_abs')

    print 'Calculating initial energy ...',
    energy = calc_energy(img, G)
    print 'ok'

    gx = cv2.Sobel(energy, ddepth=cv2.CV_64F, dx=1, dy=0)
    gy = cv2.Sobel(energy, ddepth=cv2.CV_64F, dx=0, dy=1)
    mag = np.sqrt((gx ** 2) + (gy ** 2))
    orientation = np.arctan2(gy, gx) * (180 / np.pi) % 180

    seeds_evo = []
    # iterace
    for i in range(n_iters):
        print '\niteration #%i --------------' % (i + 1)

        updated_seeds = []
        for s in seeds:
            su = update_seed(energy, s, mag, orientation, r)
            updated_seeds.append(su)
            print '{} -> {}'.format(s, su)
        if (np.array(seeds) == np.array(updated_seeds)).all() or \
                        abs(np.array(seeds) - np.array(updated_seeds)).sum() < 1:
            break
        else:
            seeds = updated_seeds
            seeds_evo.append(updated_seeds[:])

    # seeds path visualization
    cv2.namedWindow('vis', cv2.WINDOW_NORMAL)
    seeds = np.array(seeds_evo)  # (n_iter+1, n_seeds, 2)
    imv = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    colors = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0], [255, 0, 255], [0, 255, 255]])
    for i, it in enumerate(seeds):
        for j, s in enumerate(it):
            color = ((i + 1) / seeds.shape[0] * colors[j, :]).astype(np.int64)
            # cv2.circle(imv, (s[1], s[0]), 3, (color[0], color[1], color[2]), -1)
            cv2.circle(imv, (s[1], s[0]), 1, color, -1)
    cv2.imshow('vis', imv)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()

    # plt.figure()
    # plt.subplot(121), plt.imshow(img, 'gray', interpolation='nearest')
    # plt.subplot(122), plt.imshow(energy, 'gray', interpolation='nearest')
    # plt.show()


# ---------------------------------------------------------------
if __name__ == '__main__':
    data_fname = '/home/tomas/Data/medical/liver_segmentation/org-exp_183_46324212_venous_5.0_B30f-.pklz'
    data, mask, voxel_size = tools.load_pickle_data(data_fname)

    slice_ind = 17
    data_s = data[slice_ind, :, :]
    data_s = tools.windowing(data_s)
    mask_s = mask[slice_ind, :, :]

    data_s, mask_s = tools.crop_to_bbox(data_s, mask_s)

    # mean_v = int(data_s[np.nonzero(mask_s)].mean())
    # data_s = np.where(mask_s, data_s, mean_v)
    data_s *= mask_s.astype(data_s.dtype)



    # data = np.zeros((100, 100), dtype=np.uint8)
    # seeds = [(50, 10), (45, 40), (50, 60)]
    # seeds = [(147, 23), (167, 68), (74, 79)]
    # seeds = [(137, 44)]
    # n_classes = 3
    n_iters = 50
    # debug = True

    data_s = cv2.imread('coridor.png', 0)
    seeds = [(326, 79), (299, 148), (188, 52), (164, 130), (43, 104)]

    # plt.figure()
    # plt.imshow(data_s, 'gray')
    # pts = plt.ginput(5)
    # seeds = [(int(x[1]), int(x[0])) for x in pts]
    # print seeds


    run(data_s, seeds, n_iters=n_iters)