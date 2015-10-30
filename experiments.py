from __future__ import division

__author__ = 'tomas'

from matplotlib import pyplot as plt
import numpy as np

from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
from skimage.color import rgb2gray
from skimage.feature import CENSURE, daisy, hog
import skimage.exposure as skiexp
import skimage.morphology as skimor

import io3d
import tools

import skfuzzy as fuzz

import py3DSeedEditor


def blobs(image):
# image = data.hubble_deep_field()[0:500, 0:500]
# image_gray = rgb2gray(image)

    blobs_log = blob_log(image, max_sigma=30, num_sigma=10, threshold=.1)
    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    blobs_dog = blob_dog(image, max_sigma=30, threshold=.1)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

    blobs_doh = blob_doh(image, max_sigma=30, threshold=.01)

    blobs_list = [blobs_log, blobs_dog, blobs_doh]
    colors = ['yellow', 'lime', 'red']
    titles = ['Laplacian of Gaussian', 'Difference of Gaussian', 'Determinant of Hessian']
    sequence = zip(blobs_list, colors, titles)

    for blobs, color, title in sequence:
        fig, ax = plt.subplots(1, 1)
        ax.set_title(title)
        ax.imshow(image, 'gray', interpolation='nearest')
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            ax.add_patch(c)

    plt.show()

def censure_features(img):
    detector = CENSURE()
    detector.detect(img)

    fig, ax = plt.subplots(nrows=1, ncols=1)

    plt.gray()

    ax.imshow(img)
    ax.axis('off')
    ax.scatter(detector.keypoints[:, 1], detector.keypoints[:, 0],
              2 ** detector.scales, facecolors='none', edgecolors='r')
    plt.show()

def daisy_features(img):
    descs, descs_img = daisy(img, step=40, radius=15, rings=2, histograms=8,
                             orientations=8, visualize=True)

    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(descs_img)
    descs_num = descs.shape[0] * descs.shape[1]
    ax.set_title('%i DAISY descriptors extracted:' % descs_num)
    plt.show()

def hog_features(img):
    fd, hog_image = hog(img, orientations=4, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualise=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    ax1.axis('off')
    ax1.imshow(img, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = skiexp.rescale_intensity(hog_image, in_range=(0, 0.02))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()

def imshow(image, title, **kwargs):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(image, **kwargs)
    plt.gray()
    ax.axis('off')
    ax.set_title(title)

def holes_and_peaks(image, segmentation):
    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.max()
    mask = image

    filled = skimor.reconstruction(seed, mask, method='erosion')

    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.min()
    rec = skimor.reconstruction(seed, mask, method='dilation')

    # imshow(image, 'original', cmap='gray', vmin=0, vmax=255)
    # imshow(filled, 'after filling holes', cmap='gray', vmin=image.min(), vmax=image.max())
    # imshow(filled, 'after filling holes', cmap='gray', vmin=0, vmax=255)
    # imshow(image - filled, 'holes')
    # imshow(image - rec, 'peaks')

    # ints = image[np.nonzero(segmentation)]
    # hist_1, bins_1 = skiexp.histogram(ints, nbins=256)
    # ints = filled[np.nonzero(segmentation)]
    # hist_2, bins_2 = skiexp.histogram(ints, nbins=256)
    # plt.figure()
    # plt.subplot(211)
    # plt.plot(bins_1, hist_1)
    # plt.subplot(212)
    # plt.plot(bins_2, hist_2)
    #
    # plt.show()

    return filled

def fcm(img, mask, max_centers=2, all_labels=False):
    if img.ndim == 2:
        is_2D = True
    else:
        is_2D = False

    coords = np.nonzero(mask)
    data = img[coords]
    # alldata = np.vstack((coords[0], coords[1], data))
    alldata = np.vstack((data, data))
    # alldata = np.vstack((data, data))

    fpcs = []
    fpc_max = 0
    u_o = None

    if all_labels:
        labels_all = np.zeros(np.hstack((img.shape[0], img.shape[1], max_centers - 1)))

    for ncenters in range(2, max_centers + 1):
        print 'calculating for %i centers...' % ncenters
        # cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)

        if all_labels:
            # Store partitioning
            cm = np.argmax(u, axis=0) + 1  # cluster membership
            labs = np.zeros(img.shape)
            labs[coords] = cm
            labels_all[:, :, ncenters - 2] = labs

        # Store fpc values for later
        fpcs.append(fpc)

        # Test the result
        if fpc > fpc_max:
            fpc_max = fpc
            u_o = u

    cm = np.argmax(u_o, axis=0) + 1  # cluster membership
    labels = np.zeros(img.shape)
    labels[coords] = cm

    print fpcs

    if not all_labels:
        if is_2D:
            plt.figure()
            plt.subplot(121), plt.imshow(img, 'gray', interpolation='nearest'), plt.axis('off')
            plt.subplot(122), plt.imshow(labels, 'gray', interpolation='nearest'), plt.axis('off')
        else:
            py3DSeedEditor.py3DSeedEditor(labels).show()

    if all_labels and is_2D:
        plt.figure()
        plt.subplot(221), plt.imshow(img, 'gray', interpolation='nearest'), plt.axis('off'), plt.title('original')
        plt.subplot(222), plt.imshow(labels_all[:, :, 0], 'gray', interpolation='nearest'), plt.axis('off')
        plt.title('fpcs=%.2f' % fpcs[0])
        plt.subplot(223), plt.imshow(labels_all[:, :, 1], 'gray', interpolation='nearest'), plt.axis('off')
        plt.title('fpcs=%.2f' % fpcs[1])
        plt.subplot(224), plt.imshow(labels_all[:, :, 2], 'gray', interpolation='nearest'), plt.axis('off')
        plt.title('fpcs=%.2f' % fpcs[2])

    plt.show()

        # Plot assigned clusters, for each data point in training set
        # cluster_membership = np.argmax(u, axis=0)
        # for j in range(ncenters):
        #     ax.plot(xpts[cluster_membership == j],
        #             ypts[cluster_membership == j], '.', color=colors[j])

        # Mark the center of each fuzzy cluster
        # for pt in cntr:
        #     ax.plot(pt[0], pt[1], 'rs')

        # ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
        # ax.axis('off')

    # fig1.tight_layout()

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

    dr = io3d.DataReader()
    if fname is not None:
        datap = dr.Get3DData(fname, dataplus_format=True)
        img = datap['data3d']#[slice, :, :]
        mask = datap['segmentation']#[slice, :, :]

        # windowing
        img = tools.windowing(img).astype(np.uint8)

        # reverting
        # img = skiexp.rescale_intensity(img, in_range=(0,255), out_range=(255,0))

        # smoothing
        img = tools.smoothing(img, d=20, sigmaColor=20, sigmaSpace=10)

        # masking
        img *= mask > 0
        # img = skiexp.rescale_intensity(img, out_range=np.uint8).astype(np.uint8) * (mask > 0)
    else:
        image_rgb = data.hubble_deep_field()[0:500, 0:500]
        img = rgb2gray(image_rgb)
        mask = np.ones_like(img)

    img_s = img[slice, :, :]
    mask_s = mask[slice, :, :]
    # blobs(img_s)
    # censure_features(img_s)
    # daisy_features(img_s)
    # hog_features(img_s)
    # filled = holes_and_peaks(img_s, mask_s)
    fcm(img_s, mask_s, all_labels=False)