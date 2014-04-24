__author__ = 'tomas'

import numpy as np
import matplotlib.pyplot as plt
import tools
import graph_tools as gt
import cv2
import skimage.morphology as skimor
import skimage.segmentation as skiseg
import misc
import networkx as nx
import scipy.ndimage.morphology as scindimor
import scipy.ndimage.measurements as  scindimea
import pylab
import py3DSeedEditor


def make_neighborhood_matrix(im, nghood=4):
    im = np.array(im, ndmin=3)
    n_slices, n_rows, n_cols = im.shape
    npts = n_rows * n_cols * n_slices
    if nghood == 8:
        nr = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
        nc = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
        ns = np.zeros(nghood)
    elif nghood == 4:
        nr = np.array([-1, 0, 0, 1])
        nc = np.array([0, -1, 1, 0])
        ns = np.zeros(nghood, dtype=np.int32)
    elif nghood == 26:
        nr_center = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
        nc_center = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
        nr_border = np.zeros([-1, -1, -1, 0, 0, 0, 1, 1, 1])
        nc_border = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
        nr = np.array(np.hstack((nr_border, nr_center, nr_border)))
        nc = np.array(np.hstack((nc_border, nc_center, nc_border)))
        ns = np.array(np.hstack((-np.ones_like(nr_border), np.zeros_like(nr_center), np.ones_like(nr_border))))
    elif nghood == 6:
        nr_center = np.array([-1, 0, 0, 1])
        nc_center = np.array([0, -1, 1, 0])
        nr_border = np.array([0])
        nc_border = np.array([0])
        nr = np.array(np.hstack((nr_border, nr_center, nr_border)))
        nc = np.array(np.hstack((nc_border, nc_center, nc_border)))
        ns = np.array(np.hstack((-np.ones_like(nr_border), np.zeros_like(nr_center), np.ones_like(nr_border))))
    else:
        print 'Wrong neighborhood passed. Exiting.'
        return None

    lind = np.ravel_multi_index(np.indices(im.shape), im.shape)  # linear indices in array form
    lindv = np.reshape(lind, npts)  # linear indices in vector form
    coordsv = np.array(np.unravel_index(lindv, im.shape))  # coords in array [dim * nvoxels]

    neighborsM = np.zeros((nghood, npts))
    for i in range(npts):
        s, r, c = tuple(coordsv[:,i])
        for nghb in range(nghood):
            rn = r + nr[nghb]
            cn = c + nc[nghb]
            sn = s + ns[nghb]
            if rn < 0 or rn > (n_rows - 1) or cn < 0 or cn > (n_cols - 1) or sn < 0 or sn > (n_slices - 1):
                neighborsM[nghb, i] = np.NaN
            else:
                indexN = np.ravel_multi_index((sn, rn, cn), im.shape)
                neighborsM[nghb, i] = indexN

    return neighborsM


def create_graph(im, nghood=4, wtype=1):
    nghb_m = make_neighborhood_matrix(im, nghood)
    n_nodes = nghb_m.shape[1]
    imv = np.reshape(im, n_nodes).astype(float)
    G = nx.Graph()
    # adding nodes
    G.add_nodes_from(range(n_nodes))
    # adding edges
    sigma = 10
    for n in range(n_nodes):
        for nghb_i in range(1, nghood):
            nghb = nghb_m[nghb_i, n]
            if np.isnan(nghb):
                continue
            if wtype == 1:
                w = 1. / np.exp(- np.absolute(imv[n] - imv[nghb]) / sigma)  # w1
            elif wtype == 2:
                w = 1. / np.exp(- (imv[n] - imv[nghb])**2 / (2 * sigma**2))  # w2
            else:
                w = np.absolute(imv[n] - imv[nghb])  # w3
            G.add_edge(n, nghb, {'weight': w})
    return G


def get_gradient_penalty(im, scale=1):
    im2 = cv2.resize(im, dsize=(0,0), fx=scale, fy=scale)

    grad_x = cv2.Sobel(im2, ddepth=cv2.CV_16S, dx=1, dy=0, ksize=3)
    grad_y = cv2.Sobel(im2, ddepth=cv2.CV_16S, dx=0, dy=1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    dst = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    dst = dst.reshape(im.shape)

    n_pixels = im.shape[0] * im.shape[1]
    dst_vec = dst.reshape(n_pixels)

    return dst, dst_vec


def get_centers_of_non_labeled_areas(label_im, mask):
    non_labeled_im = mask * (label_im == 0)
    if label_im.ndim == 2:
        labels = skimor.label(non_labeled_im, background=0)
    else:
        labels = scindimea.label(non_labeled_im)[0]
    n_labels = labels.max() + 1
    cent_dists = np.zeros(n_labels)
    cent_idxs = -1 * np.zeros(n_labels, dtype=np.int)

    dists = scindimor.distance_transform_edt(labels + 1)

    for i in range(n_labels):
        lab = labels == i
        lab_dists = lab * dists
        cent_dists[i] = lab_dists.max()
        cent_idxs[i] = np.argmax(lab_dists)

    cent_idx = cent_idxs[np.argmax(cent_dists)]

    return cent_idx


def visualize(im, energy_s_im, srcs_en_im, label_im, seed_coords_l, coor, is_interactive):
    if is_interactive:
            plt.subplot(222), plt.imshow(energy_s_im, 'gray', interpolation='nearest'), plt.hold(True), plt.axis('image')
            plt.plot(coor[1], coor[0], 'ro')
            plt.title('energy_s_im')

            plt.subplot(223), plt.imshow(srcs_en_im, 'gray', interpolation='nearest'), plt.hold(True), plt.axis('image')
            for coor in seed_coords_l:
                plt.plot(coor[1], coor[0], 'ro')
            plt.title('srcs_en_im')

            plt.subplot(224), plt.imshow(label_im, interpolation='nearest'), plt.hold(True), plt.axis('image')
            for coor in seed_coords_l:
                plt.plot(coor[1], coor[0], 'ro')
            plt.title('label_im')
    else:
        plt.figure()
        plt.subplot(221), plt.imshow(im, 'gray', vmin=0, vmax=255, interpolation='nearest')
        plt.title('input')

        plt.subplot(222), plt.imshow(energy_s_im, 'gray', interpolation='nearest'), plt.hold(True), plt.axis('image')
        plt.plot(coor[1], coor[0], 'ro')
        plt.title('energy_s_im')

        plt.subplot(223), plt.imshow(srcs_en_im, 'gray', interpolation='nearest'), plt.hold(True), plt.axis('image')
        for coor in seed_coords_l:
            plt.plot(coor[1], coor[0], 'ro')
        plt.title('srcs_en_im')

        plt.subplot(224), plt.imshow(label_im, interpolation='nearest'), plt.hold(True), plt.axis('image')
        for coor in seed_coords_l:
            plt.plot(coor[1], coor[0], 'ro')
        plt.title('label_im')

        plt.show()


def get_shopabas(G, seed, max_d, suppxls, im, using_superpixels, init_dist_val):
    if im.ndim == 3:
        is_3D = True
    else:
        is_3D = False

    n_rows, n_cols = im.shape[:2]
    if is_3D:
        n_slices = im.shape[2]
    else:
        n_slices = 1

    # urceni vzdalenosti od noveho seedu
    dists, _ = nx.single_source_dijkstra(G, seed, cutoff=max_d)

    # z rostouci vzdalenosti udelam klesajici (penalizacni) energii
    energy_s = np.zeros(n_rows * n_cols * n_slices)
    energy_s_im = np.zeros(im.shape)
    dists_items_array = np.array(dists.items())
    dist_layer = init_dist_val * np.ones(im.shape)

    if using_superpixels:
        idxs = dists_items_array[:, 0].astype(np.uint32)
        dists = dists_items_array[:, 1]
        for i in range(len(idxs)):
            suppxl = suppxls == idxs[i]
            energy_s_im[np.nonzero(suppxl)] = max_d - dists[i]
            energy_s = energy_s_im.flatten()
            dist_layer[np.nonzero(suppxl)] = dists[i]
    else:
        energy_s[dists_items_array[:, 0].astype(np.uint32)] = max_d - dists_items_array[:, 1]

        # vsechny body inicializuji max. vzdalenost max_d
        dist_layer = init_dist_val * np.ones(n_rows * n_cols* n_slices)
        dist_layer[dists_items_array[:, 0].astype(np.uint32)] = dists_items_array[:, 1]
        dist_layer = dist_layer.reshape(im.shape)

    return dist_layer, energy_s


def iterate(G, im, mask, max_d=10, max_iter=10, using_superpixels=False, suppxls=None, is_interactive=False):
    if im.ndim == 3:
        n_slices, n_rows, n_cols = im.shape
        n_points = n_rows * n_cols * n_slices
    else:
        n_rows, n_cols = im.shape
        n_points = n_rows * n_cols
    max_sib_diff = 3  # maximal allowed difference from seed to be a sibling

    srcs_en = np.zeros(n_points, dtype=np.uint8)
    init_dist_val = 2 * max_d

    if im.ndim == 2:
        dist_im = init_dist_val * np.ones((n_rows, n_cols))
    else:
        dist_im = init_dist_val * np.ones((n_slices, n_rows, n_cols))
    seed_coords_l = list()  # list of coordinates of seed points
    label_im = np.where(mask == 0, -1, 0)

    if is_interactive and im.ndim == 2:
        plt.figure()
        plt.subplot(221), plt.imshow(im, 'gray', vmin=0, vmax=255, interpolation='nearest')
        plt.title('input')

    iteration = 0
    while iteration < max_iter:
        iteration += 1
        print 'iteration #%i' % iteration

        # generate new seed point
        if is_interactive and im.ndim == 2:
            plt.subplot(221)
            coor = plt.ginput(1, timeout=-1)
            if not coor:
                break
            coor = np.int64(coor).squeeze()
            coor = coor[::-1]
        else:
            new_seed = get_centers_of_non_labeled_areas(label_im, mask)
            if new_seed is None:  # vsechny spely olabelovany
                break
            coor = np.unravel_index(new_seed, im.shape)
        seed_coords_l.append(coor)

        if using_superpixels:
            if im.ndim == 2:
                linx = suppxls[coor[0], coor[1]]
                print 'coor = [%i, %i], linx = %i' % (coor[0], coor[1], linx)
            else:
                linx = suppxls[coor[0], coor[1], coor[2]]
                print 'coor = [%i, %i, %i], linx = %i' % (coor[0], coor[1], coor[2], linx)
        else:
            linx = np.ravel_multi_index(coor, im.shape)

        py3DSeedEditor.py3DSeedEditor(im, contour=suppxls==linx).show()

        seeds = set()
        seeds.add(linx)

        masked_layer = np.zeros(im.shape)
        energy_s_im = np.zeros(im.shape)
        investigated = set()
        while len(seeds) > 0:
            seed = seeds.pop()
            investigated.add(seed)

            # najdu shopabas
            dist_layer, energy_s = get_shopabas(G, seed, max_d, suppxls, im, using_superpixels, init_dist_val)

            # pokud byl uz bod nekam prirazen s mensi vzdalenosti, tak se neprelabeluje
            # if im.ndim == 2:
            #     masked_layer += np.argmin(np.dstack((dist_im, dist_layer)), axis=2) == 1
            # else:
            #     masked_layer += np.where(dist_layer <= dist_im, 1, 0)
            masked_layer += np.where(dist_layer <= dist_im, 1, 0)

            # vymaskuji energii shopabasu podle aktualni masky
            energy_s_sib_im = energy_s.reshape(im.shape)
            energy_s_sib_im = energy_s_sib_im * masked_layer
            energy_s_im += energy_s_sib_im
            energy_s = energy_s_sib_im.flatten()

            siblings = get_siblings(masked_layer, suppxls, im, seed, max_diff=max_sib_diff)
            siblings = siblings.difference(investigated)

            # debug visualization --------------------------------------------
            siblings_im = np.zeros(im.shape)
            siblings_im = np.where(suppxls == seed, 3, siblings_im)  # seed ma label 3 = zelena
            for i in siblings:
                siblings_im += 2 * (suppxls == i)  # siblings maji label 2 = cervena
            for i in seeds:
                siblings_im += suppxls == i  # neprozkoumane seedy maji label 1 = modra

            # plt.figure()
            # plt.imshow(masked_layer, 'gray'), plt.title('masked layer')
            # draw_overlays(im, siblings_im-1)
            #-----------------------------------------------------------------

            seeds = seeds.union(siblings)

        # urceni noveho labelu
        mergers = get_mergers_felzenswalb(G, masked_layer, suppxls, label_im)
        if mergers:
            print 'number of mergers: %i' % len(mergers)
        else:
            print 'number of mergers: 0'

        # component merging
        if mergers:
            label_im = merge_with_components(masked_layer, mergers, label_im)
        else:
            label = label_im.max() + 1
            label_im = np.where(masked_layer, label, label_im)
        dist_im = np.where(masked_layer, dist_layer, dist_im)

        srcs_en += energy_s
        srcs_en_im = srcs_en.reshape(im.shape)

        # kontrola zda jeste existuje neolabelovana oblast
        if (label_im == 0).sum() == 0:
            break

        # visualization ----------------------------------------------------------------------------------
        visualize(im, energy_s_im, srcs_en_im, label_im, seed_coords_l, coor, is_interactive)

    # final visualization -----------------------------------------------------------------------------
    # visualize(im, energy_s_im, srcs_en_im, label_im, seed_coords_l, coor, is_interactive)
    # draw_overlays(im, label_im, cmap='jet', show_now=False)

    # plt.figure()
    # plt.imshow(im, 'gray')
    #
    # plt.figure()
    # plt.imshow(label_im, 'jet')
    #
    # plt.figure()
    # plt.imshow(skiseg.mark_boundaries(im, label_im))
    # plt.show()


def get_siblings(dists, suppxls, im, seed_idx, max_diff=2):
    suppxl_ints = gt.get_suppxl_ints(im, suppxls)
    masked = (dists > 0) * suppxls

    idxs = np.unique(masked[np.nonzero(masked)])
    masked_ints = np.zeros(idxs.shape, dtype=np.float)
    for i in range(len(idxs)):
        suppxl = suppxls == idxs[i]
        intens = suppxl_ints[np.nonzero(suppxl)][0]
        masked_ints[i] = intens

    seed_int = suppxl_ints[np.nonzero(suppxls==seed_idx)][0]

    diffs = np.abs(masked_ints - seed_int)
    siblings = idxs[np.argwhere(diffs <= max_diff)]
    if len(siblings) > 1:  # minimalne tam bude samotny seed
        siblings = np.squeeze(siblings)
        siblings = set(siblings)
        siblings.remove(seed_idx)
    else:
        siblings = set()

    #TODO: filtrovat siblingy podle soucasneho labelovani
    # Pokud ma sibling nejaky label, prevzit ho.
    # Pokud maji siblingy nekolik labelu, prislusne tridy spojit.
    # Olabelovaneho siblinga jiz dale nerozvijet.
    return siblings


def get_label(dists, seed_linxs_l, curr_linx, seed_labels, seed_intens, curr_intens, method='geom', max_inner_dist=5, max_intens_diff=5):
    # inicialization of seed label
    label = max(seed_labels) + 1

    # geom ... finds the closest of the seeds and if it's close enough, then assign its label
    if method == 'geom':
        try:
            srcs_dists = np.array( [dists[s] for s in seed_linxs_l] )  # najde vzdalenost od vsech ostatnich sourcu
        except IndexError:
            label = max(seed_labels) + 1
            return label
        if len(srcs_dists) != 0 and srcs_dists.min() <= max_inner_dist:
            print 'Merging seeds.'
            i = srcs_dists.argmin()
            label = seed_labels[i]

    # intens ... finds a seed that has more similar intensity and if it's similar enough then assign its label
    elif method == 'intens' and len(seed_labels) != 0:
        srcs_dists = np.abs(curr_intens - np.array(seed_intens).astype(np.int))  # najde vzdalenost od vsech ostatnich sourcu
        if len(srcs_dists) != 0 and srcs_dists.min() <= max_intens_diff:
            print 'Merging seeds.'
            i = srcs_dists.argmin()
            label = seed_labels[i]

    return label


def merge_with_components(comp, mergers, label_im):
    label = min(mergers)
    # get neighbors that are to be merged
    mask = np.in1d(label_im, mergers).reshape(label_im.shape)
    # relabel them
    label_im[np.nonzero(mask)] = label
    # label the component with the same label
    label_im[np.nonzero(comp)] = label
    # relabel label_im image to remove unassigned label indices
    label_im = skimor.label(label_im, background=0) + 1
    return label_im


def get_neighbors(comp, suppxls, label_im):
    # get all neighboring components of input component 'comp'
    comp_b = skimor.binary_dilation(comp, skimor.square(3)) - comp

    labeled_suppxls_in_b = suppxls[np.nonzero((label_im > 0) * comp_b)]

    neighbors = np.unique(labeled_suppxls_in_b)
    return neighbors


def get_mergers_felzenswalb(G, comp, suppxls, label_im, t=10):
    # get labels of all neighboring components of input component 'comp'
    suppxl_inds_in_b = get_neighbors(comp, suppxls, label_im)

    if len(suppxl_inds_in_b) == 0:
        return None

    # calculate intra-class similarity
    intra_comp = get_intra_class_similarity(G.copy(), comp, suppxls)

    merge_nghb_idxs = list()
    for i in range(len(suppxl_inds_in_b)):
        # get neighboring superpixel with given index
        suppxl = suppxls == suppxl_inds_in_b[i]
        # get label of given suppxl
        label = np.unique(label_im[np.nonzero(suppxl)])
        # get component with given label
        comp_nghb = label_im == label
        # calculate intra-class similarity
        intra_comp_nghb = get_intra_class_similarity(G.copy(), comp_nghb, suppxls)
        inter_comp = get_inter_class_similarity(G.copy(), comp, comp_nghb, suppxls)

        mint = min(intra_comp + t, intra_comp_nghb + t)
        if inter_comp < mint:
            merge_nghb_idxs.append(label)

    return merge_nghb_idxs


def get_mergers_graphcut(G, comp, suppxls, label_im, t=10):
    # get labels of all neighboring components of input component 'comp'
    suppxl_inds_in_b = get_neighbors(comp, suppxls, label_im)

    if len(suppxl_inds_in_b) == 0:
        return None


def get_intra_class_similarity(G, comp, suppxls):
    # get indices of graph nodes outside the component
    node_idxs = np.unique(suppxls * (comp == 0))
    # remove these nodes from the graph
    G.remove_nodes_from(node_idxs)
    # compute minimum spanning tree (MST)
    T = nx.minimum_spanning_tree(G)

    #if there's only one node in MST, then max_weight = 0
    if len(T.nodes()) > 1:
        # get the edge with biggest weight in the MST
        max_edge = sorted(T.edges(data=True), key=lambda (source,target,data): data['weight'])[-1]
        # the weight of this edge is desired output
        max_weight = max_edge[-1]['weight']
    else:
        max_weight = 0

    return max_weight


def get_inter_class_similarity(G, comp1, comp2, suppxls):
    # get indices of graph nodes in components
    node_idxs1 = np.unique(suppxls * comp1).astype(np.uint8)
    node_idxs2 = np.unique(suppxls * comp2).astype(np.uint8)
    edges = G.edges(data=True)

    borders = [i for i in edges if (i[0] in node_idxs1 and i[1] in node_idxs2) or
                                   (i[0] in node_idxs2 and i[1] in node_idxs1)]
    weights = [i[2]['weight'] for i in borders]
    max_weight = np.max(weights)
    return max_weight


def draw_overlays(im, labels, colors=(('b','r','g','c','m','y')), cmap=None, show_now=True):
    x = np.arange( 0, im.shape[1] )
    y = np.arange( 0, im.shape[0] )
    xgrid, ygrid = np.meshgrid( x, y )

    f = plt.figure()
    plt.hold( True )
    a = pylab.Axes( f, [0,0,1,1], yticks=[], xticks=[], frame_on=False )
    f.delaxes( plt.gca() )
    f.add_axes( a )
    plt.imshow(im, 'gray')
    if cmap is not None:
        plt.contourf(xgrid, ygrid, labels, cmap=cmap, levels=np.arange(labels.max()+2)-0.5, alpha=0.7)
    else:
        plt.contourf(xgrid, ygrid, labels, levels=np.arange(labels.max()+2)-0.5, colors=colors, alpha=0.3)

    if show_now:
        plt.show()


def run(data, mask, slice, max_d=50, using_superpixels=False, method_type='automatic', max_iter=10):

    if data.dtype != np.uint8:
        data = data.astype(np.uint8)
    if mask == None:
        mask = np.ones(data.shape, dtype=np.bool)

    # img = data[slice, :, :]
    # mask = mask[slice, :, :].astype(np.bool)
    img = data

    img, mask = tools.crop_to_bbox(img, mask)

    # data smoothing
    ims = tools.smoothing_tv(img, 0.05)

    liver_s = ims * mask

    en = liver_s
    wtype = 3

    #-----------------------------------
    # suppxls = skiseg.slic(cv2.cvtColor(en[12,:,:], cv2.COLOR_GRAY2RGB))
    # tmp = np.zeros((en.shape[1], en.shape[2], en.shape[0]))
    # for i in range(en.shape[0]):
    #     tmp[:,:,i] = en[i,:,:]
    #TODO: graf vytvaret pouze ze superpixelu v masce!!!
    print 'Deriving superpixels...'
    # suppxls, suppxls_sw = tools.slics_3D(en, n_segments=100, get_slicewise=True)
    suppxls = tools.slics_3D(en, n_segments=100, pseudo_3D=False)
    # py3DSeedEditor.py3DSeedEditor(suppxls).show()

    print 'Removing empty superpixels...'
    suppxls = tools.remove_empty_suppxls(suppxls)

    print 'Calculating superpixel intensities...'
    suppxl_ints_im = tools.suppxl_ints2im(suppxls, im=en)

    # py3DSeedEditor.py3DSeedEditor(suppxls).show()
    # suppxls = skiseg.slic(cv2.cvtColor(en[12,:,:], cv2.COLOR_GRAY2RGB))
    # plt.figure()
    # plt.imshow(skiseg.mark_boundaries(en[12,:,:], suppxls[12,:,:]))
    # plt.show()
    #-----------------------------------

    print 'Graph under construction...'
    if using_superpixels:
        G, suppxls = gt.create_graph_from_suppxls(en, suppxls=suppxls, suppxl_ints=suppxl_ints_im, wtype=wtype)
    else:
        G = create_graph(en, wtype=wtype)
        suppxls = None
    print '...done.'

    iterate(G, liver_s, mask, max_d=max_d, max_iter=max_iter, using_superpixels=using_superpixels, suppxls=suppxls, is_interactive=method_type=='interactive')


#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # dcmdir = '/home/tomas/Dropbox/Work/Data/medical/org-53596059-export_liver.pklz'
    # dcmdir = '/home/tomas/Dropbox/Work/Data/medical/org-53009707-export_liver.pklz'
    # dcmdir = '/home/tomas/Dropbox/Work/Data/medical/org-52496602-export_liver.pklz'
    dcmdir = '/home/tomas/Dropbox/Work/Data/medical/org-38289898-export1.pklz'
    data = misc.obj_from_file(dcmdir, filetype = 'pickle')

    data3d = data['data3d']
    data3d = tools.windowing(data3d, level=50+(-1000 - data3d.min()), width=300, sliceId=0)
    segmentation = data['segmentation'].astype(np.uint8)

    # downscaling
    scale = 0.5
    data3d = tools.resize3D(data3d, scale, sliceId=0).astype(np.uint8)
    segmentation = tools.resize3D(segmentation, scale, sliceId=0).astype(np.uint8)

    # slice = 46
    slice = 12
    using_suppxls = True
    max_d = 5
    # method_type = 'interactive'
    method_type = 'automatic'
    run(data3d, mask=segmentation, slice=slice, max_d=max_d, using_superpixels=using_suppxls, method_type=method_type)