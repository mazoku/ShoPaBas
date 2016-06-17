__author__ = 'tomas'

import networkx as nx
import numpy as np
# import shopabas_suppxls as spbsuppxl
import tools
import graph_tools as gt


# SHOPABAS_SUPPXLS -------------------------------------------------------------------
def test_intra_class_similarity():
    G = nx.Graph()
    nodes = (1, 2, 3, 4)
    edges = ([(1, 2, 10), (1, 3, 30), (1, 4, 40), (2, 4, 30), (3, 4, 10)])
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(edges)
    suppxls = np.array(((1, 1, 2), (1, 1, 2), (3, 4, 4)), dtype=np.uint8)
    # comp = np.logical_or(suppxls == 1, suppxls == 2)
    comp = np.ones(suppxls.shape, dtype=np.bool)
    v = spbsuppxl.get_intra_class_similarity(G, comp, suppxls)
    print v


def test_inter_class_similarity():
    G = nx.Graph()
    nodes = (1, 2, 3, 4)
    edges = ([(1, 2, 10), (1, 3, 30), (1, 4, 40), (2, 4, 30), (3, 4, 10)])
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(edges)
    suppxls = np.array(((1, 1, 2), (1, 1, 2), (3, 4, 4)), dtype=np.uint8)
    comp1 = suppxls == 1
    comp2 = np.logical_or(suppxls == 2, suppxls == 4)
    v = spbsuppxl.get_inter_class_similarity(G, comp1, comp2, suppxls)
    print v


def test_merge_with_components():
    label_im = np.array(((1, 1, 2), (1, 1, 2), (3, 4, 4)), dtype=np.uint8)
    label_im_o = label_im.copy()
    comp = np.array(((0, 0, 0), (0, 0, 0), (1, 0, 0)), dtype=np.bool)
    mergers = list((1, 2))

    label_im = spbsuppxl.merge_with_components(comp, mergers, label_im)
    print 'label_im before:'
    print label_im_o
    print 'comp:'
    print comp
    print 'mergers:', mergers
    print 'label_im after:'
    print label_im
#-----------------------------------------------------------------------------------


# GRAPH TOOLS ----------------------------------------------------------------------
def test_make_neighborhood_matrix_from_suppxls_roi():
    suppxls = np.array(((0, 0, 0, 1), (0, 0, 1, 1), (2, 2, 3, 3), (2, 2, 3, 3)))
    suppxl_ints = np.array(((0, 0, 0, 1), (0, 0, 1, 1), (2, 2, 3, 3), (2, 2, 3, 3)))
    roi = np.array(((1, 1, 1, 0), (1, 1, 0, 0), (1, 1, 1, 1), (1, 1, 1, 1)))
    nghbm = gt.make_neighborhood_matrix_from_suppxls(suppxls, suppxl_ints, roi)

    print nghbm
#-----------------------------------------------------------------------------------


# TOOLS ----------------------------------------------------------------------------
def test_crop_to_bbox():
    im = np.array((((1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4)),
                    ((1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4)),
                    ((1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4)),
                    ((1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4))))
    mask = np.array((((0, 1, 0, 0), (0, 1, 0, 0), (0, 1, 0, 0), (0, 1, 0, 0)),
                     ((0, 1, 0, 0), (0, 1, 0, 0), (0, 1, 0, 0), (0, 1, 0, 0)),
                     ((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
                     ((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))))

    im_bb, mask_bb = tools.crop_to_bbox(im, mask)

    print 'im: '
    for i in range(im_bb.shape[2]):
        print im_bb[i, :, :]
    print '\n-----------------\n'
    print 'mask: '
    for i in range(mask_bb.shape[2]):
        print mask_bb[i, :, :]


def test_relabel():
    data = np.array((((0, 0, 0), (0, 1, 1), (1, 1, 1)),
                     ((0, 0, 0), (-1, -1, -1), (-1, 4, 4))))
    rel = tools.relabel(data)

    print 'data:'
    print data
    print 'rel:'
    print rel





if __name__ == "__main__":
    test_relabel()