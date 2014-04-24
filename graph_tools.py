__author__ = 'Ryba'

import numpy as np
import networkx as nx
import skimage.segmentation as skiseg
import skimage.morphology as skimor
import scipy.ndimage.morphology as scindimor
import cv2

#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
def make_neighborhood_matrix(im, nghood=4):
    im = np.array( im, ndmin=3 )
    nslices, nrows, ncols = im.shape
    # if len(im.shape) == 3:
    #     nslices = im.shape[2]
    # else:
    #     nslices = 1
    npts = nrows * ncols * nslices
    # print 'start'
    if nghood == 8:
        nr = np.array( [-1, -1, -1, 0, 0, 1, 1, 1] )
        nc = np.array( [-1, 0, 1, -1, 1, -1, 0, 1] )
        ns = np.zeros( nghood )
    elif nghood == 4:
        nr = np.array( [-1, 0, 0, 1] )
        nc = np.array( [0, -1, 1, 0] )
        ns = np.zeros( nghood, dtype=np.int32 )
    elif nghood == 26:
        nrCenter = np.array( [-1, -1, -1, 0, 0, 1, 1, 1] )
        ncCenter = np.array( [-1, 0, 1, -1, 1, -1, 0, 1] )
        nrBorder = np.zeros( [-1, -1, -1, 0, 0, 0, 1, 1, 1] )
        ncBorder = np.array( [-1, 0, 1, -1, 0, 1, -1, 0, 1] )
        nr = np.array( np.hstack( (nrBorder, nrCenter, nrBorder) ) )
        nc = np.array( np.hstack( (ncBorder, ncCenter, ncBorder) ) )
        ns = np.array( np.hstack( (-np.ones_like(nrBorder), np.zeros_like(nrCenter), np.ones_like(nrBorder)) ) )
    elif nghood == 6:
        nrCenter = np.array( [-1, 0, 0, 1] )
        ncCenter = np.array( [0, -1, 1, 0] )
        nrBorder = np.array( [0] )
        ncBorder = np.array( [0] )
        nr = np.array( np.hstack( (nrBorder, nrCenter, nrBorder) ) )
        nc = np.array( np.hstack( (ncBorder, ncCenter, ncBorder) ) )
        ns = np.array( np.hstack( (-np.ones_like(nrBorder), np.zeros_like(nrCenter), np.ones_like(nrBorder)) ) )
    else:
        print 'Wrong neighborhood passed. Exiting.'
        return None

    lind = np.ravel_multi_index( np.indices( im.shape ), im.shape ) #linear indices in array form
    lindv = np.reshape( lind, npts ) #linear indices in vector form
    coordsv = np.array( np.unravel_index( lindv, im.shape ) ) #coords in array [dim * nvoxels]

    neighborsM = np.zeros( (nghood, npts) )
    for i in range( npts ):
        s, r, c = tuple( coordsv[:,i] )
        for nghb in range(nghood ):
            rn = r + nr[nghb]
            cn = c + nc[nghb]
            sn = s + ns[nghb]
            if rn < 0 or rn > (nrows-1) or cn < 0 or cn > (ncols-1) or sn < 0 or sn > (nslices-1):
                neighborsM[nghb, i] = np.NaN
            else:
                indexN = np.ravel_multi_index( (sn, rn, cn), im.shape )
                neighborsM[nghb, i] = indexN

    return neighborsM


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
def graph2img( g, size):
    im = np.zeros( size, dtype=np.bool )
    nodes = g.nodes()
    nodescoords = np.array( np.unravel_index( np.array(nodes, dtype=np.int), size ) )#.transpose()
    im[nodescoords[0,:], nodescoords[1,:]] = 1
    return im


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
def create_graph( im, nghood=4, wtype=3, talk_to_me=True ):
    if talk_to_me:
        print 'Creating graph...'
        print '\t- constructing neighborhood matrix...'
    nghbm = make_neighborhood_matrix( im, nghood )
    nnodes = nghbm.shape[1]
    imv = np.reshape( im, nnodes ).astype(float)
    G = nx.Graph()
    #adding nodes
    if talk_to_me:
        print '\t- adding nodes...'
    G.add_nodes_from( range(nnodes) )
    #adding edges
    #sigma = imv.max() - imv.min()
    sigma = 10
    if talk_to_me:
        print '\t- adding edges...'
    for n in range( nnodes ):
        for nghbi in range( 1, nghood ):
            nghb = nghbm[nghbi,n]
            if np.isnan(nghb):
                continue
            if wtype == 1:
                w = 1. / np.exp( - np.absolute(imv[n] - imv[nghb]) / sigma ) #w1
            elif wtype == 2:
                w = 1. / np.exp( - (imv[n] - imv[nghb])**2 / ( 2 * sigma**2 )) #w2
            else:
                w = np.absolute(imv[n] - imv[nghb]) #w3
                # if w != 1:
            #     print '%.0f-%.0f -> %e || %f'%(imv[n], imv[nghb], w, 1./w)
            #     pass
            G.add_edge( n, nghb, {'weight':w} )
    if talk_to_me:
        print '...done.'
    return G


def get_suppxl_ints(im, suppxls):
    """Calculates mean intensities of pixels in superpixels
    inputs:
        im ... grayscale image, ndarray [MxN]
        suppxls ... image with suppxls labels, ndarray [MxN]-same size as im
    outputs:
        suppxl_intens ... image with suppxls mean intensities, ndarray [MxN]-same size as im
    """
    n_suppxl = suppxls.max() + 1
    # suppxl_intens = np.zeros(n_suppxl)
    suppxl_ints = np.zeros(suppxls.shape)

    for i in range(n_suppxl):
        sup = suppxls == i
        vals = im[np.nonzero(sup)]
        # try:
        #     suppxl_intens[i] = np.mean(vals)
        # except:
        #     suppxl_intens[i] = -1
        val = np.mean(vals)
        # suppxl_int[np.nonzero(sup)] = suppxl_intens[i]
        suppxl_ints[np.nonzero(sup)] = val

    return suppxl_ints


def remove_empty_suppxls(suppxls):
    """Remove empty superpixels. Sometimes there are superpixels(labels), which are empty. To overcome subsequent
    problems, these empty superpixels should be removed.
    inputs:
        suppxls ... image with suppxls labels, ndarray [MxN]-same size as im
    outputs:
        new_supps ... image with suppxls labels, ndarray [MxN]-same size as im, empty superpixel labels are removed
    """
    n_suppxls = suppxls.max() + 1
    new_supps = np.zeros(suppxls.shape, dtype=np.integer)
    idx = 0
    for i in range(n_suppxls):
        sup = suppxls == i
        if sup.sum() > 0:
            new_supps[np.nonzero(sup)] = idx
            idx += 1
    return new_supps


def make_neighborhood_matrix_from_suppxls(suppxls, suppxls_ints):
    n_suppxls = suppxls.max() + 1
    nghb_m = list()
    for i in range(n_suppxls):
        suppxl = suppxls == i
        if suppxl.ndim == 2:
            suppxl_dil = skimor.binary_dilation(suppxl, skimor.square(3))
        else:
            suppxl_dil = scindimor.binary_dilation(suppxl, np.ones((3,3,3)))
        surround = suppxl_dil - suppxl

        labels = np.unique(suppxls[np.nonzero(surround)])
        nghb_m.append(labels)
    return nghb_m


def create_graph_from_suppxls(im, wtype=3, suppxl_ints=None, suppxls=None, n_segments=100, compactness=10):
    if suppxls is None:
        if im.ndim == 2:
            im_rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            suppxls = skiseg.slic(im_rgb, n_segments=n_segments, compactness=compactness)
            suppxls = remove_empty_suppxls(suppxls)
        else:
            print 'Error - works only on grayscale images.'
            return None
    if suppxl_ints is None:
        suppxl_ints = get_suppxl_ints(im, suppxls)
    suppxl_ints = suppxl_ints.astype(np.int)

    n_nodes = suppxls.max() + 1

    # creating vector of superpixel intensities: suppxl_ints_v[suppxl index] = intensity
    suppxl_ints_v = np.zeros(n_nodes, dtype=np.int)
    for i in range(n_nodes):
        suppxl = suppxls == i
        ints = suppxl_ints[np.nonzero(suppxl)]
        if len(np.unique(ints)) == 1:
            val = ints[0]
        else:
            print 'Warning! A superpixel has two different intensities.'
            val = np.mean(ints)
        suppxl_ints_v[i] = val

    # neighbothood matrix
    nghb_m = make_neighborhood_matrix_from_suppxls(suppxls, suppxl_ints)

    G = nx.Graph()

    # adding nodes
    G.add_nodes_from(range(n_nodes))

    # adding edges
    sigma = 10
    for n in range(n_nodes):
        nghbs = nghb_m[n]
        for i in range(len(nghbs)):
            nghb = nghbs[i]
            if wtype == 1:
                w = 1. / np.exp(- np.absolute(suppxl_ints_v[n] - suppxl_ints_v[nghb]) / sigma)  # w1
            elif wtype == 2:
                w = 1. / np.exp(- (suppxl_ints_v[n] - suppxl_ints_v[nghb])**2 / (2 * sigma**2))  # w2
            else:
                w = np.absolute(suppxl_ints_v[n] - suppxl_ints_v[nghb])  # w3
            G.add_edge(n, nghb, {'weight': w})

    return G, suppxls


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
def splitMST( T, getimg=False, imshape=(0,0) ):
    maxw = 0
    maxn = 0
    maxnbr = 0
    for n, nbrs in T.adjacency_iter():
        if len(nbrs.items()) == 1: #n is a leaf
            continue
        for nbr, eattr in nbrs.items():
            data = eattr['weight']
            if data > maxw and len(T.adj[nbr]) > 1: #don't remove edge to leafs
                maxw = data
                maxn = n
                maxnbr = nbr
                # if data > 0:
                #     print('(%d, %d, %.3f)' % (n, nbr, data))
    #print('(%d, %d, %.3f)' % (maxn, maxnbr, maxw))
    if maxn == 0 and maxnbr == 0:# and len(T) == 2: #when splitting tree of two nodes
        maxn = T.nodes()[0]
        maxnbr = T.nodes()[1]
    T.remove_edge( maxn, maxnbr )
    cclist = nx.connected_component_subgraphs( T )

    if getimg:
        ccim1 = graph2img( cclist[0], imshape )
        #ccim2 = graph2img( cclist[1], imshape )
        ccim = np.where( ccim1, 1, 2)
        return cclist, ccim
    else:
        return cclist


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
def getGraphCost( G ):
    wsum = 0
    for u, v, edata in G.edges( data=True ):
        wsum += edata['weight']
    try:
        score = wsum / len(G.edges())
    except ZeroDivisionError:
        score = 0

    #print 'score = %.3f = %i / %i'%(score, wsum, len(G.edges()))

    return score


def get_graph_dists(g, seed, maxd, shape):
    #compute dists in graph from current seed
    dists, path = nx.single_source_dijkstra( g, seed, cutoff=maxd )
    distsItemsArray = np.array(dists.items())

    #converting dists from tuple to image
    distLayer = np.zeros( g.number_of_nodes() )
    distLayer[ distsItemsArray[:,0].astype(np.uint32) ] = distsItemsArray[:,1]
    distLayer = distLayer.reshape(shape)

    return distLayer