__author__ = 'Ryba'

import numpy as np
import scipy.io as scio
from skimage import measure, segmentation
from skimage.filter import sobel
import skimage.morphology as skmor
import cv2
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
from pylab import *
import Tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class PathBasins(object):

    def __init__( self):
        self.data = None
        self.mask = None
        set = 'man_with_hat'
        fname = '/home/tomas/Dropbox/images/Berkeley_Benchmark/set/%s/original.jpg' % set
        self.data = cv2.imread(fname, 0)

        self.data = cv2.resize( self.data, dsize=(0,0), fx=0.6, fy=0.6 )#fx=0.2, fy=0.2 )
        self.mask3d = np.ones(self.data.shape, dtype=np.integer)

        self.en = None
        self.G = None
        self.maskroi = np.ones(self.data.shape, dtype=np.bool)
        self.imroi = self.data

        # self.im3d = np.dstack((self.im3d,self.im3d))
        # self.mask3d = np.ones( self.im3d.shape, dtype=np.bool )
        x = np.arange( 0, self.data.shape[1] )
        y = np.arange( 0, self.data.shape[0] )
        self.xgrid, self.ygrid = np.meshgrid( x, y )

        self.distT = 30
        self.srcDistT = 30
        self.sourcesL = list()
        self.labelIm = None
        self.distIm =  None
        self.srclblD = dict()

        self.fig = plt.figure()
        # self.fig1 = plt.figure()
        # self.fig2 = plt.figure()
        self.nRows = self.data.shape[0]
        self.nCols = self.data.shape[1]
        self.nPixels = self.nRows * self.nCols
        self.energyS = np.zeros( self.nPixels )

        warnings.warn('Seedy se chytaji na hrany - zkouset je umistovat do homogennich oblasti. '
                      'Napr. penalizovat pomoci gradientu')

#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def run( self, slide ):

        n_iterations = 20
        self.currSlide = slide
        # mask = self.mask3d[:,:,self.currSlide].astype(np.int)
        # im = self.im3d[:,:,self.currSlide] * mask
        # mask = self.mask3d
        # im = self.im3d * mask
        #
        # rp = measure.regionprops( mask, properties=['BoundingBox'] )
        # bbox = np.array(rp[0]['BoundingBox'])
        #bbox += np.array((-1,-1,1,1)).tolist()

        # self.maskroi = mask[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        # self.imroi = im[bbox[0]:bbox[2],bbox[1]:bbox[3]]

        # self.imroi = im

        # self.en = cv2.bilateralFilter( self.imroi.astype(np.uint8), d=10, sigmaColor=20, sigmaSpace=10 )
        # self.en = cv2.bilateralFilter( self.imroi.astype(np.uint8), d=5, sigmaColor=15, sigmaSpace=5 )
        self.en = cv2.bilateralFilter( self.data, d=5, sigmaColor=15, sigmaSpace=5 )
        self.labelIm = np.zeros( self.en.shape )
        self.distIm = 255 * np.ones( self.en.shape, dtype=np.float )

        wtype = 3
        if wtype == 1:
            self.distT = 10
            self.srcDistT = 10
        elif wtype == 2:
            self.distT = 30
            self.srcDistT = 30
        elif wtype == 3:
            self.distT = 30
            self.srcDistT = 30
        elif wtype == 4:
            self.max = 10
            #self.alpha =

        grad_thresh = 10
        self.grad, self.grad_vec = self.get_gradient_penalty(self.en)
        self.grad_vec_norm = cv2.normalize( self.grad_vec.astype(np.float64), dst=None, alpha=0, beta=100, norm_type=cv2.NORM_MINMAX)
        #computing energy of gradient in the sense of distance transform of thresholded edge images
        self.grad_norm = cv2.normalize( self.grad.astype(np.float64), dst=None, alpha=0, beta=100, norm_type=cv2.NORM_MINMAX)
        edges = self.grad_norm > grad_thresh
        edges = skmor.binary_opening(edges, skmor.square(3))
        edges = skmor.binary_closing(edges, skmor.square(3))
        thrash, self.grad_energy = skmor.medial_axis(np.logical_not(edges), return_distance=True)
        #prevratim hodnoty, abych penalizoval body, ktere jsou blize gradientu
        self.grad_energy = self.grad_energy.max() - self.grad_energy
        self.grad_energy = self.grad_energy.reshape(self.nPixels, 1)

        print 'Graph under construction...'
        # self.en = self.imroi.astype(np.uint8)
        self.G = self.create_graph( self.en, wtype=wtype )
        print '...done.'

        #---------------------
        # self.fig = plt.figure()
        # self.root = tk.Tk()
        # self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        # self.canvas.draw()
        # plt.hold( True )
        # self.ax = Axes( self.fig, [0,0,1,1], yticks=[], xticks=[], frame_on=False )
        # self.fig.delaxes( plt.gca() )
        # self.fig.add_axes( self.ax )
        # self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        # self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        ##---------------------
        self.redraw()
        # self.automatic_random( iterations=50 )
        # self.automatic_farest( iterations=50 )

        # self.automatic_lowest_energy( iterations=n_iterations )
        self.automatic_until_all_labeled(maxd=200)

        plt.figure()
        plt.imshow( self.imroi.astype(np.uint8) , 'gray')
        plt.hold(True)
        coords = np.unravel_index( self.sourcesL, self.en.shape )
        plt.plot( coords[1], coords[0], 'wo', markersize=6, markeredgewidth=2 )
        plt.axis('image'), plt.axis('off')

        plt.figure()
        plt.imshow(self.labelIm)
        # plt.imshow( segmentation.mark_boundaries(self.imroi.astype(np.uint8), self.labelIm, color=(1,1,1)))
        # coords = np.unravel_index( self.sourcesL, self.en.shape )
        # plt.plot( coords[1], coords[0], 'wo', markersize=6, markeredgewidth=2 )
        plt.axis('image'), plt.axis('off')

        plt.show()

        # self.root.mainloop()

        # self.srcsen = np.zeros( self.G.number_of_nodes() )
        # self.redraw()
        # self.interactiv()


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def interactiv( self ):
        self.srcsen = np.zeros( self.G.number_of_nodes() )
        while True:
            x = plt.ginput(1)
            if not x:
                break
            x = np.round(x).astype(np.int).squeeze()
            linx = (x[1]-1) * self.en.shape[1] + x[0]
            maxd = 200
            dists, path = nx.single_source_dijkstra( self.G, linx, cutoff=maxd )
            # self.distLayer = np.array(dists.items())[:,1].reshape( self.en.shape )

            nrows, ncols = self.en.shape
            self.energyS = np.zeros( nrows*ncols )
            #z rostouci vzdalenosti udelam klesajici (penalizacni) energii
            distsItemsArray = np.array(dists.items())
            self.energyS[ distsItemsArray[:,0].astype(np.uint32) ] = maxd - distsItemsArray[:,1]
            self.srcsen += self.energyS

            distLayer = (maxd + 20) * np.ones( nrows*ncols )
            distLayer[ distsItemsArray[:,0].astype(np.uint32) ] = distsItemsArray[:,1]
            self.distLayer = distLayer.reshape(self.en.shape)

            # plt.figure()
            # plt.imshow(self.distLayer)
            # plt.show()

            maskedLayer = np.argmin( np.dstack( (self.distIm, self.distLayer) ), axis=2 ) == 1

            label = self.getLabel( dists, linx, method='intens' )

            #labelImOld = self.labelIm.copy()
            self.labelIm = np.where( maskedLayer, label, self.labelIm )
            self.distIm = np.where( maskedLayer, self.distLayer, self.distIm )
            self.srclblD[linx] = label
            self.sourcesL.append( linx )

            self.redraw(drawSources=True)

            #additional visualization
            # plt.figure(self.fig1.number), plt.gray()
            # plt.subplot(121), plt.imshow(energyS.reshape(self.en.shape)), plt.title('energie seedu')
            # plt.subplot(122), plt.imshow(srcsen.reshape(self.en.shape)), plt.title('celkova energie')
            # self.fig1.canvas.draw()


            # plt.figure()
            # plt.imshow(self.distLayer), plt.colorbar()
            # while True:
            #     x = plt.ginput(n=1,timeout=0)
            #     if not x:
            #         break
            #     x = np.round(x).astype(np.int).squeeze()
            #     linx = (x[1]-1) * self.en.shape[1] + x[0]
            #     print dists[linx]


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def automatic_random( self, iterations=10 ):
        iteration = 0
        # idxs = np.ravel_multi_index( np.nonzero(self.maskroi), self.en.shape )
        #not_used = np.ones( len(np.nonzero(self.maskroi)[0]), dtype=np.bool )
        idxs = np.arange( self.G.number_of_nodes(), dtype=np.uint16 )
        used = np.ones( self.G.number_of_nodes(), dtype=np.bool )
        used[ np.ravel_multi_index( np.nonzero(self.maskroi), self.en.shape ) ] = False

        saveimgs = np.zeros( (self.en.shape[0],self.en.shape[1],iterations) )

        nrows, ncols = self.en.shape
        bounds = np.ravel_multi_index( ((0,(nrows-1)/2,(nrows-1),(nrows-1)/2),((ncols-1)/2,(ncols-1),(ncols-1)/2,0)), self.en.shape)
            #np.array([[0,ncols/2],[nrows/2,ncols],[nrows,ncols/2],[nrows/2,0]])
        # idxs = np.hstack( (bounds, idxs) )
        # not_used = np.hstack( (np.zeros( 4, dtype=np.bool ) ,not used) )
        iteration -= 4
        while iteration < iterations:
            to_use = idxs[ np.logical_not(used) ]
            if iteration < 0:
                linx = bounds[iteration + 4]
            else:
                linx = idxs[ np.logical_not(used) ][np.random.randint( low=0, high=len(to_use) )]
                used[linx] = True
            iteration += 1
            #linx = (x[1]-1) * self.en.shape[1] + x[0]


            dists, path = nx.single_source_dijkstra( self.G, linx )
            self.distLayer = np.array(dists.items())[:,1].reshape( self.en.shape )

            maskedLayer = np.argmin( np.dstack( (self.distIm, self.distLayer) ), axis=2 ) == 1

            label = self.getLabel( dists, linx, method='intens' )

            self.labelIm = np.where( maskedLayer, label, self.labelIm )
            self.distIm = np.where( maskedLayer, self.distLayer, self.distIm )
            self.srclblD[linx] = label
            self.sourcesL.append( linx )
            self.redraw( drawSources=True )

            saveimgs[:,:,iteration-1] = self.labelIm

        # plt.imsave(fname='dijkstraSplitNoFiltration_out.png', arr=saveimgs[:,:,-1], cmap=cm.jet, vmin=0, vmax=saveimgs.max())
        # plt.imsave(fname='dijkstraSplitNoFiltration_in.png', arr=self.imroi, cmap=cm.gray)
        # coords = np.unravel_index( self.sourcesL, self.en.shape )
        # f = plt.figure()
        # for i in range(saveimgs.shape[2]):
        #     plt.imshow(saveimgs[:,:,i], vmin=0, vmax=saveimgs.max())
        #     plt.jet()
        #     plt.hold(True)
        #     plt.plot( coords[1][4:i+5], coords[0][4:i+5], 'rx', markersize=6, markeredgewidth=2 )
        #     plt.hold(False)
        #     plt.axis('image')
        #     f.savefig( 'dijkstra_split_no_filter/im_%03i.png'%(i+1), bbox_inches='tight' )
        #     plt.clf()
            # plt.imsave(fname='dijkstrasplit2/im_%03i.png'%(i+1), arr=saveimgs[:,:,i], cmap=cm.jet, vmin=0, vmax=saveimgs.max())


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def automatic_farest( self, iterations=10 ):
        iteration = 0
        srcsen = np.zeros( self.G.number_of_nodes() )
        # idxs = np.ravel_multi_index( np.nonzero(self.maskroi), self.en.shape )
        #not_used = np.ones( len(np.nonzero(self.maskroi)[0]), dtype=np.bool )
        idxs = np.arange( self.G.number_of_nodes(), dtype=np.uint16 )
        used = np.ones( self.G.number_of_nodes(), dtype=np.bool )
        used[ np.ravel_multi_index( np.nonzero(self.maskroi), self.en.shape ) ] = False

        saveimgs = np.zeros( (self.en.shape[0],self.en.shape[1],iterations) )

        nrows, ncols = self.en.shape
        bounds = np.ravel_multi_index( ((0,(nrows-1)/2,(nrows-1),(nrows-1)/2),((ncols-1)/2,(ncols-1),(ncols-1)/2,0)), self.en.shape)
        #np.array([[0,ncols/2],[nrows/2,ncols],[nrows,ncols/2],[nrows/2,0]])
        # idxs = np.hstack( (bounds, idxs) )
        # not_used = np.hstack( (np.zeros( 4, dtype=np.bool ) ,not used) )
        iteration-=4
        while iteration < iterations:
            print iteration
            if iteration < 0:
                linx = bounds[iteration + 4]
            else:
                energy = np.array( dists.items() )
                energy[:,1] -= srcsen
                da = energy[np.logical_not(used),:]
                farest = np.argmax( da[:,1] )
                linx = np.int( da[farest,0] )
                used[linx] = True
            iteration += 1
            #linx = (x[1]-1) * self.en.shape[1] + x[0]

            dists, path = nx.single_source_dijkstra( self.G, linx )
            self.distLayer = np.array(dists.items())[:,1].reshape( self.en.shape )

            maskedLayer = np.argmin( np.dstack( (self.distIm, self.distLayer) ), axis=2 ) == 1

            label = self.getLabel( dists, linx, method='intens' )
            # label = iteration + 1

            # radius = 20
            # srcsenS = nx.single_source_shortest_path_length( self.G, linx )
            # srcsenS = np.array( srcsenS.items() )[:,1]
            # srcsenS = np.where( srcsenS<=radius, 50*(radius-srcsenS), 0 )
            # srcsen += srcsenS

            maxd = 200
            # d, p = nx.single_source_dijkstra( self.G, linx )
            dl = np.array(dists.items())[:,1]
            dl = np.where( dl<maxd, 50*(maxd-dl), 0 )
            srcsen += dl

            self.labelIm = np.where( maskedLayer, label, self.labelIm )
            self.distIm = np.where( maskedLayer, self.distLayer, self.distIm )
            self.srclblD[linx] = label
            self.sourcesL.append( linx )
            self.redraw( drawSources=True )

            saveimgs[:,:,iteration-1] = self.labelIm

            if iteration > 1:
                plt.figure()
                plt.imshow(dl.reshape( self.en.shape ))

                plt.figure()
                plt.imshow(srcsen.reshape( self.en.shape ))

                coords = np.unravel_index( self.sourcesL, self.en.shape )
                plt.hold(True)
                plt.plot( coords[1], coords[0], 'rx', markersize=6, markeredgewidth=2 )

                plt.show()

        coords = np.unravel_index( self.sourcesL, self.en.shape )
        f = plt.figure()
        for i in range(saveimgs.shape[2]):
            plt.imshow(saveimgs[:,:,i], vmin=0, vmax=saveimgs.max())
            plt.jet()
            plt.hold(True)
            plt.plot( coords[1][4:i+5], coords[0][4:i+5], 'rx', markersize=6, markeredgewidth=2 )
            plt.hold(False)
            plt.axis('image')
            f.savefig( 'dijkstra_split_source_en2/im_%03i.png'%(i+1), bbox_inches='tight' )
            plt.clf()

        # plt.figure()
        # plt.imshow( srcsen.reshape(self.en.shape) )
        plt.show()
            # plt.imsave(fname='dijkstraSpliNoFiltration_out.png', arr=saveimgs[:,:,-1], cmap=cm.jet, vmin=0, vmax=saveimgs.max())
        # plt.imsave(fname='dijkstraSpliNoFiltration_in.png', arr=self.imroi, cmap=cm.gray)
        # for i in range(saveimgs.shape[2]):
        #     plt.imsave(fname='dijkstrasplit2/im_%03i.png'%(i+1), arr=saveimgs[:,:,i], cmap=cm.jet, vmin=0, vmax=saveimgs.max())
        #     pass


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def automatic_until_all_labeled( self, maxd=100000 ):
        # srcsen = np.zeros( self.G.number_of_nodes() )
        srcsen = self.grad_vec_norm.squeeze()
        used = np.ones( self.G.number_of_nodes(), dtype=np.bool )
        used[ np.ravel_multi_index( np.nonzero(self.maskroi), self.en.shape ) ] = False

        nrows, ncols = self.en.shape
        initDistVal = 2 * maxd
        self.distIm = initDistVal * np.ones( (nrows, ncols) )

        iteration = 0
        curr_maxd = 500000
        #indexy
        inds = np.arange(0, nrows*ncols, dtype=np.uint32)
        # while curr_maxd > maxd and iteration < 5:
        # while curr_maxd > maxd:
        while iteration < 15:
            iteration += 1
            print 'iteration #%i'%(iteration)

            #energie nepouzitych pixelu
            energyNotUsed = srcsen[np.logical_not(used)]
            indsNotUsed = inds[np.logical_not(used)]

            #novy seed
            # linx= self.get_new_seed_gradient_based(energyNotUsed, indsNotUsed, self.grad)
            linx, seeds_energy = self.get_new_seed_gradient_based(srcsen, inds, self.grad)
            used[linx] = True

            # urceni vzdalenosti od noveho seedu
            dists, path = nx.single_source_dijkstra( self.G, linx, cutoff=maxd )

            #z rostouci vzdalenosti udelam klesajici (penalizacni) energii
            energyS = np.zeros( nrows*ncols )
            distsItemsArray = np.array(dists.items())
            energyS[ distsItemsArray[:,0].astype(np.uint32) ] = maxd - distsItemsArray[:,1]

            #vsechny body inicializuji max. vzdalenost maxd
            distLayer = initDistVal * np.ones( nrows*ncols )
            #priradim prislusne vzdalenosti
            distLayer[ distsItemsArray[:,0].astype(np.uint32) ] = distsItemsArray[:,1]#energyS.reshape( self.en.shape )
            self.distLayer = distLayer.reshape(self.en.shape)

            #pokud byl uz bod nekam prirazen s mensi vzdalenosti, tak se neprelabeluje
            maskedLayer = np.argmin( np.dstack( (self.distIm, self.distLayer) ), axis=2 ) == 1

            #urceni noveho labelu
            label = self.getLabel( dists, linx, method='intens' )
            # label = iteration + 1
            srcsen += energyS

            #update matice labelu a vzdalenosti
            self.labelIm = np.where( maskedLayer, label, self.labelIm )
            self.distIm = np.where( maskedLayer, self.distLayer, self.distIm )
            self.srclblD[linx] = label
            self.sourcesL.append( linx )
            # self.redraw( drawSources=True )

            #urceni curr_maxd
            curr_maxd = self.distIm.max()
            print '\tcurr_maxd = %f (maxd = %i)'%(curr_maxd, maxd)


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def automatic_lowest_energy( self, iterations=10 ):
        # iteration = 0
        srcsen = np.zeros( self.G.number_of_nodes() )
        used = np.ones( self.G.number_of_nodes(), dtype=np.bool )
        used[ np.ravel_multi_index( np.nonzero(self.maskroi), self.en.shape ) ] = False

        saveimgs = np.zeros( (self.en.shape[0],self.en.shape[1], iterations) )
        saveimgs2 = np.zeros( (self.en.shape[0],self.en.shape[1], iterations) )

        nrows, ncols = self.en.shape
        #bounds = np.ravel_multi_index( ((0,(nrows-1)/2,(nrows-1),(nrows-1)/2),((ncols-1)/2,(ncols-1),(ncols-1)/2,0)), self.en.shape)

        # iteration -= 5
        iteration = 0
        while iteration < iterations:
            print 'iteration #%i'%(iteration + 1)

            # if iteration < 4:
            #     linx = bounds[iteration]
            #     used[linx] = True
            # else:
            inds = np.arange(0, nrows*ncols, dtype=np.uint32)
            energyNotUsed = srcsen[np.logical_not(used)]
            indsNotUsed = inds[np.logical_not(used)]

            # minimalEnergyIndex = np.argmin(energyNotUsed)
            # linx = indsNotUsed[minimalEnergyIndex]
            # linx = self.get_new_seed(energyNotUsed, indsNotUsed)
            linx = self.get_new_seed_gradient_based(energyNotUsed, indsNotUsed, self.grad)
            used[linx] = True

            maxd = 150
            dists, path = nx.single_source_dijkstra( self.G, linx, cutoff=maxd )

            #bodum vzdalenejsim nez maxd se priradi hodnota cutedValue
            # cutedValue = maxd + 20
            # energy = cutedValue * np.ones( nrows*ncols )
            energyS = np.zeros( nrows*ncols )
            #z rostouci vzdalenosti udelam klesajici (penalizacni) energii
            distsItemsArray = np.array(dists.items())
            energyS[ distsItemsArray[:,0].astype(np.uint32) ] = maxd - distsItemsArray[:,1]

            distLayer = (maxd + 20) * np.ones( nrows*ncols )
            distLayer[ distsItemsArray[:,0].astype(np.uint32) ] = distsItemsArray[:,1]#energyS.reshape( self.en.shape )
            self.distLayer = distLayer.reshape(self.en.shape)

            maskedLayer = np.argmin( np.dstack( (self.distIm, self.distLayer) ), axis=2 ) == 1

            # label = self.getLabel( dists, linx, method='intens' )
            label = iteration + 1
            # if iteration > 0:
            #     srcsenO = srcsen.copy()
            #     srcsenN = srcsen + energyS
            #     srcsenDiff = srcsenN - srcsenO
            #     plt.figure(), plt.imshow(energyS.reshape(self.en.shape)), plt.title('energyS'), plt.colorbar()
            #     plt.figure(), plt.imshow(srcsenO.reshape(self.en.shape)), plt.title('srcsen pred'), plt.colorbar()
            #     plt.figure(), plt.imshow(srcsenN.reshape(self.en.shape)), plt.title('srcsen po'), plt.colorbar()
            #     plt.figure(), plt.imshow(srcsenDiff.reshape(self.en.shape)), plt.title('srcsen diff'), plt.colorbar()
            #     plt.show()
            srcsen += energyS

            self.labelIm = np.where( maskedLayer, label, self.labelIm )
            self.distIm = np.where( maskedLayer, self.distLayer, self.distIm )
            self.srclblD[linx] = label
            self.sourcesL.append( linx )
            # self.redraw( drawSources=True )

            saveimgs[:,:,iteration] = self.labelIm
            saveimgs2[:,:,iteration] = srcsen.reshape(self.en.shape)

            iteration += 1


        # print 'Saving images...'
        # coords = np.unravel_index( self.sourcesL, self.en.shape )
        # f = plt.figure()
        # for i in range(saveimgs.shape[2]):
        #     plt.imshow(saveimgs[:,:,i], vmin=0, vmax=saveimgs.max())
        #     plt.jet()
        #     plt.hold(True)
        #     # plt.plot( coords[1][4:i+5], coords[0][4:i+5], 'rx', markersize=6, markeredgewidth=2 )
        #     plt.plot( coords[1][0:i+1], coords[0][0:i+1], 'rx', markersize=6, markeredgewidth=2 )
        #     plt.hold(False)
        #     plt.axis('image')
        #     f.savefig( 'dijkstra_split_source_en2/im_%03i.png'%(i), bbox_inches='tight' )
        #     plt.clf()
        #
        #     plt.imshow(saveimgs2[:,:,i], vmin=0, vmax=saveimgs2.max())
        #     plt.jet()
        #     plt.hold(True)
        #     # plt.plot( coords[1][4:i+5], coords[0][4:i+5], 'rx', markersize=6, markeredgewidth=2 )
        #     plt.plot( coords[1][0:i+1], coords[0][0:i+1], 'rx', markersize=6, markeredgewidth=2 )
        #     plt.hold(False)
        #     plt.axis('image')
        #     f.savefig( 'dijkstra_split_source_en2/im_srcsen_%03i.png'%(i), bbox_inches='tight' )
        #     plt.clf()

            # plt.imshow(saveimgs2[:,:,-1], vmin=0, vmax=saveimgs.max())
            # plt.axis('image')
            # f.savefig( 'dijkstra_split_source_en2/im_srcsen_final_no_sources.png', bbox_inches='tight' )
            # plt.clf()
        # print '...done'

        plt.figure()
        plt.imshow(self.imroi.astype(np.uint8)),  plt.gray()
        plt.axis('image'), plt.axis('off'), plt.title('input')

        plt.figure()
        plt.imshow(saveimgs[:,:,-1], vmin=0, vmax=saveimgs.max()), plt.gray()
        plt.axis('image'), plt.axis('off'), plt.title('final')

        plt.figure()
        plt.imshow(saveimgs2[:,:,-1], vmin=0, vmax=saveimgs2.max()), plt.gray()
        plt.axis('image'), plt.axis('off'), plt.title('srcsen')

        plt.figure()
        plt.imshow( segmentation.mark_boundaries(self.imroi.astype(np.uint8), self.labelIm))
        plt.axis('image'), plt.axis('off'), plt.title('boundaries')

        plt.figure()
        plt.imshow( self.en ), plt.gray()
        plt.hold(True)
        coords = np.unravel_index( self.sourcesL, self.en.shape )
        plt.plot( coords[1], coords[0], 'wo', markersize=6, markeredgewidth=2 )
        plt.axis('image'), plt.axis('off'), plt.title('seeds in im')

        plt.figure()
        plt.imshow(self.grad_vec_norm.reshape(self.grad.shape)), plt.gray()
        plt.hold(True)
        coords = np.unravel_index( self.sourcesL, self.en.shape )
        plt.plot( coords[1], coords[0], 'wo', markersize=6, markeredgewidth=2 )
        plt.axis('image'), plt.axis('off'), plt.title('seeds in gradient')

        grad_en = sobel(saveimgs2[:,:,-1])
        grad_en_norm = cv2.normalize(grad_en, dst=None, alpha=0, beta=100, norm_type=cv2.NORM_MINMAX)
        plt.figure()
        plt.imshow( grad_en_norm), plt.gray()
        plt.axis('image'), plt.axis('off'), plt.title('energy gradient')

        # f1.savefig( 'dijkstra_split_source_en2/im_final_no_sources.png', bbox_inches='tight' )
        # f2.savefig( 'dijkstra_split_source_en2/im_srcsen_final_no_sources.png', bbox_inches='tight' )
        # f3.savefig( 'dijkstra_split_source_en2/im_boundaries.png', bbox_inches='tight' )
        plt.show()



#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def getLabel(self, dists, source, method='geom'):
        label = self.labelIm.max() + 1
        if method == 'geom':
            srcsDists = np.array( [dists[s] for s in self.sourcesL] ) #najde vzdalenost od vsech ostatnich sourcu
            #if len(srcsDists) != 0 and srcsDists.min() <= self.srcDistT:
            if len(srcsDists) != 0 and srcsDists.min() <= 15:
                i = srcsDists.argmin()
                label = self.srclblD[self.sourcesL[i]]
            # else:
            #     label = self.labelIm.max() + 1
        elif method == 'intens' and len(self.sourcesL) != 0:
            intens = self.en[np.unravel_index(source, self.en.shape)]
            #srcsDists = np.array( [self.en[np.unravel_index(s,self.en.shape)] for s in self.sourcesL] ) #najde vzdalenost od vsech ostatnich sourcu
            srcsDists = np.abs( intens - self.en[np.unravel_index(self.sourcesL, self.en.shape)] ) #najde vzdalenost od vsech ostatnich sourcu
            # if len(srcsDists) != 0 and srcsDists.min() <= 50:
            if len(srcsDists) != 0 and srcsDists.min() <= 15:
                i = srcsDists.argmin()
                label = self.srclblD[self.sourcesL[i]]
        return label


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def redraw( self, drawSources=False ):
        # plt.figure(self.fig.number)
        plt.subplot(121), plt.axis('off'), plt.gray()
        plt.imshow(self.en)
        plt.subplot(122), plt.axis('off'), plt.jet()
        plt.imshow(self.labelIm)

        if drawSources:
            #for s in self.sourcesL:
            coords = np.unravel_index( self.sourcesL, self.en.shape )
            plt.subplot(121), plt.hold(True), plt.axis('off')
            plt.plot( coords[1], coords[0], 'kx', markersize=12, markeredgewidth=5 )
            plt.hold(False), plt.axis('image')

            plt.subplot(122), plt.hold(True), plt.axis('off')
            plt.plot( coords[1], coords[0], 'kx', markersize=12, markeredgewidth=5 )
            plt.hold(False), plt.axis('image')

        self.fig.canvas.draw()
        # plt.show()
        pass


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def make_neighborhood_matrix(self, im, nghood=4):
        im = np.array( im, ndmin=3 )
        nslices, nrows, ncols = im.shape
        npts = nrows * ncols * nslices
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
    def create_graph( self, im, nghood=4, wtype=1 ):
        nghbm = self.make_neighborhood_matrix( im, nghood )
        nnodes = nghbm.shape[1]
        imv = np.reshape( im, nnodes ).astype(float)
        G = nx.Graph()
        #adding nodes
        G.add_nodes_from( range(nnodes) )
        #adding edges
        #sigma = imv.max() - imv.min()
        sigma = 10
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
        return G


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def get_new_seed(self, energy, nodesInds):
    # Logicky pristup, ale neskutecne neefektivni. Stejnou energii muze mit tisice bodu a pri 10. iteraci to znamena
    # pocitat obrovske mnozstvi nejkratsich cest jen pro urceni noveho seedu.
        inds = np.argwhere(energy==energy.min())

        #pokud je pouze jeden adept, je automaticky vybran
        if inds.shape[0] == 1:
            seedInd = inds[0,0]
            return seedInd

        if len(self.sourcesL) == 0:
            seedInd = inds[0,0]
            return seedInd

        dsts = np.zeros((inds.shape[0],len(self.sourcesL)))
        #spocitam vzdalenost kazdeho adepta od jiz existujicich zdroju
        for ind in range(inds.shape[0]):
            for src in range(len(self.sourcesL)):
                dst = nx.dijkstra_path_length(self.G, nodesInds[ind], self.sourcesL[src])
                dsts[ind, src] = dst

        #urcim stredni vzdalenost adeptu od existujicich zdroju
        meanDists = np.mean(dsts, axis=1)
        #jako novy seed je vybran seed ten, jehoz prumerna vzdalenost od ostatnich seedu je maximalni
        seedInd = meanDists.argmax()

        # seedInd = np.unravel_index(dsts.argmax(), dsts.shape)

        return seedInd


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def get_new_seed_gradient_based(self, energy, inds, grad):
    # Energii normalizovat na hodnotu <0, 100>, stejne tak i gradient. Udelat vazeny prumer a teprve pote vybrat novy
    # seed jako bod s nejnizsim prumerem. Diky tomu se zabrani vybirani seedu na hranach, ke kteremu casto dochazi.
    #     norm_en = 100 * (energy / energy.max())
        norm_en = cv2.normalize( energy, dst=None, alpha=0, beta=100, norm_type=cv2.NORM_MINMAX)
        # seeds_energy1 = cv2.addWeighted(norm_en, 0.5, self.grad_vec_norm[inds], 0.5, 0)
        # seeds_energy = 1./3 * norm_en + 1./3 * self.grad_vec_norm[inds] + 1./3 * self.grad_energy[inds]
        seeds_energy = 0.5 * norm_en + 0.5 * self.grad_vec_norm

        new_seed_ind = inds[seeds_energy.argmin()]
        return new_seed_ind, seeds_energy


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def get_gradient_penalty(self, im, scale=1):
        im2 = cv2.resize( im, dsize=(0,0), fx=scale, fy=scale )

        grad_x = cv2.Sobel(im2, ddepth=cv2.CV_16S, dx=1, dy=0, ksize=3)
        grad_y = cv2.Sobel(im2, ddepth=cv2.CV_16S, dx=0, dy=1, ksize=3)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        dst = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        # plt.imshow(dst), plt.show()

        dst = dst.reshape(im.shape)

        nPixels = im.shape[0] * im.shape[1]
        dst_vec = dst.reshape(nPixels)

        return dst, dst_vec

#----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pbc = PathBasins()
    pbc.run( 13 )