__author__ = 'tomas'

import matplotlib.pyplot as plt
import numpy as np
import numbers

class MyFigure(object):

    def __init__(self, data, title=None, vmin=0, vmax=255, win_cw=None, interpolation='nearest', cmap='gray',
                 int_range=False, colorbar=False, show=True):

        if isinstance(data, np.ndarray):
            data = tuple((data,))

        n_imgs = len(data)

        # converting tuple of ints to tuple of tuples
        if win_cw is not None:
            if isinstance(win_cw[0], numbers.Real):# or isinstance(win_cw[0], float):
                win_cw = n_imgs * tuple((win_cw,))
            elif len(win_cw) != n_imgs:
                raise AttributeError('Wrong length of win_cw attribute.')

        if int_range:
            vmin = [x.min() for x in data]
            vmax = [x.max() for x in data]
        if isinstance(vmin, int):
            vmin = n_imgs * tuple((vmin,))
        elif len(vmin) != n_imgs:
            raise AttributeError('Wrong length of win_cw attribute.')

        if isinstance(vmax, int):
            vmax = n_imgs * tuple((vmax,))
        elif len(vmax) != n_imgs:
            raise AttributeError('Wrong length of win_cw attribute.')

        if isinstance(cmap, str):
            cmap = n_imgs * tuple((cmap,))
        elif len(cmap) != n_imgs:
            raise AttributeError('Wrong length of cmap attribute.')

        rows_count = [1, 1, 2, 2, 2, 2, 3, 3, 3]
        cols_count = [1, 2, 2, 2, 3, 3, 3, 3, 3]

        n_rows = rows_count[n_imgs - 1]
        n_cols = cols_count[n_imgs - 1]

        if title is None:
            title = n_imgs * ((""),)#["" for i in range(n_imgs)]
        elif isinstance(title, str):
            title = n_imgs * tuple((title,))
        elif len(title) < n_imgs:
            for i in range(n_imgs - len(title)):
                title.append("")

        plt.figure()

        for i in range(n_imgs):
            im = data[i]
            if win_cw is not None:
                im = self.window_img(im, *win_cw[i])
            plt.subplot(n_rows, n_cols, i+1)

            plt.imshow(im, cmap[i], interpolation=interpolation, vmin=vmin[i], vmax=vmax[i])
            if colorbar:
                plt.colorbar()
            plt.title(title[i])

        if show:
            plt.show()

    def get_vmin_vmax(self, c, w):
        vmin = c - w / 2
        vmax = c + w / 2
        return vmin, vmax

    def window_img(self, im, c, w):
        if w > 0:
           mul = 255. / float(w)
        else:
            mul = 0

        lb = c - w / 2
        im_w = (im - lb) * mul
        im_w = np.where(im_w < 0, 0, im_w)
        im_w = np.where(im_w > 255, 255, im_w)

        return im_w

if __name__ == '__main__':
    im = np.array([[0, 80, 160, 255],
                   [255, 0, 80, 160],
                   [160, 255, 0, 80],
                   [80, 160, 255, 0]])

    # MyFigure(im, title='test')
    MyFigure(im, title='test', win_cw=(125, 12))