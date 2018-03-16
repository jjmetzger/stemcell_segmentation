import sys
import os
import re
import h5py
import warnings
import matplotlib as mpl
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
import skimage
from scipy.ndimage.filters import maximum_filter, gaussian_filter
# from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from skimage.morphology import ball, disk, binary_erosion, remove_small_objects, binary_dilation, binary_closing, dilation
# from skimage.segmentation import random_walker
import numpy as np
import skimage.io as io
import matplotlib.pylab as plt
import pandas as pd
from matplotlib.pyplot import rcParams
from scipy.ndimage.measurements import center_of_mass  # note this gives center of mass in column, row format, i.e. to plot need to reverse
from distutils.version import LooseVersion
from matplotlib.colors import LogNorm
from skimage.filters import threshold_otsu
from ipywidgets import interact
from numba import jit
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib

# import seaborn as sns

io.use_plugin('tifffile')
# sys.path.append('/Users/jakob/Documents/RU/Code/segment')


class Ws3d(object):
    """
    Main watershed 3d class. Takes an image file and expects a _Probabillites.h5 file from Ilastik. If there is an
    object classifier from Ilastik, uses this for seeding. Otherwise tries to guess blobs for seeding.
    """

    def __init__(self, imagefile, xy_scale=1.0, z_scale=1.0, z=None, verbose=False):
        """
        Initializes Ws3d with an imagefile. Can set the relative x,y and z scales of the z-stack, which is important
        for the blob
        detection.

        :param imagefile: File to segment
        :param xy_scale: scale in micrometer of the x and y pixels in microns/pixel, e.g. have 800 pixels for 1000um this should be 800/1000
        :param z_scale: scale in micrometer of z pixels in microns/pixel

        """

        self.z = z
        self.fontsize = 16
        self.filename = imagefile
        self.image_stack = io.imread(imagefile)
        self.image_dim = self.image_stack.ndim
        if self.image_dim == 2:
            if verbose:
                print('image is two-dimensional')
            self.x_size, self.y_size = self.image_stack.shape
            self.z_size, self.z_scale = None, None

        elif self.image_dim == 3:
            if verbose:
                print('image is three-dimensional')
            self.z_size, self.x_size, self.y_size = self.image_stack.shape
            if self.z_size > self.x_size or self.z_size > self.y_size:
                raise RuntimeError("ERROR: z must be the first and smallest dimension")
            self.z_scale = z_scale

        else:
            raise RuntimeError("Image is neither 2 or 3 dimensional")

        self.xy_scale = xy_scale
        # self.center_of_mass = None
        # self.radius_of_gyration = None
        self.peaks = None
        self.peak_array = None
        self.probability_map = None
        self.ws = None
        self.df = None
        self.labels_cyto = None
        # self.good_nuclei = None
        self.center = None
        self.cyto_size = None
        # self.channels = {}
        self.channels_image = {}
        p_re = re.compile('\.tif{1,2}$', re.I)
        if p_re.search(self.filename) is None:
            raise RuntimeError('did not recognize file ' + self.filename + ' as tif file')
        self.probability_map_filename = p_re.sub('_Probabilities.h5', self.filename, re.I)
        if not os.path.isfile(self.probability_map_filename):
            # raise RuntimeWarning('WARNING: file', self.probability_map_filename, 'not found. Did you do the Ilastik '
            #                                                                  'classification? Using Otsu thresholding instead')
            raise RuntimeError('Error: file', self.probability_map_filename, 'not found. Did you do the Ilastik '
                                                                             'classification?')
        else:
            if verbose:
                print('found probability map', self.probability_map_filename)
        self.mask = None
        self.mask_selected = None

        # try to locate object prediction
        self.filename_op = p_re.sub('_Object Predictions.h5', self.filename, re.IGNORECASE)
        self.op = None
        self.have_op = True
        if not os.path.isfile(self.filename_op):
            # print('No object prediction file ({:s}) found - segmenting without.'.format(self.filename_op))
            self.have_op = False

    def load_mask(self, method='ilastik', prob=0.5, foreground_index=1, verbose=False):
        """
        Load the probability mask and object prediction if it exists.

        :param prob: Set the probability for which
        :param foreground_index: Set the index that contains the foreground (i.e. did you use the first or the second label in Ilastik for the background? Default is that the first label (i.e. 0) is background and 1 corresponds to foreground.
        """

        if method == 'ilastik':
            if self.image_dim == 3:
                with h5py.File(self.probability_map_filename, 'r') as h:  # r+: read/write, file must exist
                    self.probability_map = h['exported_data'][:][:, :, :, foreground_index]  # index 1 is the nuclei
            elif self.image_dim == 2:
                with h5py.File(self.probability_map_filename, 'r') as h:  # r+: read/write, file must exist
                    self.probability_map = h['exported_data'][:][:, :, foreground_index]  # index 1 is the nuclei
            else:
                raise RuntimeError('unkonwn dimension')

            if verbose:
                print("shape", self.probability_map.shape, self.image_stack.shape)
            if not self.probability_map.shape == self.image_stack.shape:
                print("ERROR: probability map does not have same dimensions as image", self.probability_map.shape,
                      self.image_stack.shape)
                return

            self.mask = self.probability_map > prob
            if verbose:
                print('loaded probability map')
            # self.probability_map[self.probability_map < prob] = 0.

            # load object prediction if there
            if self.have_op:
                if self.image_dim == 3:
                    with h5py.File(re.sub('\.tif$', '_Object Predictions.h5', self.filename, re.IGNORECASE), 'r') as h:
                        self.op = np.swapaxes(np.squeeze(h['exported_data']), 2, 0)  # object prediction
                    if verbose:
                        print('loaded object prediction')
                else:
                    raise NotImplementedError("not implemented for 2d")

            if self.z is not None:
                print('selecting z =' + str(self.z))
                self.image_dim = 2
                self.image_stack = self.image_stack[self.z]
                self.mask = self.mask[self.z]
                self.probability_map = self.probability_map[self.z]

        elif method == 'otsu':
            otsu = threshold_otsu(self.image_stack)
            self.mask = self.image_stack > otsu

        else:
            raise RuntimeError('unknown method')

        self.find_center_of_colony()

    def plot_probability_map(self, z=None, contrast_stretch=False, figsize=None):

        if self.probability_map is None:
            if self.mask is None:
                raise RuntimeError('neither probability map nor mask loaded, cannot plot - use load_mask first')
            else:
                print('do not have probability map, plotting mask instead')
                self.grid_plot(self.mask, z, contrast_stretch, figsize)
        else:
            self.grid_plot(self.probability_map, z, contrast_stretch, figsize)


    def grid_plot(self, z=None, contrast_stretch=False, figsize=None, cmap=None, return_fig=False):
        """
        Plot all z-slices

        :param image_to_plot:
        :param z: can select a specific z to plot. Ignore for 2D
        :param contrast_stretch: enhance contrast
        :param figsize: set the size of the figure when plotting a single z-slice, default is (10,10)
        :param cmap: colormap
        :return:
        """

        image_to_plot = self.image_stack

        if z is not None and image_to_plot.ndim == 2:
            print("Warning: Gave z-information but only have 2d image")
            z = None

        if cmap is None:
            cmap = plt.cm.plasma
        # cmap = plt.cm.Greys
        if figsize is None:
            figsize = (10, 10)

        if image_to_plot.ndim == 2:
            fig, axes = plt.subplots(figsize=figsize)
            if contrast_stretch:
                axes.imshow(self._contrast_stretch(image_to_plot), cmap=cmap)
            else:
                axes.imshow(image_to_plot, interpolation='None', cmap=cmap)
        else:  #3d

            if z is not None:
                fig, axes = plt.subplots(figsize=figsize)
                if contrast_stretch:
                    axes.imshow(self._contrast_stretch(image_to_plot[z]), cmap=cmap)
                else:
                    ax = axes.imshow(image_to_plot[z], interpolation='None', cmap=cmap)
                    ax.set_clim(0, image_to_plot[z].max())
            else:

                nrows = np.int(np.ceil(np.sqrt(self.z_size)))
                ncols = np.int(self.z_size // nrows + 1)

                fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
                peak_counter = 1
                for n in range(self.z_size):
                    i = n // ncols
                    j = n % ncols

                    if contrast_stretch:
                        axes[i, j].imshow(self._contrast_stretch(image_to_plot[n]), interpolation='None', cmap=cmap)
                    else:
                        ax = axes[i, j].imshow(image_to_plot[n], interpolation='None', cmap=cmap)
                        ax.set_clim(0, image_to_plot.max())

                    # show seeds of watershed. TODO: Do it on the dataframe since may have deleted cells
                    # seeds_in_current_z = self.peaks[self.peaks[:, 0] == n][:,1:] # find seeds that are in the current z
                    # axes[i, j].plot(seeds_in_current_z[:, 1], seeds_in_current_z[:, 0], 'xr')

                    # # label seeds
                    # for ipeaks in range(seeds_in_current_z.shape[0]):
                    #     axes[i, j].text(seeds_in_current_z[ipeaks, 1], seeds_in_current_z[ipeaks, 0], str(peak_counter), color='r', fontsize=22)
                    #     peak_counter += 1

                    # # if show_labels:
                    assert len(self.df) == self.peaks.shape[0]
                    for ipeaks in range(len(self.df)):
                        if int(np.round(self.df.iloc[ipeaks].centroid[0])) == n:
                            axes[i,j].text(self.df.iloc[ipeaks].centroid[2], self.df.iloc[ipeaks].centroid[1],
                                       str(self.df.iloc[ipeaks].label),
                                       color='r', fontsize=22)
                            axes[i,j].plot(self.df.iloc[ipeaks].centroid[2], self.df.iloc[ipeaks].centroid[1], 'xr')

                    axes[i, j].set_xlim(self.peaks[:, 2].min() - 20, self.peaks[:, 2].max() + 20)
                    axes[i, j].set_ylim(self.peaks[:, 1].min() - 20, self.peaks[:, 1].max() + 20)

                # Remove empty plots
                for ax in axes.ravel():
                    if not (len(ax.images)):
                        fig.delaxes(ax)
                fig.tight_layout()

        if return_fig:
            plt.close(fig)
            return fig

    def intensity_histogram(self):

        with plt.style.context('ggplot'):
            fig, ax = plt.subplots(figsize=(8, 4))
            flat = self.image_stack.flatten()
            ax.hist(flat, log=True, bins=np.arange(0, flat.max(), 200), range=(0, flat.max()))

            _ = ax.set_title('Min value: %i, Max value: %i, Image shape: %s \n'
                             % (self.image_stack.min(),
                                self.image_stack.max(),
                                self.image_stack.shape))

    def _filter_probability_map(self, sigma=None):
        """
        Filter the probability map using a gaussian filter. Use the dimensions provided.

        :param sigma: Width of the gaussian filter. Needs to have len(sigma) == 3.
        :return: filtered probability map
        """

        if self.image_dim == 3:
            try:
                assert len(sigma) == 3
            except AssertionError:
                print('sigma needs to have 3 elements for the three dimensions.')
                return
            except TypeError:  # None
                print('setting sigma to default (2, 6, 6)')
                sigma = (2, 6, 6)

        elif self.image_dim == 2:
            try:
                assert len(sigma) == 2
            except AssertionError:
                print('sigma needs to have 2 elements for the two dimensions.')
                return
            except TypeError:  # None
                print('setting sigma to default (6, 6)')
                sigma = (6, 6)

        else:
            raise RuntimeError('wrong dimensions')

        if self.probability_map is not None:
            pm = self.probability_map.copy()
        else:
            pm = self.mask.copy()
        # print(pm.shape)
        pm = gaussian_filter(pm, sigma=sigma)
        # normalize back to [0,1]
        pm[pm > 1] = 1.
        pm[pm < 0.05] = 0.
        return pm

    # def segment_each_z(self, z, min_distance=2, sigma=None, do_not_use_object_classifier=True,
    #                    opensize_small_objects=10, remove_small_nuclei=False, cyto_size=None):
    #     """
    #     Segment the image.
    #
    #     :param z: if have 3d image, can give z or range of z to be segmentent as 2d images
    #     :param min_distance:
    #     :param sigma: default is (2, 6, 6) for 3d or (6, 6) for 2d
    #     :param do_not_use_object_classifier:
    #     """
    #
    #     if z is not None:
    #         if self.image_dim != 3:
    #             raise RuntimeError('gave z but do not have a 3d image')
    #         else:
    #             try:
    #                 len(z)
    #             except TypeError:
    #                 z = [z]
    #
    #     if self.probability_map is None:
    #         print('ERROR: probability map not loaded, cannot segment - use load_mask first')
    #         return
    #
    #     if sigma is None:
    #         if self.image_dim == 3:
    #             sigma = (2, 2, 6)
    #         elif self.image_dim == 2:
    #             sigma = (6, 6)
    #         else:
    #             raise NotImplementedError
    #         print('set sigma =', sigma)
    #     else:
    #         if not self.image_dim == len(sigma):
    #             raise RuntimeError('sigma needs to be same dimension as image')
    #
    #     # have object classifier or don't
    #     if self.have_op and not do_not_use_object_classifier:
    #         raise NotImplementedError
    #
    #     # NO object classifier, use smoothed map and find peaks in probability map
    #     else:
    #         # get smoothed probability map
    #         pm = self._filter_probability_map(sigma=sigma)
    #
    #         for zi in z:
    #
    #             self.peak_array = peak_local_max(pm[zi], min_distance=min_distance, indices=False)
    #             self.peaks = np.transpose(np.nonzero(self.peak_array))  # same as above with indices True, but need that too
    #
    #             markers = ndi.label(self.peak_array)[0]
    #
    #             self.ws = skimage.morphology.watershed(-self.image_stack[zi], markers, mask=self.mask[zi])
    #
    #             if remove_small_nuclei:
    #                 self.ws = remove_small_objects(self.ws, min_size=opensize_small_objects)
    #                 self.peak_array *= self.ws > 0  # keep only the watershed seeds that are on top of data that has not been
    #                 # removed
    #                 # as small object
    #                 self.peaks = np.transpose(np.nonzero(self.peak_array))
    #
    #             # cytoplasm
    #             if cyto_size is not None:
    #                 selem = disk(cyto_size)
    #                 extended_mask = binary_dilation(self.ws, selem=selem)
    #                 self.labels_cyto = skimage.morphology.watershed(-self.image_stack[zi], markers, mask=extended_mask)
    #                 self.labels_cyto[self.ws > 0] = 0
    #                 # self.perimeter_cyto = np.ma.masked_where(self.labels_cyto < 1, self.labels_cyto)
    #                 # border = binary_erosion(self.labels, selem=disk(8))
    #                 # border = np.logical_xor(self.labels, border)
    #                 # self.perimeter = np.ma.masked_where(~border, self.labels)
    #
    #             # also do cytoplasm
    #             self.df = self._regionprops_to_dataframe(self.ws, self.image_stack, self.labels_cyto)
    #
    #             print('segmentation done, found', self.peaks.shape[0], 'cells')

    def segment(self, min_distance=2, sigma=None, do_not_use_object_classifier=True, opensize_small_objects=10,
                remove_small_nuclei=False, cyto_size=None, compactness=0.01, verbose=False):
        """
        Segment the image.

        :param min_distance:
        :param sigma: default is (2, 6, 6) for 3d or (6, 6) for 2d
        :param do_not_use_object_classifier:
        :param z: if have 3d image, can give z or range of z to be segmentent as 2d images
        """

        self.cyto_size = cyto_size

        if self.probability_map is None and self.mask is None:
            print('ERROR: neither probability map or mask loaded, cannot segment - use load_mask first')
            return

        if sigma is None:
            if self.image_dim == 3:
                sigma = (2, 2, 6)
            elif self.image_dim == 2:
                sigma = (6, 6)
            else:
                raise NotImplementedError
            print('set sigma =', sigma)
        else:
            if not self.image_dim == len(sigma):
                raise RuntimeError('sigma needs to be same dimension as image')

        # have object classifier or don't
        if self.have_op and not do_not_use_object_classifier:
            raise NotImplementedError

        # NO object classifier, use smoothed map and find peaks in probability map
        else:
            # if min_distance is None:
            #     min_distance = 2

            # get smoothed probability map
            pm = self._filter_probability_map(sigma=sigma)

            if self.probability_map is None:
                distance = ndi.distance_transform_edt(self.mask)
                self.peak_array = peak_local_max(distance, min_distance=min_distance, indices=False)
            else:
                self.peak_array = peak_local_max(pm, min_distance=min_distance, indices=False)

            self.peak_array = relabel_peaks(self.peak_array)

            self.peaks = np.transpose(np.nonzero(self.peak_array))  # same as above with indices True, but need that too
            markers = ndi.label(self.peak_array)[0]
            # self.markers = markers

            # check for version because compactness is only available in watershed > 0.12
            if LooseVersion(skimage.__version__) > LooseVersion('0.13'):
                self.ws = skimage.morphology.watershed(pm, markers, mask=self.mask, compactness=compactness)
                # print('using compactness')
            else:
                self.ws = skimage.morphology.watershed(pm, markers, mask=self.mask)

            if remove_small_nuclei:
                self.ws = remove_small_objects(self.ws, min_size=opensize_small_objects)
                self.peak_array *= self.ws > 0  # keep only the watershed seeds that are on top of data that has not been
                # removed
                # as small object
                self.peaks = np.transpose(np.nonzero(self.peak_array))

            # cytoplasm
            if cyto_size is not None:
                if self.image_dim == 2:
                    selem = disk(cyto_size)
                else:
                    selem = ball(cyto_size)

                # extended_mask = binary_dilation(self.mask, selem=selem)
                # self.em = extended_mask
                # self.labels_cyto = skimage.morphology.watershed(pm, ndi.label(self.peak_array)[0], mask=self.mask, compactness=compactness)
                self.labels_cyto = dilation(self.ws, selem=selem)
                self.labels_cyto[self.ws > 0] = 0
                # self.perimeter_cyto = np.ma.masked_where(self.labels_cyto < 1, self.labels_cyto)
                # border = binary_erosion(self.labels, selem=disk(8))
                # border = np.logical_xor(self.labels, border)
                # self.perimeter = np.ma.masked_where(~border, self.labels)

            # also do cytoplasm


            # self.ws = mh.cwatershed(-self.image_stack,markers)
            # self.ws *= self.mask

            # make a dataframe with some of the regionprops in it
            # rp = skimage.measure.regionprops(self.ws, intensity_image=self.image_stack)
            # columns = ('area', 'total_intensity', 'mean_intensity', 'centroid')
            # rpd = [[i1.area, i1.mean_intensity * i1.area, i1.mean_intensity, i1.coords.mean(axis=0)] for i1 in rp]
            # indices = [i1.label for i1 in rp]
            # indices = pd.Index(indices, name='cell_id')
            # self.df = pd.DataFrame(rpd, index=indices, columns=columns)
            self.df = self._regionprops_to_dataframe(self.ws, self.image_stack, self.labels_cyto, xyscale=self.xy_scale, zscale=self.z_scale)
            # self.center_of_mass = np.vstack(self.df.centroid_rescaled).mean(axis=0)
            # self.radius_of_gyration = radius_of_gyration()
            # self.center_of_mass, self.radius_of_gyration = radius_of_gyration(self.df.centroids_rescaled)

            if verbose:
                print('segmentation done, found', self.peaks.shape[0], 'cells')

    @staticmethod
    def _regionprops_to_dataframe(ws, image_stack, cyto=None, average_method=np.median, xyscale=1., zscale=1.):
        """
        Return pd.DataFrame with relevant entries from watershed image. Keep static so can use it for other images as
        well.

        :param ws: watershed array
        :param image_stack: image array
        :param cyto: cytoplasm
        :return: pd.DataFrame with entries ('area', 'total_intensity', 'mean_intensity', 'centroid') and index 'cell_id'
        """

        if not (np.median == np.array([np.max, np.median, np.sum])).any():
            raise TypeError('average_method must be one of np.max, np.median or np.sum')

        rp = skimage.measure.regionprops(ws, intensity_image=image_stack)

        if cyto is None:
            columns = ('area', 'total_intensity', 'mean_intensity', 'centroid', 'centroid_rescaled', 'label')
            # rpd = [[i1.area, i1.mean_intensity * i1.area, i1.mean_intensity, i1.coords.mean(axis=0)] for i1 in rp]
            rpd = [[i1.area, average_method(image_stack[i1.coords[:,0], i1.coords[:,1], i1.coords[:,2]]), i1.mean_intensity, i1.coords.mean(axis=0), i1.coords.mean(axis=0)/np.array([zscale, xyscale, xyscale]), i1.label] for i1 in rp]

            indices = [i1.label for i1 in rp]
            indices = pd.Index(indices, name='cell_id')
            # print('len rp=', len(rp))
            return pd.DataFrame(rpd, index=indices, columns=columns)
        else:
            rp_cyto = skimage.measure.regionprops(cyto, intensity_image=image_stack)
            columns = ('area', 'total_intensity', 'mean_intensity', 'centroid', 'centroid_rescaled', 'label', 'area_cyto', 'total_intensity_cyto',
                       'mean_intensity_cyto')
            # rpd = [[i1.area, i1.mean_intensity * i1.area, i1.mean_intensity, i1.coords.mean(axis=0)] for i1 in rp]
            rpd = [[i1.area, average_method(image_stack[i1.coords[:,0], i1.coords[:,1], i1.coords[:,2]]), i1.mean_intensity, i1.coords.mean(axis=0), i1.coords.mean(axis=0)/np.array([zscale, xyscale, xyscale]), i1.label] for i1 in rp]
            # rpd_cyto = [[i1.area, i1.mean_intensity * i1.area, i1.mean_intensity, i1.label] for i1 in rp_cyto]
            rpd_cyto = [[i1.area, average_method(image_stack[i1.coords[:,0], i1.coords[:,1], i1.coords[:,2]]), i1.mean_intensity, i1.label] for i1 in rp_cyto]

            # indices = [i1.label for i1 in rp]
            # indices_cyto = [i1.label for i1 in rp_cyto]
            # indices = pd.Index(indices, name='cell_id')
            # print('len rp=', len(rp))

            columns1 = ('area', 'total_intensity', 'mean_intensity', 'centroid', 'centroid_rescaled', 'label')
            columns2 = ('area_cyto', 'total_intensity_cyto', 'mean_intensity_cyto', 'label')
            df1 = pd.DataFrame(rpd, columns=columns1)
            df2 = pd.DataFrame(rpd_cyto, columns=columns2)

            # need to remove those without cytoplasm

            # return pd.DataFrame(pd.concat([df1, df2], axis=1), index=indices, columns=columns)
            return pd.merge(df1,df2, on='label')
            # return pd.DataFrame(rpd, index=indices, columns=columns, axis=1)

    def show_segmentation(self, z=None, contrast_stretch=True, figsize=None, seed=130, show_labels=False, title=None, plot_cyto=False):
        """
        Show segmentation on the maximum intensity projection or per z-slice

        :param z: plot only this z
        :param contrast_stretch: enhance contrast for better visibility
        :param figsize: figure size
        :param seed:  seed for the random color map
        """

        if self.peaks is None:
            print('do not have cell positions, run segment first')

        if figsize is None:
            figsize = (12, 12)
        # self.grid_plot(self.image_stack,  z=z, contrast_stretch=contrast_stretch, figsize=figsize)
        fig, ax = plt.subplots(1, 2, figsize=figsize)

        # show seeds on the original image
        if self.image_dim == 3:
            if z is None:
                # plot the maximum intensity projection
                mip = self.image_stack.max(axis=0)
            else:
                mip = self.image_stack[z]
            if contrast_stretch:
                ax[0].imshow(self._contrast_stretch(mip), cmap=plt.cm.viridis)
            else:
                ax[0].imshow(mip, cmap=plt.cm.viridis)
        elif self.image_dim == 2:
            if contrast_stretch:
                ax[0].imshow(self._contrast_stretch(self.image_stack), cmap=plt.cm.viridis)
            else:
                ax[0].imshow(self.image_stack, cmap=plt.cm.viridis)
        else:
            raise NotImplementedError

        if self.image_dim == 3:
            if z is None:
                if title is None:
                    ax[0].set_title('maximum intensity projection')
                else:
                    ax[0].set_title(title)
            else:
                ax[0].set_title('z = ' + str(z))
            ax[0].plot(self.peaks[:, 2], self.peaks[:, 1], 'xr')
            ax[0].set_xlim(self.peaks[:, 2].min() - 20, self.peaks[:, 2].max() + 20)
            ax[0].set_ylim(self.peaks[:, 1].min() - 20, self.peaks[:, 1].max() + 20)
        else:
            ax[0].plot(self.peaks[:, 1], self.peaks[:, 0], 'xr')
            ax[0].set_xlim(self.peaks[:, 1].min() - 20, self.peaks[:, 1].max() + 20)
            ax[0].set_ylim(self.peaks[:, 0].min() - 20, self.peaks[:, 0].max() + 20)

        # show watershed. Do it on the dataframe since may have deleted cells
        if self.image_dim == 3:
            if z is None:
                ax[1].imshow(self.ws.max(axis=0), cmap=self.myrandom_cmap(seed=seed), vmin=0,vmax=1024)
                # if show_labels:
                #     peak_counter = 1
                #     for ipeaks in range(self.peaks.shape[0]):
                #         ax[1].text(self.peaks[ipeaks, 2], self.peaks[ipeaks, 1], str(peak_counter),
                #                         color='r', fontsize=22)
                #         peak_counter += 1
                if show_labels:
                    assert len(self.df) == self.peaks.shape[0]
                    for ipeaks in range(len(self.df)):
                        ax[1].text(self.df.iloc[ipeaks].centroid[2], self.df.iloc[ipeaks].centroid[1], str(self.df.iloc[ipeaks].label),
                                        color='r', fontsize=22)

            else:
                if plot_cyto and self.cyto_size is not None:
                    ax[1].imshow(np.ma.masked_equal(self.labels_cyto[z], 0), cmap=random_cmap(seed=123, return_darker=1)[1], vmin=0,vmax=1024)
                    ax[1].imshow(np.ma.masked_equal(self.ws[z], 0), cmap=random_cmap(seed=123), vmin=0,vmax=1024)
                    # ax[1].imshow(self.ws[z], cmap=self.myrandom_cmap(seed=seed, return_darker=1))
                    # ax[1].imshow(self.labels_cyto[z], cmap=self.myrandom_cmap(seed=seed))
                else:
                    ax[1].imshow(self.ws[z], cmap=self.myrandom_cmap(seed=seed, return_darker=1)[0], vmin=0,vmax=1024)


            ax[1].plot(self.peaks[:, 2], self.peaks[:, 1], 'xr')
            ax[1].set_xlim(self.peaks[:, 2].min() - 20, self.peaks[:, 2].max() + 20)
            ax[1].set_ylim(self.peaks[:, 1].min() - 20, self.peaks[:, 1].max() + 20)

        else:
            ax[1].imshow(self.ws, cmap=self.myrandom_cmap(seed=seed), vmin=0,vmax=1024)
            ax[1].plot(self.peaks[:, 1], self.peaks[:, 0], 'xr')
            ax[1].set_xlim(self.peaks[:, 1].min() - 20, self.peaks[:, 1].max() + 20)
            ax[1].set_ylim(self.peaks[:, 0].min() - 20, self.peaks[:, 0].max() + 20)

        # return fig


    def write_image_with_seeds(self, filename='image_with_seeds.tif'):
        """
        Write an image to tif file readable by ImageJ, with the seeds as bright white dots.
        :param filename: filename to write to
        :return: None
        """

        to_write = self.image_stack.copy()

        # below would be RGB
        # a = np.zeros((*w.image_stack.shape, 3), dtype=w.image_stack.dtype)
        # a[:, :, :, 0] = w.image_stack
        # a[:, :, :, 1] = w.image_stack
        # a[:, :, :, 2] = w.image_stack
        # a[w.peak_array, 0] = np.iinfo(w.image_stack.dtype).max

        # to_write[self.peaks[:, 0], self.peaks[:, 1], self.peaks[:, 2]] = np.iinfo(self.image_stack.dtype).max
        dilated_peak_array = binary_dilation(self.peak_array, selem=ball(3))
        to_write[dilated_peak_array] = np.iinfo(self.image_stack.dtype).max
        io.imsave(filename, to_write)


    def select_nuclei(self, quantiles=[0.1, 0.9], cutoff=None, plot=True, z=None, seed=130):
        """
        Select nuclei based on their size. Use quantiles or a hard cutoff. Cutoff overrides quantiles.

        :param quantiles: Quantiles (default [0.1, 0.9]).
        :param cutoff: [lower, upper] cutoff. If specified, overrides quantiles!
        :param plot: plot distributions before and after
        """

        assert self.df is not None
        # df_before = self.df

        if cutoff is not None:
            if len(cutoff) != 2:
                print("ERROR, len(cutoff) must be 2 (lower and upper cutoff).")
                return
            else:
                good_nuclei = (self.df.area > cutoff[0]) & (self.df.area < cutoff[1])
                # self.df = self.df[(self.df.area > cutoff[0]) & (self.df.area < cutoff[1])]

        else:
            lims = [self.df.area.quantile(quantiles[0]), self.df.area.quantile(quantiles[1])]
            good_nuclei = (self.df.area > lims[0]) & (self.df.area < lims[1])
            # self.df = self.df[(self.df.area > lims[0]) & (self.df.area < lims[1])]

        # good nuclei is a Series, not a DataFrame
        good_nuclei.name = 'good_nuclei'
        # check whether already exists
        try:
            self.df.drop('good_nuclei', axis=1, inplace=1)
        except ValueError: # does not exist
            pass

        self.df = pd.concat([self.df, good_nuclei], axis=1)
        if plot:
            # _, ax = plt.subplots()

            # a_heights, a_bins = np.histogram(df_before.area, 50)
            # b_heights, b_bins = np.histogram(self.df.area, bins=a_bins)
            #
            # width = (a_bins[1] - a_bins[0]) / 3
            #
            # ax.bar(a_bins[:-1], a_heights, width=width, facecolor='cornflowerblue')
            # ax.bar(b_bins[:-1] + width, b_heights, width=width, facecolor='seagreen')

            # df_before.hist(color='k', bins=50, alpha=0.6)
            ax1 = self.df.hist(color='k', bins=50, alpha=0.6, xlabelsize=10)
            for ax in ax1.ravel():
                ax.set_xticks(ax.get_xticks()[::2])
            plt.suptitle('before selection of nuclei')
            #plt.tight_layout()

            ax1 = self.df[self.df.good_nuclei].hist(color='k', bins=50, alpha=0.6, xlabelsize=10)
            for ax in ax1.ravel():
                ax.set_xticks(ax.get_xticks()[::2])
            plt.suptitle('after selection of nuclei')
            #plt.tight_layout()



            _, ax = plt.subplots()
            w2 = self.ws.copy()  # note the copy here, otherwise next line also affects self.ws
            # w2[~np.in1d(self.ws, np.array(self.df[self.good_nuclei].index)).reshape(self.ws.shape)] = 0 # set to 0
            w2[~np.in1d(self.ws, np.array(self.df[self.df['good_nuclei']].index)).reshape(self.ws.shape)] = 0 # set to 0
            self.mask_selected = w2 > 0
            # all elements that are NOT in the good nuclei

            # show watershed after selections.
            if self.image_dim == 3:
                if z is None:
                    ax.imshow(w2.max(axis=0), cmap=self.myrandom_cmap(seed=seed), origin='lower', vmin=0,vmax=1024)
                else:
                    ax.imshow(w2[z], cmap=self.myrandom_cmap(seed=seed), origin='lower', vmin=0,vmax=1024)
            else:
                ax.imshow(w2, cmap=self.myrandom_cmap(seed=seed), origin='lower', vmin=0,vmax=1024)
            ax.set_title('after selection of good nuclei')

    def apply_to_channels(self, filename, channel_id, remove_background=False, average_method=np.median):
        """
        Apply nuclear marker to other channels.

        :param filename:
        :param channel_id: ID for this channel. Can be a number or a string, e.g. 'Sox17'.
        :param remove_background:
        :return:
        """
        if self.ws is None:
            print('ERROR: run segment first')
            return

        im = io.imread(filename)
        if remove_background:
            im = remove_background_func(im)

        if self.ws.shape != im.shape:
            raise RuntimeError('image shapes are not the same, original is ' + str(self.ws.shape) + ' and to be '
                                'superimpose is ' + str(im.shape))
        # self.channels[channel_id] = self._regionprops_to_dataframe(self.ws, im)
        channel_df = self._regionprops_to_dataframe(self.ws, im, average_method=average_method, xyscale=self.xy_scale, zscale=self.z_scale)
        # channel_df.columns = ['area', channel_id, 'mean_intensity', 'centroid']
        channel_df.columns = ['area', channel_id, 'mean_intensity', 'centroid', 'centroid_rescaled', 'label']

        # remove channel if it exists. This happens when reapplying an image to an updated segmentation.
        try:
            self.df.drop(channel_id, axis=1, inplace=1)
        except ValueError:  # does not exist
            pass

        self.df = pd.concat([self.df, channel_df[channel_id]], axis=1) # only take the channeld_id channel, i.e. total intensity
        self.channels_image[channel_id] = im

        normed_id = channel_id + '_norm'
        try:
            self.df.drop(normed_id, axis=1, inplace=1)
        except ValueError:  # does not exist
            pass

        self.df = pd.concat([self.df, self.df[channel_id] / self.df.mean_intensity], axis=1).rename(columns={0: normed_id})

    def find_center_of_colony(self):
        """
        find center of mass
        """

        # self.center = np.array(center_of_mass(np.abs(self.image_stack)))
        if self.image_dim == 2:
            center_xy = center_of_mass(binary_closing(self.mask, selem=disk(10)))
        else:
            center_xy = center_of_mass(binary_closing(self.mask.max(axis=0), selem=disk(10)))

        self.center = np.array([0., center_xy[0], center_xy[1]])
        # take middle of stack        
#         middle = int(np.round(self.image_stack.shape[0] /2))
#         center_xy = np.array(center_of_mass(np.abs(self.image_stack[middle])))
#         self.center = np.array([0., center_xy[0], center_xy[1]])
        
        

    def _get_indices(self, only_selected_cells):
        if only_selected_cells:
            return self.df.good_nuclei
        else:
            return self.df.index  # all cells

    def copy_data_to_clipboard(self):
        """
        copy data to clipboard

        """
        self.df.to_clipboard()

    ###################################
    ####### END OF SEGMENTATION #######
    ###################################

    ###################################
    ####### ANALYSIS ##################
    ###################################

    #

    # def radial_intensity(self, channel_id, only_selected_nuclei=False, plot=True):
    #     """
    #     Get radial intensity either for all nuclei or only selected ones. This uses the pixel information,
    #     not the segmentation.
    #
    #     :param channel_id:
    #     :param only_selected_nuclei:
    #     :param plot:
    #     :return:
    #     """
    #
    #     data = self.channels_image[channel_id]
    #
    #     if only_selected_nuclei:
    #         mask = self.mask_selected
    #     else:
    #         mask = self.mask
    #     data[~mask] = 0  # set all False values to 0
    #
    #     if self.image_dim == 3:
    #         x, y = np.indices(data.shape[1:])  # note changed NON-inversion of x and y
    #         r = np.sqrt((x - self.center[1]) ** 2 + (y - self.center[2]) ** 2)
    #         r = r.astype(np.int)
    #         # now make 3d
    #         r = np.tile(r, (data.shape[0], 1, 1))
    #     else:
    #         x, y = np.indices(data.shape)  # note changed NON-inversion of x and y
    #         r = np.sqrt((x - self.center[0]) ** 2 + (y - self.center[1]) ** 2)
    #         r = r.astype(np.int)
    #
    #     tbin = np.bincount(r.ravel(), data.ravel())  # bincount makes +1 counts
    #
    #     if mask is None:
    #         nr = np.bincount(r.ravel())
    #     else:
    #         nr = np.bincount(r.ravel(), mask.astype(np.int).ravel())  # this line makes the average, i.e. since
    #         # we have
    #         # more bins with certain r's, must divide by abundance. If have mask, some of those should not be
    #         # counted, because they were set to zero above and should not contribute to the average.
    #     radialprofile = tbin / nr
    #     if np.isnan(radialprofile).any():
    #         print("WARNING: there were empty bins, i.e. at some radii there seem to be no cells.")
    #         radialprofile[np.isnan(radialprofile)] = 0.  # set these to 0
    #
    #     if plot:
    #         fig, ax = plt.subplots()
    #         ax.plot(np.arange(radialprofile.shape[0])*self.xy_scale, radialprofile)
    #         ax.set_ylim([0., ax.get_ylim()[1]])
    #         ax.set_xlim([0., ax.get_xlim()[1]/np.sqrt(2)])  # plot sqrt(2) less far because this is in the corners
    #         ax.set_xlabel('distance ($\mu m$)', fontsize=self.fontsize)
    #         ax.set_ylabel(str(channel_id) + ' intensity', fontsize=self.fontsize)
    #         # where there is no colony anyway
    #         self.nice_spines(ax)
    #
    #     return np.arange(radialprofile.shape[0])*self.xy_scale, radialprofile


    # def dot_plot(self, channel_id, colormap_cutoff=0.5, only_selected_cells=False):
    #     """
    #     Dot-plot as in Warmflash et al.
    #
    #     :param channel_id:
    #     :param colormap_cutoff: percentage of maximum for cutoff. Makes smaller differences more visible.
    #     :param only_selected_cells:
    #     :return:
    #     """
    #
    #     index = self._get_indices(only_selected_cells)
    #
    #     fig, ax = plt.subplots()
    #     if self.image_dim == 3:
    #         indices = (1,2)
    #     else:
    #         indices = (0,1)
    #
    #     cax = ax.scatter(np.vstack(self.df.centroid[index].values.flat)[:, indices[0]], np.vstack(
    #         self.df.centroid[index].values.flat)[:, indices[1]], c=self.df[channel_id][index].values,
    #                      s=40, edgecolors='none', cmap=plt.cm.viridis, vmax=colormap_cutoff*self.df[
    #             channel_id][index].values.max())
    #     self.nice_spines(ax)
    #     ax.autoscale(tight=1)
    #     ax.set_aspect('equal')
    #     fig.colorbar(cax)

    # def radial_profile_per_cell(self, channel_id, nbins=30, plot=True, only_selected_cells=False):
    #     """
    #
    #     :param channel_id:
    #     :param nbins: number of bins
    #     :param plot:
    #     :param only_selected_cells:
    #     :return:
    #     """
    #     index = self._get_indices(only_selected_cells)
    #
    #     indx_change = 0 if self.image_dim == 3 else 1
    #
    #     x = np.vstack(self.df.centroid[index].values.flat)[:, 1 - indx_change]
    #     y = np.vstack(self.df.centroid[index].values.flat)[:, 2 - indx_change]
    #     i = self.df[channel_id][index].values
    #
    #     r = np.sqrt((x-self.center[1-indx_change])**2+(y-self.center[2-indx_change])**2)
    #     # r = np.round(r).astype(np.int)
    #
    #     # n = np.bincount(r, i)
    #     # n2 = np.bincount(r)
    #     # xn = np.arange(r.min(),r.max())
    #     n, xn = np.histogram(r, bins=nbins, weights=i)
    #     n2, _ = np.histogram(r, bins=nbins)
    #
    #     if plot:
    #         _, ax = plt.subplots()
    #         ax.step(xn[:-1] - xn[0], n/n2, where='mid')
    #         ax.fill_between(xn[:-1] - xn[0], n/n2, alpha=0.2, step='mid')
    #         ax.set_xlabel('r', fontsize=self.fontsize)
    #         ax.set_ylabel(channel_id, fontsize=self.fontsize)
    #         # ax.set_xlim([0, ax.get_xlim()[1]])
    #         self.nice_spines(ax)
    #     return xn[:-1] - xn[0], n/n2

    # def coexpression_per_cell(self, channel_id1, channel_id2, only_selected_cells=False):
    #     """
    #     Scatter plot visualizing co-expression of two channels, with each datapoint the intensity of one cell.
    #
    #     :param channel_id1:
    #     :param channel_id2:
    #     :param only_selected_cells:
    #     :return:
    #     """
    #
    #     index = self._get_indices(only_selected_cells)
    #
    #     ch1 = self.df[channel_id1][index].values
    #     ch2 = self.df[channel_id2][index].values
    #
    #     fig, ax = plt.subplots()
    #     ax.scatter(ch1, ch2, edgecolors='none', c='k', alpha=0.8)
    #     nice_spines(ax)
    #     ax.set_xlabel(channel_id1, fontsize=self.fontsize)
    #     ax.set_ylabel(channel_id2, fontsize=self.fontsize)
    #     ax.autoscale(tight=1)
    #
    # def coexpression_per_pixel(self, channel_id1, channel_id2, downsample=10, only_selected_cells=False):
    #
    #     """
    #     Scatter plot visualizing co-expression of two channels, with each datapoint the intensity of one nuclear pixel.
    #
    #     :param channel_id1:
    #     :param channel_id2:
    #     :param downsample: Usually have a lot of point, so can only use ever downsample'th point.
    #     :param only_selected_cells:
    #     :return:
    #     """
    #
    #     ch1 = self.channels_image[channel_id1]
    #     ch2 = self.channels_image[channel_id2]
    #
    #     if only_selected_cells:
    #         mask = self.mask_selected
    #     else:
    #         mask = self.mask
    #
    #     fig, ax = plt.subplots()
    #     ax.scatter(ch1[mask > 0][::downsample], ch2[mask > 0][::downsample], marker='o', edgecolors='none', c='k',
    #                alpha=0.8, s=4)
    #     nice_spines(ax)
    #     ax.set_xlabel(channel_id1, fontsize=self.fontsize)
    #     ax.set_ylabel(channel_id2, fontsize=self.fontsize)
    #     ax.autoscale(tight=1)

    def z_heat_map(self, plot=True):
        """
        make a heat map of the z-position
        :return:
        """

        # make an index array of the same shape as image, but with z-index as value
        index_array = np.rollaxis(np.tile(np.arange(self.mask.shape[0]), (self.mask.shape[1], self.mask.shape[2], 1)),
                                  2, 0)
        z_extent = np.max(index_array * self.mask, axis=0)

        if plot:
            plt.imshow(z_extent, cmap='viridis')
            plt.colorbar()

        return z_extent

    def radial_z_height(self, plot=True):
        """
        Get radial height

        :param channel_id:
        :param only_selected_nuclei:
        :param plot:
        :return:
        """

        data = self.z_heat_map(plot=False)

        if self.image_dim == 3:
            x, y = np.indices(data.shape)  # note changed NON-inversion of x and y
            r = np.sqrt((x - self.center[1]) ** 2 + (y - self.center[2]) ** 2)
            r = r.astype(np.int)
        else:
            x, y = np.indices(data.shape)  # note changed NON-inversion of x and y
            r = np.sqrt((x - self.center[0]) ** 2 + (y - self.center[1]) ** 2)
            r = r.astype(np.int)

        tbin = np.bincount(r.ravel(), data.ravel())  # bincount makes +1 counts
        nr = np.bincount(r.ravel())
        radialprofile = tbin / nr

        if np.isnan(radialprofile).any():
            print("WARNING: there were empty bins, i.e. at some radii there seem to be no cells.")
            radialprofile[np.isnan(radialprofile)] = 0.  # set these to 0

        rvec = np.arange(radialprofile.shape[0])/self.xy_scale

        # smooth a bit
        radialprofile = self.running_mean(radialprofile, 20)
        rvec = rvec[:radialprofile.shape[0]]

        if plot:
            fig, ax = plt.subplots()
            ax.plot(rvec, radialprofile)
            ax.set_ylim([0., ax.get_ylim()[1]])
            ax.set_xlim([0., ax.get_xlim()[1]/np.sqrt(2)])  # plot sqrt(2) less far because this is in the corners
            ax.set_xlabel('distance ($\mu m$)', fontsize=self.fontsize)
            ax.set_ylabel('z', fontsize=self.fontsize)
            nice_spines(ax)

        return rvec, radialprofile


    @staticmethod
    def myrandom_cmap(seed=None, return_darker=False, n=1024):

        """
        make random colormap, good for plotting segmentation

        :param seed: seed for random colormap
        :param return_darker: also return a darker version
        :param n: number of colors
        :return: colormap(s)
        """
        np.random.seed(seed)
        random_array = np.random.rand(n, 3)
        random_array[0, :] = 1.
        random_array2 = np.zeros((n, 4))
        random_array2[:, :3] = random_array
        random_array2[:, 3] = .8
        # random_array2[0, 3] = 1. # keep 0 white

        if return_darker:
            return mpl.colors.ListedColormap(random_array), mpl.colors.ListedColormap(random_array2)
        else:
            return mpl.colors.ListedColormap(random_array)

    @staticmethod
    def _contrast_stretch(img, percentile=(2, 98)):
        """
        Stretch the contrast of an image to within given percentiles.

        :param img: Input image
        :param percentile: (lower percentile, upper percentile)
        :return: contrast stretched image
        """
        p2, p98 = np.percentile(img, percentile)
        return skimage.exposure.rescale_intensity(img, in_range=(p2, p98))

    @staticmethod
    def running_mean(x, N):
        # http://stackoverflow.com/questions/13728392/moving-average-or-running-mean
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / N


# static methods
#@staticmethod


def dump_mask_and_images(w, filename, write_every_nth_z=1, compression=4):
    """

    :param w: watershed object
    :param filename: filename to write output to
    :param compression: compression level, from 0 to 9 (default is 4)
    :return:
    """

    f = h5py.File(filename, 'w')
    f.create_dataset('mask', data=w.mask[::write_every_nth_z], compression="gzip", compression_opts=compression)
    f.create_dataset('dapi', data=w.image_stack[::write_every_nth_z], compression="gzip", compression_opts=compression)

    if not w.channels_image: #empty channels
        warnings.warn('No image channels to dump. Did you run apply_to_channels()?')
    else:
        for key in w.channels_image:
            f.create_dataset(key, data=w.channels_image[key][::write_every_nth_z], compression="gzip", compression_opts=compression)

    f.close()


def remove_background_func(im, n=None):
    """
    basic method to remove background, returns background subtracted image

    :param im: image
    :param n: lowest non-zero n pixels are used to estimate background
    :return: background subtracted image
    """
    # TODO: check whether this can make negative intensities

    if im.ndim == 3:
        imm = im.mean(axis=0)
    elif im.ndim == 2:
        imm = im.copy()
    else:
        raise RuntimeError("image has neither dimension 2 or 3")
    # following in comments has a bug
    # im -= np.partition(imm[imm.nonzero()], n)[:n].mean().astype(im.dtype)
    # im[im<0.] = 0.  # set negatives to 0
    # return im

    if n is None:
#            if im.ndim == 2:
#                n = 100
#            else:
#                n = 1000
        n = int(0.028 * im.shape[0]*im.shape[1]*im.shape[2])

    return im - np.partition(imm[imm.nonzero()], n)[:n].max()  # fast way of getting lowest n nonzero values


def radial_intensity(w_list, channel_id, only_selected_nuclei=False, plot=True, binsize=None, filename=None, xcutoff=None):
    """
    Get radial intensity either for all nuclei or only selected ones. This uses the pixel information,
    not the segmentation.

    :param channel_id:
    :param only_selected_nuclei:
    :param plot:
    :param binsize:
    :param filename: if specified, write the data (r, radial_intensity) to a txt file
    :param xcutoff:
    :return:
    """

    w_list = make_iterable(w_list)
    radial_profile_all = []

    for w in w_list:
        data = w.channels_image[channel_id]

        if only_selected_nuclei:
            mask = w.mask_selected
        else:
            mask = w.mask
        data[~mask] = 0  # set all False values to 0 - doesnt seem to have an effect anyway
        # data = data[~mask]   # set all False values to 0

        if w.image_dim == 3:
            x, y = np.indices(data.shape[1:])  # note changed NON-inversion of x and y
            r = np.sqrt((x - w.center[1]) ** 2 + (y - w.center[2]) ** 2)
            r = r.astype(np.int)
            # now make 3d
            r = np.tile(r, (data.shape[0], 1, 1))
        else:
            x, y = np.indices(data.shape)  # note changed NON-inversion of x and y
            r = np.sqrt((x - w.center[0]) ** 2 + (y - w.center[1]) ** 2)
            r = r.astype(np.int)
        
        tbin = np.bincount(r.ravel(), data.ravel())  # bincount makes +1 counts

        if mask is None:
            nr = np.bincount(r.ravel())
        else:
            nr = np.bincount(r.ravel(), mask.astype(np.int).ravel())  # this line makes the average, i.e. since
            # we have
            # more bins with certain r's, must divide by abundance. If have mask, some of those should not be
            # counted, because they were set to zero above and should not contribute to the average.
        radialprofile = tbin / nr
        if np.isnan(radialprofile).any():
            #print("WARNING: there were empty bins, i.e. at some radii there seem to be no cells.")
            radialprofile[np.isnan(radialprofile)] = 0.  # set these to 0

        radial_profile_all.append(radialprofile)

    # now average all radial profiles, need to pad all to the longest one
    maxlength = max([len(rp) for rp in radial_profile_all])

    averaged_radial_profile = np.zeros((len(radial_profile_all),maxlength))

    if binsize is not None:
        running_mean_vec_length = running_mean(averaged_radial_profile[0], binsize).shape[0]
        averaged_radial_profile_rm = np.zeros((len(radial_profile_all), running_mean_vec_length))
    
    for i1, rp in enumerate(radial_profile_all):
        averaged_radial_profile[i1] = np.pad(rp, (0, maxlength - len(rp)), 'constant')
        if binsize is not None:
            averaged_radial_profile_rm[i1] = running_mean(averaged_radial_profile[i1], binsize)
        
    if binsize is None:
        averaged_radial_profile = averaged_radial_profile.mean(axis=0)
        averaged_radial_profile_std = averaged_radial_profile.std(axis=0)
    else:
        averaged_radial_profile = averaged_radial_profile_rm.mean(axis=0)
        averaged_radial_profile_std = averaged_radial_profile_rm.std(axis=0)

    if plot:
        fig, ax = plt.subplots()
        xx, yy = np.arange(averaged_radial_profile.shape[0]) / w.xy_scale, averaged_radial_profile
        ax.plot(xx, yy)
        if len(w_list) > 1: # only makes sense to plot std when there are several colonies to average over
            ax.fill_between(xx, yy+averaged_radial_profile_std, yy-averaged_radial_profile_std, alpha=.2)

        #ax.errorbar(np.arange(averaged_radial_profile.shape[0]) / w.xy_scale, averaged_radial_profile, yerr=averaged_radial_profile_std)
        
        ax.set_ylim([0., ax.get_ylim()[1]])
        if xcutoff is not None:
            ax.set_xlim(0, xcutoff)
        #ax.set_xlim([0., ax.get_xlim()[1] / np.sqrt(2)])  # plot sqrt(2) less far because this is in the corners
        ax.set_xlabel('distance ($\mu m$)', fontsize=w.fontsize)
        ax.set_ylabel(str(channel_id) + ' intensity', fontsize=w.fontsize)
        # where there is no colony anyway
        nice_spines(ax)

    # save to file
    if filename is not None:
        np.savetxt(filename, np.vstack((np.arange(averaged_radial_profile.shape[0]) / w.xy_scale, averaged_radial_profile, averaged_radial_profile_std)).transpose(), delimiter=',')
               
    return np.arange(averaged_radial_profile.shape[0]) / w.xy_scale, averaged_radial_profile


def radial_intensity_z(w_list, channel_id, only_selected_nuclei=False, plot=True, binsize=None, filename=None,
                     xcutoff=None, mask=True, normalize=False):
    """
    Get radial intensity AS A FUNCTION OF Z, either for all nuclei or only selected ones. This uses the pixel information,
    not the segmentation.

    :param channel_id:
    :param only_selected_nuclei:
    :param plot:
    :param binsize:
    :param filename: if specified, write the data (r, radial_intensity) to a txt file
    :param xcutoff:
    :return:
    """

    if mask:
        use_mask = True
    else:
        use_mask = False

    w_list = make_iterable(w_list)
    radial_profile_all = []

    # if len(w_list) > 1:
    #     raise NotImplementedError('currently not implemented for multiple colonies')

    for w in w_list:
        data = w.channels_image[channel_id]

        if use_mask:
            mask = np.ones_like(w.mask)
        elif only_selected_nuclei:
            mask = w.mask_selected
        else:
            mask = w.mask
        data[~mask] = 0  # set all False values to 0

        #normalize data
        if normalize:
            data = data/w.image_stack
            data[np.isinf(data)] = 0. # set division by zero to 0
            # data /= data.max() #normalize to between 0..1

        if w.image_dim == 3:
            x, y = np.indices(data.shape[1:])  # note changed NON-inversion of x and y
            r = np.sqrt((x - w.center[1]) ** 2 + (y - w.center[2]) ** 2)
            r = r.astype(np.int)
            # now make 3d
            r = np.tile(r, (data.shape[0], 1, 1))
        else:
            x, y = np.indices(data.shape)  # note changed NON-inversion of x and y
            r = np.sqrt((x - w.center[0]) ** 2 + (y - w.center[1]) ** 2)
            r = r.astype(np.int)

        # split into different z
        rshape = np.unique(r.ravel()).shape[0]  # this is the r bin vector
        tbinz = np.zeros((data.shape[0], rshape))
        # print(tbinz.shape)
        for zz in range(data.shape[0]):
            # print(np.bincount(r[zz].ravel(), data[zz].ravel()).shape)
            tbinz[zz] = np.bincount(r[zz].ravel(), data[zz].ravel())  # bincount makes +1 counts

        # tbin = np.bincount(r.ravel(), data.ravel())  # bincount makes +1 counts

        nrz = np.zeros((data.shape[0], rshape))
        if mask is None:
            for zz in range(data.shape[0]):
                nrz[zz] = np.bincount(r[zz].ravel())
        else:
            for zz in range(data.shape[0]):
                nrz[zz] = np.bincount(r[zz].ravel(), mask[zz].astype(np.int).ravel())  # this line makes the average, i.e. since
            # we have
            # more bins with certain r's, must divide by abundance. If have mask, some of those should not be
            # counted, because they were set to zero above and should not contribute to the average.
        radialprofilez = tbinz / nrz
        if np.isnan(radialprofilez).any():
            # print("WARNING: there were empty bins, i.e. at some radii there seem to be no cells.")
            radialprofilez[np.isnan(radialprofilez)] = 0.  # set these to 0

        radial_profile_all.append(radialprofilez)

    return np.arange(rshape), radial_profile_all

    # now average all radial profiles, need to pad all to the longest one
    maxlength = max([rp.shape[1] for rp in radial_profile_all])

    averaged_radial_profile = np.zeros((data.shape[0], len(radial_profile_all), maxlength)) # z is first index

    # if binsize is not None:
    #     running_mean_vec_length = running_mean(averaged_radial_profile[0], binsize).shape[0]
    #     averaged_radial_profile_rm = np.zeros((len(radial_profile_all), running_mean_vec_length))

    for i1, rp in enumerate(radial_profile_all):
        averaged_radial_profile[i1] = np.pad(rp, (0, maxlength - len(rp)), 'constant')
        # if binsize is not None:
        #     averaged_radial_profile_rm[i1] = running_mean(averaged_radial_profile[i1], binsize)

    if binsize is None:
        averaged_radial_profile = averaged_radial_profile.mean(axis=0)
        averaged_radial_profile_std = averaged_radial_profile.std(axis=0)
    # else:
    #     averaged_radial_profile = averaged_radial_profile_rm.mean(axis=0)
    #     averaged_radial_profile_std = averaged_radial_profile_rm.std(axis=0)

    if plot:
        fig, ax = plt.subplots()
        xx, yy = np.arange(averaged_radial_profile.shape[0]) / w.xy_scale, averaged_radial_profile
        ax.plot(xx, yy)
        if len(w_list) > 1:  # only makes sense to plot std when there are several colonies to average over
            ax.fill_between(xx, yy + averaged_radial_profile_std, yy - averaged_radial_profile_std, alpha=.2)

        # ax.errorbar(np.arange(averaged_radial_profile.shape[0]) / w.xy_scale, averaged_radial_profile, yerr=averaged_radial_profile_std)

        ax.set_ylim([0., ax.get_ylim()[1]])
        if xcutoff is not None:
            ax.set_xlim(0, xcutoff)
        # ax.set_xlim([0., ax.get_xlim()[1] / np.sqrt(2)])  # plot sqrt(2) less far because this is in the corners
        ax.set_xlabel('distance ($\mu m$)', fontsize=w.fontsize)
        ax.set_ylabel(str(channel_id) + ' intensity', fontsize=w.fontsize)
        # where there is no colony anyway
        nice_spines(ax)

    # save to file
    if filename is not None:
        np.savetxt(filename, np.vstack((np.arange(averaged_radial_profile.shape[0]) / w.xy_scale,
                                        averaged_radial_profile, averaged_radial_profile_std)).transpose(),
                   delimiter=',')

    return np.arange(averaged_radial_profile.shape[0]) / w.xy_scale, averaged_radial_profile



def radial_intensity_z_stack(stack, center, plot=True, binsize=None, filename=None,
                     xcutoff=None, normalize=False):
    """
    Get radial intensity AS A FUNCTION OF Z FOR AN ARBITRARY STACK

    :param channel_id:
    :param only_selected_nuclei:
    :param plot:
    :param binsize:
    :param filename: if specified, write the data (r, radial_intensity) to a txt file
    :param xcutoff:
    :return:
    """

    data = stack
    #
    # #normalize data
    # if normalize:
    #     data = data/w.image_stack
    #     data[np.isinf(data)] = 0. # set division by zero to 0
    #     # data /= data.max() #normalize to between 0..1

    x, y = np.indices(data.shape[1:])  # note changed NON-inversion of x and y
    r = np.sqrt((x - center[1]) ** 2 + (y - center[2]) ** 2)
    r = r.astype(np.int)
    # now make 3d
    r = np.tile(r, (data.shape[0], 1, 1))

    # split into different z
    rshape = np.unique(r.ravel()).shape[0]  # this is the r bin vector
    tbinz = np.zeros((data.shape[0], rshape))
    for zz in range(data.shape[0]):
        tbinz[zz] = np.bincount(r[zz].ravel(), data[zz].ravel())  # bincount makes +1 counts

    nrz = np.zeros((data.shape[0], rshape))
    for zz in range(data.shape[0]):
        nrz[zz] = np.bincount(r[zz].ravel())
    # else:
    #     for zz in range(data.shape[0]):
    #         nrz[zz] = np.bincount(r[zz].ravel(), mask[zz].astype(np.int).ravel())  # this line makes the average, i.e. since
        # we have
        # more bins with certain r's, must divide by abundance. If have mask, some of those should not be
        # counted, because they were set to zero above and should not contribute to the average.
    radialprofilez = tbinz / nrz
    if np.isnan(radialprofilez).any():
        # print("WARNING: there were empty bins, i.e. at some radii there seem to be no cells.")
        radialprofilez[np.isnan(radialprofilez)] = 0.  # set these to 0

    return np.arange(rshape), radialprofilez





def radial_profile_per_cell(w_list, channel_id, nbins=30, plot=True, only_selected_cells=False, fontsize=16):
    """

    :param w_list: watershed object or list of objects
    :param channel_id:
    :param nbins: number of bins
    :param plot:
    :param only_selected_cells:
    :return:
    """

    w_list = make_iterable(w_list)
    r_all = np.zeros((0,))
    i_all = np.zeros((0,))

    for w in w_list:
        index = w._get_indices(only_selected_cells)

        indx_change = 0 if w.image_dim == 3 else 1

        x = np.vstack(w.df.centroid[index].values.flat)[:, 1 - indx_change]
        y = np.vstack(w.df.centroid[index].values.flat)[:, 2 - indx_change]
        i = w.df[channel_id][index].values

        r = np.sqrt((x - w.center[1 - indx_change]) ** 2 + (y - w.center[2 - indx_change]) ** 2)

        r_all = np.hstack((r_all, r))
        i_all = np.hstack((i_all, i))

    n, xn = np.histogram(r_all, bins=nbins, weights=i_all)
    n2, _ = np.histogram(r_all, bins=nbins)

    if plot:
        _, ax = plt.subplots()
        ax.step(xn[:-1] - xn[0], n / n2, where='mid')
        ax.fill_between(xn[:-1] - xn[0], n / n2, alpha=0.2, step='mid')
        ax.set_xlabel('r', fontsize=fontsize)
        ax.set_ylabel(channel_id, fontsize=fontsize)
        nice_spines(ax)

    # make sure to replace nans by 0.
    return_vec = n/n2
    return_vec[np.isnan(return_vec)] = 0.
    
    return xn[:-1] - xn[0], return_vec


def dot_plot(w_list, channel_id, color_range=None, colormap_cutoff=0.5, only_selected_cells=False, markersize=30, colorbar=False, r_limit=None, axis=False, filename=None, cmap=plt.cm.viridis):
    """
    Dot-plot as in Warmflash et al.

    :param w_list: watershed object or list of objects
    :param channel_id:
    :param color_range: color range, overrides colormap_cutoff
    :param colormap_cutoff: percentage of maximum for cutoff. Makes smaller differences more visible.
    :param only_selected_cells:
    :param markersize:
    :param colorbar: whether to plot a colorbar
    :param r_limit: only plot cells smaller than this radius (in um)
    :param axis: plot axes or not
    :return:
    """

    w_list = make_iterable(w_list)
    x_all = np.zeros((0,))
    y_all = np.zeros((0,))
    c_all = np.zeros((0,))

    for w in w_list:

        if w.image_dim == 3:
            indices = (1, 2)
        else:
            indices = (0, 1)

        index = w._get_indices(only_selected_cells)
        x = np.vstack(w.df.centroid[index].values.flat)[:, indices[0]] - w.center[indices[0]]
        y = np.vstack(w.df.centroid[index].values.flat)[:, indices[1]] - w.center[indices[1]]
        c = w.df[channel_id][index].values

        if r_limit is not None:
            index_smaller_than_r = (x**2 + y**2)<r_limit**2
            x = x[index_smaller_than_r]
            y = y[index_smaller_than_r]
            c = c[index_smaller_than_r]

        x_all = np.hstack((x_all, x))
        y_all = np.hstack((y_all, y))
        c_all = np.hstack((c_all, c))

    fig, ax = plt.subplots()

    # sort according to intensity for displaying the high intensity dots on top of the low ones, which just look better
    sort_array = c_all.argsort()
    x_all = x_all[sort_array]
    y_all = y_all[sort_array]
    c_all = c_all[sort_array]

    if color_range is not None:
        cax = ax.scatter(x_all/w.xy_scale, y_all/w.xy_scale, c=c_all, s=markersize,
                         edgecolors='none', cmap=cmap, vmin=color_range[0], vmax=color_range[1])
    else:
        cax = ax.scatter(x_all/w.xy_scale, y_all/w.xy_scale, c=c_all/c_all.max()*1000, s=markersize,
                         edgecolors='none', cmap=cmap, vmax=colormap_cutoff*1000)
    nice_spines(ax, grid=False)
    ax.autoscale(tight=1)
    ax.set_aspect('equal')
    if not axis:
        ax.axis('off')
    if colorbar:
        fig.colorbar(cax)

    if filename is not None:
        fig.savefig(filename)


def radial_histogram_plot(w_list, channel_id, clip_quantile=100, bins=25, average_func=np.mean, only_selected_cells=False, colorbar=False, r_limit=None, axis=False, filename=None, cmap=plt.cm.viridis, figsize=(4,4)):
    """
    Radial histogram weighted by expression

    :param w_list: watershed object or list of objects
    :param channel_id:
    :param color_range: cutoff at this quantile (100 doesn't cut off anything, try 97 or so to saturate colormap
    :param only_selected_cells:
    :param markersize:
    :param colorbar: whether to plot a colorbar
    :param r_limit: only plot cells smaller than this radius (in um)
    :param axis: plot axes or not
    :return:
    """

    w_list = make_iterable(w_list)
    x_all = np.zeros((0,))
    y_all = np.zeros((0,))
    c_all = np.zeros((0,))

    for w in w_list:

        if w.image_dim == 3:
            indices = (1, 2)
        else:
            indices = (0, 1)

        index = w._get_indices(only_selected_cells)
        x = np.vstack(w.df.centroid[index].values.flat)[:, indices[0]] - w.center[indices[0]]
        y = np.vstack(w.df.centroid[index].values.flat)[:, indices[1]] - w.center[indices[1]]
        c = w.df[channel_id][index].values

        if r_limit is not None:
            index_smaller_than_r = (x**2 + y**2)<r_limit**2
            x = x[index_smaller_than_r]
            y = y[index_smaller_than_r]
            c = c[index_smaller_than_r]

        x_all = np.hstack((x_all, x))
        y_all = np.hstack((y_all, y))
        c_all = np.hstack((c_all, c))

    # normalize c and saturate (clip) using clip_quantile
    c_all /= c_all.max()
    c_clip = np.clip(c_all, c_all.min(), np.percentile(c_all, clip_quantile))
    c_clip -= c_clip.min()
    c_clip /= c_clip.max()

    # NOTE physt radial histogram seems to have a bug in the weight
    # if bins is None:
    #     hist = special.polar_histogram(x_all, y_all, "human", weights=c_clip)
    # else:
    #     hist = special.polar_histogram(x_all, y_all, "human", bins, weights=c_clip)
    #
    # ax = hist.plot.polar_map(density=False, figsize=figsize, lw=0.1, show_zero=True, cmap=cmap)
    # ax.set_yticklabels([])
    # ax.set_xticklabels([])
    # ax.grid('off')
    # ax.set_theta_zero_location('N')

    # can 'shake' existing data to fill holes - this is a bug in hexbin with weights
    # shake = False
    # if shake:
    #     x2 = x_all + (np.random.randn(x_all.shape[0])) * x_all.max()/20 # move randomly by 1%
    #     y2 = y_all + (np.random.randn(y_all.shape[0])) * y_all.max() /20  # move randomly by 1%
    #     x_all = np.hstack((x_all, x2))
    #     y_all = np.hstack((y_all, y2))
    #     c_clip = np.hstack((c_clip,c_clip))
    #     if r_limit is not None:
    #         index_smaller_than_r = (x_all**2 + y_all**2)<r_limit**2
    #         x_all = x_all[index_smaller_than_r]
    #         y_all = y_all[index_smaller_than_r]
    #         c_clip = c_clip[index_smaller_than_r]


    fig,ax=plt.subplots(figsize=figsize)
    hb = ax.hexbin(x_all, y_all, C=c_clip, gridsize=bins, mincnt=0, cmap=cmap, reduce_C_function=average_func)
    cmap_mpl = matplotlib.cm.get_cmap(cmap)
    ax.set_facecolor(cmap_mpl(0.))
    if filename is not None:
        fig.savefig(filename)



def dot_plot_radial(w_list, channel_id, color_range=None, colormap_cutoff=0.5, only_selected_cells=False, markersize=30, colorbar=False, r_limit=None, axis=False, filename=None, cmap=plt.cm.viridis, dark_background=False, ec='none'):
    """
    Dot-plot, but radial and z-resolved.

    :param w_list: watershed object or list of objects
    :param channel_id:
    :param color_range: color range, overrides colormap_cutoff
    :param colormap_cutoff: percentage of maximum for cutoff. Makes smaller differences more visible.
    :param only_selected_cells:
    :param markersize:
    :param colorbar: whether to plot a colorbar
    :param r_limit: only plot cells smaller than this radius (in um)
    :param axis: plot axes or not
    :param dark_background: whether to plot on a dark background
    :return:

    """

    w_list = make_iterable(w_list)
    x_all = np.zeros((0,))
    y_all = np.zeros((0,))
    z_all = np.zeros((0,))
    c_all = np.zeros((0,))

    for w in w_list:

        if w.image_dim == 3:
            indices = (1, 2, 0)
        else:
            indices = (0, 1)

        index = w._get_indices(only_selected_cells)
        x = np.vstack(w.df.centroid[index].values.flat)[:, indices[0]] - w.center[indices[0]]
        y = np.vstack(w.df.centroid[index].values.flat)[:, indices[1]] - w.center[indices[1]]
        print(np.vstack(w.df.centroid[index].values.flat).shape)
        z = np.vstack(w.df.centroid[index].values.flat)[:, indices[2]]
        c = w.df[channel_id][index].values

        if r_limit is not None:
            index_smaller_than_r = (x**2 + y**2)<r_limit**2
            x = x[index_smaller_than_r]
            y = y[index_smaller_than_r]
            c = c[index_smaller_than_r]
            z = z[index_smaller_than_r]

        x_all = np.hstack((x_all, x))
        y_all = np.hstack((y_all, y))
        c_all = np.hstack((c_all, c))
        z_all = np.hstack((z_all, z))

    fig, ax = plt.subplots()

    # sort according to intensity for displaying the high intensity dots on top of the low ones, which just look better
    sort_array = c_all.argsort()
    x_all = x_all[sort_array]
    y_all = y_all[sort_array]
    c_all = c_all[sort_array]
    z_all = z_all[sort_array]


    # if z_all is not None:
    #     z_all /= w.z_scale
    if color_range is not None:
        cax = ax.scatter(np.sqrt((x_all/w.xy_scale)**2+(y_all/w.xy_scale)**2), z_all/w.z_scale, c=c_all, s=markersize,
                         edgecolors=ec, cmap=cmap, vmin=color_range[0], vmax=color_range[1])
    else:
        cax = ax.scatter(np.sqrt((x_all/w.xy_scale)**2+(y_all/w.xy_scale)**2), z_all/w.z_scale, c=c_all/c_all.max()*1000, s=markersize,
                         edgecolors=ec, cmap=cmap, vmax=colormap_cutoff*1000)

    nice_spines(ax, grid=False, dark=dark_background)
    ax.autoscale(tight=1)
    ax.set_aspect('equal')
    ax.set_xlabel('radius ($\mu$m)')
    ax.set_ylabel('z ($\mu$m)')
    if not axis:
        ax.axis('off')
    if colorbar:
        fig.colorbar(cax)

    if filename is not None:
        fig.savefig(filename)


def coexpression_per_cell(w_list, channel_id1, channel_id2, xy_lim=None, normalize=True, only_selected_cells=False, fontsize=20, filename=None, fit=False):
    """
    Scatter plot visualizing co-expression of two channels, with each datapoint the intensity of one cell.

    :param w_list: watershed object or list of objects
    :param channel_id1:
    :param channel_id2:
    :param normalize:
    :param only_selected_cells:
    :param fontsize: fontsize for plotting
    :return:
    """

    w_list = make_iterable(w_list)
    ch1_all = np.zeros((0,))
    ch2_all = np.zeros((0,))

    for w in w_list:
        index = w._get_indices(only_selected_cells)
        ch1 = w.df[channel_id1][index].values
        ch2 = w.df[channel_id2][index].values
        ch1_all = np.hstack((ch1_all, ch1))
        ch2_all = np.hstack((ch2_all, ch2))

    #normalize
    if normalize:
        # factor of 1.01 such that highest is not exactly one. Since this is arbitrary scaling of arbitrary
        # intensity values, can chose that
        ch1_all /= 1.01*ch1_all.max()
        ch2_all /= 1.01*ch2_all.max()
        # print(ch1_all.max(), ch2_all.max())


    almost_black = '#262626'
    fig, ax = plt.subplots(figsize=(4,4))
    dd = pd.DataFrame(data=np.vstack((ch1_all, ch2_all)).T, columns=(channel_id1, channel_id2))
    import seaborn as sns
    if fit:

        # ax.plot(np.unique(ch1_all), np.poly1d(np.polyfit(ch1_all, ch1_all, 1))(np.unique(ch1_all)), '--k', lw=0.5)
        # ax.scatter(ch1_all, ch2_all, edgecolors='none', alpha=0.8, color=almost_black)
        # sns.lmplot(x=channel_id1, y=channel_id2, data=dd, palette='Set1')
        sns.regplot(x=channel_id1, y=channel_id2, data=dd, ax=ax, color=almost_black)
        # print(ch1_all.shape, ch2_all.shape`)
    else:
        # ax.scatter(ch1_all, ch2_all)
        sns.regplot(x=channel_id1, y=channel_id2, data=dd, ax=ax, fit_reg=False)
        # sns.lmplot(x=channel_id1, y=channel_id2, data=dd, palette='Set1')

    # nice_spines(ax, grid=False)
    # ax.set_xlabel(channel_id1, fontsize=fontsize)
    # ax.set_ylabel(channel_id2, fontsize=fontsize)
    # ax.autoscale(tight=1)
    # ax.tick_params(axis='both', which='major', labelsize=16)
    # ax.tick_params(axis='both', which='minor', labelsize=10)
    #
    if normalize:
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    if xy_lim is not None:
        ax.set_xlim([0, xy_lim])
        ax.set_ylim([0, xy_lim])

    sns.despine()
    fig.tight_layout()
    if filename is not None:
        plt.savefig(filename)

    return ch1_all, ch2_all


def histogram_pixel(w_list, channel_id1, downsample=1, only_selected_cells=False, fontsize=16, bins=50, normalize=False):
    """
    Histogram of pixel intensities

    :param w_list: watershed object or list of objects
    :param channel_id1:
    :param downsample: Usually have a lot of point, so can only use ever downsample'th point.
    :param only_selected_cells:
    :param fontsize: of labels
    :param bins: number of bins
    :param normalize: normalize by DAPI
    :return:
    """

    w_list = make_iterable(w_list)
    ch1_all = np.zeros((0,))

    for w in w_list:
        ch1 = w.channels_image[channel_id1]

        if only_selected_cells:
            mask = w.mask_selected
        else:
            mask = w.mask

        if normalize:
            to_add = ch1[mask > 0][::downsample]/(w.image_stack[mask>0][::downsample])
        else:
            to_add = ch1[mask > 0][::downsample]

        ch1_all = np.hstack((ch1_all, to_add))

    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(ch1_all, bins=bins);
    # d_masked = np.ma.masked_where(d == 0, d)  # mask zero bins in plot
    # ax.imshow(d_masked.T, extent=[x[0], x[-1], y[0], y[-1]], cmap='magma_r', origin='low')

    nice_spines(ax)
    ax.set_xlabel(channel_id1, fontsize=fontsize)
    # ax.autoscale(tight=1)


def coexpression_per_pixel(w_list, channel_id1, channel_id2, downsample=1, only_selected_cells=False, fontsize=16,
                           lognorm=False, bins=50, gates=None):
    """
    Scatter plot visualizing co-expression of two channels, with each datapoint the intensity of one nuclear pixel.

    :param w_list: watershed object or list of objects
    :param channel_id1:
    :param channel_id2:
    :param downsample: Usually have a lot of point, so can only use ever downsample'th point.
    :param only_selected_cells:
    :param lognorm: do a log plot
    :param bins: number of bins
    :param gates: put gates on the expressions and get percentages
    :return:
    """

    w_list = make_iterable(w_list)
    ch1_all = np.zeros((0,))
    ch2_all = np.zeros((0,))

    for w in w_list:
        ch1 = w.channels_image[channel_id1]
        ch2 = w.channels_image[channel_id2]

        if only_selected_cells:
            mask = w.mask_selected
        else:
            mask = w.mask

        ch1_all = np.hstack((ch1_all, ch1[mask>0][::downsample]))
        ch2_all = np.hstack((ch2_all, ch2[mask>0][::downsample]))

    fig, ax = plt.subplots(figsize=(6,6))
    # ax.scatter(ch1_all, ch2_all, marker='o', edgecolors='none', c='k', alpha=0.8, s=4)

    # gates
    if gates is not None:
        assert(len(gates)==2)
        # count the number of cells below and above thresholds
        l1_l2 = (np.bitwise_and(ch1_all < gates[0], ch2_all < gates[1])).sum()  # low 1 low 2
        h1_l2 = (np.bitwise_and(ch1_all >= gates[0], ch2_all < gates[1])).sum()  # high 1 low 2
        l1_h2 = (np.bitwise_and(ch1_all < gates[0], ch2_all >= gates[1])).sum()  # low 1 high 2
        h1_h2 = (np.bitwise_and(ch1_all >= gates[0], ch2_all >= gates[1])).sum()  # high 1 high 2
        total_count = l1_l2 + h1_l2 + l1_h2 + h1_h2
        assert(total_count == len(ch1_all))
        # print((np.array([l1_l2, h1_l2, l1_h2, h1_h2]/total_count)*100).round())
        print(channel_id1, '<', gates[0], 'and', channel_id2, '<', gates[1], '=', '{:.2f}%'.format(l1_l2/total_count*100))
        print(channel_id1, '>', gates[0], 'and', channel_id2, '<', gates[1], '=', '{:.2f}%'.format(h1_l2/total_count*100))
        print(channel_id1, '<', gates[0], 'and', channel_id2, '>', gates[1], '=', '{:.2f}%'.format(l1_h2/total_count*100))
        print(channel_id1, '>', gates[0], 'and', channel_id2, '>', gates[1], '=', '{:.2f}%'.format(h1_h2/total_count*100))

    if lognorm:
        plt.hist2d(ch1_all, ch2_all, bins=bins, norm=LogNorm(), cmap='magma_r')
    else:
        d, x, y = np.histogram2d(ch1_all, ch2_all, bins=bins)
        d_masked = np.ma.masked_where(d == 0, d)  # mask zero bins in plot
        ax.imshow(d_masked.T, extent=[x[0], x[-1], y[0], y[-1]], cmap='magma_r', origin='low')
    if gates is not None:
        ax.axvline(x=gates[0], color='k', ls='--', lw=1.3)
        ax.axhline(y=gates[1], color='k', ls='--', lw=1.3)
        # plt.hist2d(ch1_all, ch2_all, bins=bins)
    nice_spines(ax)
    ax.set_xlabel(channel_id1, fontsize=fontsize)
    ax.set_ylabel(channel_id2, fontsize=fontsize)
    ax.autoscale(tight=1)


def z_heat_map(w, plot=True):
    """
    make a heat map of the z-position
    :return:
    """

    # make an index array of the same shape as image, but with z-index as value
    index_array = np.rollaxis(np.tile(np.arange(w.mask.shape[0]), (w.mask.shape[1], w.mask.shape[2], 1)),
                              2, 0)
    z_extent = np.max(index_array * w.mask, axis=0)

    if plot:
        plt.imshow(z_extent)
        plt.colorbar()

    return z_extent


def radial_z_height(w_list, z_scale=None, binsize=20, plot=True, fontsize=16, filename=None):
    """
    Get radial height

    :param channel_id:
    :param only_selected_nuclei:
    :param plot:
    :return:
    """

    w_list = make_iterable(w_list)
    radial_z_all = []

    for w in w_list:
        data = w.z_heat_map(plot=False)

        if w.image_dim == 3:
            x, y = np.indices(data.shape)  # note changed NON-inversion of x and y
            r = np.sqrt((x - w.center[1]) ** 2 + (y - w.center[2]) ** 2)
            r = r.astype(np.int)
        else:
            x, y = np.indices(data.shape)  # note changed NON-inversion of x and y
            r = np.sqrt((x - w.center[0]) ** 2 + (y - w.center[1]) ** 2)
            r = r.astype(np.int)

        tbin = np.bincount(r.ravel(), data.ravel())  # bincount makes +1 counts
        nr = np.bincount(r.ravel())
        radialprofile = tbin / nr

        if np.isnan(radialprofile).any():
            print("WARNING: there were empty bins, i.e. at some radii there seem to be no cells.")
            radialprofile[np.isnan(radialprofile)] = 0.  # set these to 0

        radial_z_all.append(radialprofile)

    maxlength = max([len(rp) for rp in radial_z_all])
    averaged_radial_profile = np.zeros((len(radial_z_all),maxlength))
    for i1, rp in enumerate(radial_z_all):
        averaged_radial_profile[i1] = np.pad(rp, (0, maxlength - len(rp)), 'constant')
    averaged_radial_profile = averaged_radial_profile.mean(axis=0)

    rvec = np.arange(averaged_radial_profile.shape[0]) / w.xy_scale

    # smooth a bit
    averaged_radial_profile = running_mean(averaged_radial_profile, binsize)
    rvec = rvec[:averaged_radial_profile.shape[0]]

    if plot:
        fig, ax = plt.subplots()
        if z_scale is None:
            ax.plot(rvec, averaged_radial_profile)
        else:
            ax.plot(rvec, z_scale * averaged_radial_profile)
        ax.set_ylim([0., ax.get_ylim()[1]])
        ax.set_xlim([0., ax.get_xlim()[1] / np.sqrt(2)])  # plot sqrt(2) less far because this is in the corners
        ax.set_xlabel('distance ($\mu m$)', fontsize=fontsize)
        if z_scale is None:
            ax.set_ylabel('z', fontsize=fontsize)
        else:
            ax.set_ylabel('z ($\mu m$)', fontsize=fontsize)
        nice_spines(ax)

    # save to file
    if filename is not None:
        np.savetxt(filename, np.vstack((rvec, averaged_radial_profile)).transpose(), delimiter=',')
        
    return rvec, averaged_radial_profile


def scatter_expression_z(w_list, channel_id, downsample=1, only_selected_cells=False, fontsize=16,
                         bins=100, lognorm=False, expression_max=None):
    """
    Scatter plot visualizing expression of channel and z-location
    pixel.

    :param w_list: watershed object or list of objects
    :param channel_id:
    :param downsample: Usually have a lot of point, so can only use ever downsample'th point.
    :param only_selected_cells:
    :return:
    """

    w_list = make_iterable(w_list)
    ch1_all = np.zeros((0,))
    z_all = np.zeros((0,))

    for w in w_list:

        if only_selected_cells:
            mask = w.mask_selected
        else:
            mask = w.mask

        ch1 = w.channels_image[channel_id]
        z = np.rollaxis(np.tile(np.arange(w.mask.shape[0]), (w.mask.shape[1], w.mask.shape[2], 1)), 2, 0)

        ch1_all = np.hstack((ch1_all, ch1[mask>0][::downsample]))
        z_all = np.hstack((z_all, z[mask>0][::downsample]))

    fig, ax = plt.subplots()
    # ax.scatter(ch1_all, z_all, marker='o', edgecolors='none', c='k', alpha=0.8, s=4)
    zbins = np.arange(z_all.max())

    if expression_max is not None:
        z_all = z_all[ch1_all < expression_max]
        ch1_all = ch1_all[ch1_all < expression_max]
    if lognorm:
        ax.hist2d(ch1_all, z_all, bins=[bins,zbins], norm=LogNorm())
    else:
        ax.hist2d(ch1_all, z_all, bins=[bins, zbins])

    # if expression_max is not None:
    #     ax.set_xlim(ax.get_xlim()[0], expression_max)
    #     print(ax.get_xlim()[0], expression_max)

    nice_spines(ax)
    ax.set_xlabel(channel_id, fontsize=fontsize)
    ax.set_ylabel('z', fontsize=fontsize)
    ax.autoscale(tight=1)


def expression_histogram(w_list, channel_id, only_selected_cells=False, bins=50, log=False):
    """
    histogram of a certain channel, can use that to determine threshold

    :param w:
    :param channel_id:
    :return:
    """

    w_list = make_iterable(w_list)
    ch1_all = np.zeros((0,))

    for w in w_list:

        if only_selected_cells:
            mask = w.mask_selected
        else:
            mask = w.mask

        ch1 = w.channels_image[channel_id]
        ch1_all = np.hstack((ch1_all, ch1[mask>0]))

        with plt.style.context('ggplot'):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(ch1_all, log=log, bins=np.arange(0, ch1_all.max(), bins), range=(0, ch1_all.max()))
            ax.set_ylabel(channel_id, fontsize=16)


def expression_map(w, channel_id, threshold, only_selected_cells=False, colormap_cutoff=.2):
    """
    make a 2d map of where expression above threshold is

    :param w_list:
    :param channel_id:
    :param threshold:
    :return:
    """

    if only_selected_cells:
        mask = w.mask_selected
    else:
        mask = w.mask

    ch1 = w.channels_image[channel_id]
    ch1_thresh_project = ch1.sum(axis=0)* np.ma.masked_where((ch1 < threshold) | (mask < 0), ch1).sum(axis=0)
    fig, ax = plt.subplots()
    ax.imshow(ch1_thresh_project, vmax=colormap_cutoff * ch1_thresh_project.max())


def co_expression_map(w, channel_id1, channel_id2, threshold1, threshold2, only_selected_cells=False,
                     colormap_cutoff=.2):
    """
    make a 2d map of where expression above threshold is

    :param w_list:
    :param channel_id:
    :param threshold:
    :return:
    """

    if only_selected_cells:
        mask = w.mask_selected
    else:
        mask = w.mask

    ch1 = w.channels_image[channel_id1]
    ch2 = w.channels_image[channel_id2]

    combined = (ch1*ch2).mean(axis=0)
    ch_combined_thresh_project = combined * \
                         np.ma.masked_where((ch1 < threshold1) | (ch2 < threshold2) | (mask < 0),
                                                                       ch1*ch2).mean(axis=0)
    fig, ax = plt.subplots()
    ax.imshow(ch_combined_thresh_project, vmax=colormap_cutoff * ch_combined_thresh_project.max())


def zview(w, cell_based=False, filename=None):
    """
    make a cut through the colony
    :param w: colony object
    :param cell_based: Color according to segmentation and the z-centroids. This may look nicer depending on the quality of the segmentation. Much slower.
    :param filename: filename to save
    :return:
    """

    if cell_based:
        # go through each index and change value according to z-position
        b = np.zeros(w.ws.shape)
        for i1 in range(w.df.shape[0]):
        # for i1 in range(1,w.ws.max()+1):
            # if i1 % 100 == 0:
            #     print(i1)
            b[w.ws == w.df.iloc[i1].name] = w.df.iloc[i1].centroid[0]  # "name" is the index of the watershed, aka "cell_id"

    else:
        b = np.zeros(w.ws.shape, dtype=np.int)
        for i1 in range(b.shape[0]):
            b[i1] = i1

    wm = np.ma.masked_where(~w.mask, b)
    fig, ax = plt.subplots(figsize=(10,1))
    ax.imshow(wm[::-1, int(w.center[1]), :], cmap='copper_r')
    ax.axis('off')
    if filename is not None:
        fig.savefig(filename)


def make_iterable(a):
    try:
        len(a)
        return a
    except TypeError:
        return [a]


def nice_spines(ax, grid=True, dark=False):

    if grid:
        ax.grid(grid)

    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    for line in gridlines:
        line.set_linestyle('-')
        line.set_linewidth(.1)

    spines_to_remove = ['top', 'right']
    spines_to_keep = ['bottom', 'left']
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    almost_black = '#262626'

    for spine in spines_to_remove:
        ax.spines[spine].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    for spine in spines_to_keep:
        ax.spines[spine].set_linewidth(0.5)
        if not dark:
            ax.spines[spine].set_color(almost_black)
        else:
            ax.spines[spine].set_color('#ffffff')

def running_mean(x, N):
    # http://stackoverflow.com/questions/13728392/moving-average-or-running-mean
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


def browse_stack(w, plot_number=True):

    n = len(w.ws)
    # self._contrast_stretch
    # plt.figu
    def view_image(i):
        plt.imshow(w.ws[i], cmap=random_cmap(seed=123), alpha=0.7, vmin=0,vmax=1024)

        peak_counter = 1
        seeds_in_current_z = w.peaks[w.peaks[:, 0] == i][:, 1:]  # find seeds that are in the current z

        # if show_labels:
        #     assert len(self.df) == self.peaks.shape[0]
        #     for ipeaks in range(len(self.df)):
        #         ax[1].text(self.df.iloc[ipeaks].centroid[2], self.df.iloc[ipeaks].centroid[1],
        #                    str(self.df.iloc[ipeaks].label),
        #                    color='r', fontsize=22)
        for ipeaks in range(len(w.df)):
            if int(np.round(w.df.iloc[ipeaks].centroid[0])) == i:
                if plot_number:
                    plt.text(w.df.iloc[ipeaks].centroid[2], w.df.iloc[ipeaks].centroid[1],
                                    str(w.df.iloc[ipeaks].label),
                                    color='r', fontsize=22)
                plt.plot(w.df.iloc[ipeaks].centroid[2], w.df.iloc[ipeaks].centroid[1], 'xr')


        # for ipeaks in range(seeds_in_current_z.shape[0]):
        #     plt.text(seeds_in_current_z[ipeaks, 1], seeds_in_current_z[ipeaks, 0], str(peak_counter), color='r', fontsize=22)
        #     peak_counter += 1

        # plt.plot(seeds_in_current_z[:, 1], seeds_in_current_z[:, 0], 'xr')
        plt.xlim(w.peaks[:, 2].min() - 20, w.peaks[:, 2].max() + 20)
        plt.ylim(w.peaks[:, 1].min() - 20, w.peaks[:, 1].max() + 20)
        plt.axis('off')
        plt.show()

    interact(view_image, i=(0, n - 1))


def visualize_stack(w, plot_number=True, cmap='viridis'):
    """visualize arbitrary stack"""

    n = len(w)
    cmax, cmin = w.max(), w.min()
    def view_image(i):
        plt.imshow(w[i], cmap=cmap, vmin=cmin,vmax=cmax)
        plt.axis('off')
        plt.show()

    interact(view_image, i=(0, n - 1))


def random_cmap(seed=None, return_darker=False, n=1024):

    """
    make random colormap, good for plotting segmentation

    :param seed: seed for random colormap
    :param return_darker: also return a darker version
    :param n: number of colors
    :return: colormap(s)
    """
    np.random.seed(seed)
    random_array = np.random.rand(n, 3)
    random_array[0, :] = 1.
    random_array2 = np.zeros((n, 4))
    random_array2[:, :3] = random_array
    random_array2[:, 3] = .8
    # random_array2[0, 3] = 1. # keep 0 white

    if return_darker:
        return mpl.colors.ListedColormap(random_array), mpl.colors.ListedColormap(random_array2)
    else:
        return mpl.colors.ListedColormap(random_array)


@jit
def relabel_peaks(a):
    to_add = 1
    for i1 in range(a.shape[0]):
        for i2 in range(a.shape[1]):
            for i3 in range(a.shape[2]):
                if a[i1,i2,i3] == 1:
                    a[i1,i2,i3] += to_add
                    to_add += 1
    return a


def scatter3d(w, channel, cut=0, xyz_scale=None, only_selected_cells=False, vmax=None, filename=None, axis=False, plot_vector=None):

    almost_black = '#262626'

    index = w._get_indices(only_selected_cells)
    pts = np.vstack(w.df.centroid_rescaled[index])

    if channel is not None:
        c = np.vstack(w.df[channel][index])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if channel is not None:
        ax.scatter(pts[:, 1], pts[:, 2], pts[:, 0], c=c, vmax=vmax, cmap='inferno', marker='o', s=40)
    else:
        ax.scatter(pts[:, 1], pts[:, 2], pts[:, 0], c=almost_black, marker='o', s=40)
    if not axis:
        ax.axis('off')
    ax.grid('on')
    ax.set_axis_bgcolor('white')


    if cut:
        x = np.array([0, 100, 100, 0]) + pts[:, 1].mean()
        y = np.array([0, 0, 0, 0]) + pts[:, 2].mean()
        z = [110, 110, 0, 0] - 2 * pts[:, 0].mean() - 10
        verts = [list(zip(x, y, z))]
        poly = Poly3DCollection(verts)
        ax.add_collection3d(poly, zs='z')
        poly.set_alpha(0.1)

    if plot_vector is not None:
        if len(plot_vector) != 6:
            raise RuntimeError('array must be of length 6, (x,y,z location and u,v,w direction)')
        soa = plot_vector
        vlength = np.linalg.norm(soa)
        X, Y, Z, U, V, W = zip(soa)
        ax.quiver(X, Y, Z, U, V, W, pivot='tail', length=vlength, arrow_length_ratio=0.3 / vlength)

    # ax.auto_scale_xyz([30, 160], [30, 160], [0, 130])
    # if xyz_scale is None:
    #     xrange = pts[1,:].max() - pts[1,:].min()
    #     yrange = pts[2,:].max() - pts[2,:].min()
    #     zrange = pts[0,:].max() - pts[0,:].min()
    # maxrange = max(xrange, yrange, zrange)
    # xrange = [pts[1, :].mean() - maxrange/2, pts[1, :].mean() + maxrange/2]
    # yrange = [pts[2, :].mean() - maxrange/2, pts[2, :].mean() + maxrange/2]
    # zrange = [pts[0, :].mean() - maxrange/2, pts[0, :].mean() + maxrange/2]

    ax.auto_scale_xyz(xrange, yrange, zrange)
    # ax.auto_scale_xyz([30, 160], [30, 160], [0, 130])

    if filename is not None:
        fig.savefig(filename)


def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

def radius_of_gyration(centroid_rescaled):
    """

    :param centroid_rescaled:
    :return: center of mass, radius of gyration
    """

    center_of_mass = np.vstack(centroid_rescaled).mean(axis=0)
    rog = np.sqrt(np.sum((np.vstack(centroid_rescaled) - center_of_mass) ** 2, axis=1).mean())

    return center_of_mass, rog
