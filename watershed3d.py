import sys
import os
import re
import h5py
import matplotlib as mpl
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
import skimage
from scipy.ndimage.filters import maximum_filter, gaussian_filter
# from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from skimage.morphology import ball, binary_erosion
# from skimage.segmentation import random_walker
import numpy as np
import skimage.io as io
import matplotlib.pylab as plt
import pandas as pd
from matplotlib.pyplot import rcParams
from scipy.ndimage.measurements import center_of_mass  # note this gives center of mass in column, row format,
# i.e. to plot need to reverse

io.use_plugin('tifffile')
sys.path.append('/Users/jakob/Documents/RU/Code/segment')


class Ws3d(object):
    """
    Main watershed 3d class. Takes an image file and expects a _Probabillites.h5 file from Ilastik. If there is an
    object classifier from Ilastik, uses this for seeding. Otherwise tries to guess blobs for seeding.
    """

    def __init__(self, imagefile, xy_scale=1.0, z_scale=1.5):
        """
        Initializes Ws3d with an imagefile. Can set the relative x,y and z scales of the z-stack, which is important
        for the blob
        detection.

        :param imagefile:
        :param xy_scale:
        :param z_scale:
        """

        self.fontsize = 16
        self.filename = imagefile
        self.image_stack = io.imread(imagefile)
        self.image_dim = self.image_stack.ndim
        if self.image_dim == 2:
            print('image is two-dimensional')
            self.x_size, self.y_size = self.image_stack.shape
            self.z_size, self.z_scale = None, None

        elif self.image_dim == 3:
            print('image is three-dimensional')
            self.z_size, self.x_size, self.y_size = self.image_stack.shape
            if self.z_size > self.x_size or self.z_size > self.y_size:
                raise RuntimeError("ERROR: z must be the first and smallest dimension")
                self.z_scale = z_scale
        else:
            raise RuntimeError("Image is neither 2 or 3 dimensional")

        self.xy_scale = xy_scale

        self.peaks = None
        self.peak_array = None
        self.probability_map = None
        self.ws = None
        self.df = None
        # self.good_nuclei = None
        self.center = None
        # self.channels = {}
        self.channels_image = {}
        self.probability_map_filename = re.sub('\.tif$', '_Probabilities.h5', self.filename, re.IGNORECASE)
        if not os.path.isfile(self.probability_map_filename):
            print('ERROR: file', self.probability_map_filename, 'not found. Did you do the Ilastik classification?')
            return
        self.mask = None
        self.mask_selected = None

        # try to locate object prediction
        self.filename_op = re.sub('\.tif$', '_Object Predictions.h5', self.filename, re.IGNORECASE)
        self.op = None
        self.have_op = True

        if not os.path.isfile(self.filename_op):
            print('No object prediction file ({:s}) found - segmenting without.'.format(self.filename_op))
            self.have_op = False

        self.find_center_of_colony()

    def load_mask(self, prob=0.5, foreground_index=1):
        """
        Load the probability mask and object prediction if it exists.

        :param prob: Set the probability for which
        :param foreground_index: Set the index that contains the foreground (i.e. did you use the first or the second label in Ilastik for the background? Default is that the first label (i.e. 0) is background and 1 corresponds to foreground.
        """

        if self.image_dim == 2:
            with h5py.File(self.probability_map_filename, 'r') as h:  # r+: read/write, file must exist
                self.probability_map = h['exported_data'][:][:, :, :, foreground_index]  # index 1 is the nuclei
        elif self.image_dim == 3:
            with h5py.File(self.probability_map_filename, 'r') as h:  # r+: read/write, file must exist
                self.probability_map = h['exported_data'][:][:, :, foreground_index]  # index 1 is the nuclei
        else:
            raise RuntimeError

        print("shape", self.probability_map.shape, self.image_stack.shape)
        if not self.probability_map.shape == self.image_stack.shape:
            print("ERROR: probability map does not have same dimensions as image")
            return

        self.mask = self.probability_map > prob
        print('loaded probability map')
        # self.probability_map[self.probability_map < prob] = 0.

        # load object prediction if there
        if self.image_dim == 3:
            if self.have_op:
                with h5py.File(re.sub('\.tif$', '_Object Predictions.h5', self.filename, re.IGNORECASE), 'r') as h:
                    self.op = np.swapaxes(np.squeeze(h['exported_data']), 2, 0)  # object prediction
                print('loaded object prediction')
        else:
            raise NotImplementedError("not implemented for 2d")

    def plot_probability_map(self, z=None, contrast_stretch=False, figsize=None):

        if self.probability_map is None:
            print('probability map not loaded, cannot plot - use load_mask first')
            return

        self.grid_plot(self.probability_map, z, contrast_stretch, figsize)

    def grid_plot(self, image_to_plot, z=None, contrast_stretch=False, figsize=None, cmap=None):
        """
        Plot all z-slices

        :param image_to_plot:
        :param z: can select a specific z to plot. Ignore for 2D
        :param contrast_stretch: enhance contrast
        :param figsize: set the size of the figure when plotting a single z-slice, default is (10,10)
        :param cmap: colormap
        :return:
        """

        # try:
        #     assert image_to_plot.ndim == 3
        # except AssertionError:
        #     print('Error: image to plot must be a z-stack, i.e. must have dimension .ndim == 3')
        #     return
        # except AttributeError:
        #     print('Error: image to plot is not in the correct format - needs to be a numpy array.')
        #     return

        if z is not None and image_to_plot.ndim != 2:
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
                for n in range(self.z_size):
                    i = n // ncols
                    j = n % ncols

                    if contrast_stretch:
                        axes[i, j].imshow(self._contrast_stretch(image_to_plot[n]), interpolation='None', cmap=cmap)
                    else:
                        ax = axes[i, j].imshow(image_to_plot[n], interpolation='None', cmap=cmap)
                        ax.set_clim(0, image_to_plot.max())

                # Remove empty plots
                for ax in axes.ravel():
                    if not (len(ax.images)):
                        fig.delaxes(ax)
                fig.tight_layout()

    def intensity_histogram(self):

        with plt.style.context('ggplot'):
            fig, ax = plt.subplots(figsize=(8, 4))
            flat = self.image_stack.flatten()
            ax.hist(flat, log=True, bins=np.arange(0, flat.max(), 200), range=(0, flat.max()))

            _ = ax.set_title('Min value: %i, Max value: %i, Image shape: %s \n'
                             % (self.image_stack.min(),
                                self.image_stack.max(),
                                self.image_stack.shape))

    def filter_probability_map(self, sigma=None):
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

        pm = self.probability_map.copy()
        print(pm.shape)
        pm = gaussian_filter(pm, sigma=sigma)
        # normalize back to [0,1]
        pm[pm > 1] = 1.
        pm[pm < 0.05] = 0.
        return pm

    def segment(self, min_distance=2, sigma=(2, 2, 6), do_not_use_object_classifier=True):
        """
        Segment the image.

        :param min_distance:
        :param sigma:
        :param do_not_use_object_classifier:
        """

        if self.probability_map is None:
            print('probability map not loaded, cannot segment - use load_mask first')

        # have object classifier or don't
        if self.have_op and not do_not_use_object_classifier:
            raise NotImplementedError

        # NO object classifier, use smoothed map and find peaks in probability map
        else:
            # if min_distance is None:
            #     min_distance = 2

            # get smoothed probability map
            pm = self.filter_probability_map(sigma=sigma)
            # self.peaks = peak_local_max(pm, min_distance=min_distance)

            # distance = ndi.distance_transform_edt(pm)
            # self.peak_array = peak_local_max(distance, min_distance=min_distance, indices=False)

            self.peak_array = peak_local_max(pm, min_distance=min_distance, indices=False)
            self.peaks = np.transpose(np.nonzero(self.peak_array))  # same as above with indices True, but need that too

            markers = ndi.label(self.peak_array)[0]

            self.ws = skimage.morphology.watershed(-self.image_stack, markers, mask=self.mask)
            # self.ws = mh.cwatershed(-self.image_stack,markers)
            # self.ws *= self.mask

            # make a dataframe with some of the regionprops in it
            # rp = skimage.measure.regionprops(self.ws, intensity_image=self.image_stack)
            # columns = ('area', 'total_intensity', 'mean_intensity', 'centroid')
            # rpd = [[i1.area, i1.mean_intensity * i1.area, i1.mean_intensity, i1.coords.mean(axis=0)] for i1 in rp]
            # indices = [i1.label for i1 in rp]
            # indices = pd.Index(indices, name='cell_id')
            # self.df = pd.DataFrame(rpd, index=indices, columns=columns)
            self.df = self._regionprops_to_dataframe(self.ws, self.image_stack)

            print('segmentation done, found', self.peaks.shape[0], 'cells')

    @staticmethod
    def _regionprops_to_dataframe(ws, image_stack):
        """
        Return pd.DataFrame with relevant entries from watershed image. Keep static so can use it for other images as
        well.

        :param ws: watershed array
        :param image_stack: image array
        :return: pd.DataFrame with entries ('area', 'total_intensity', 'mean_intensity', 'centroid') and index 'cell_id'
        """

        rp = skimage.measure.regionprops(ws, intensity_image=image_stack)
        columns = ('area', 'total_intensity', 'mean_intensity', 'centroid')
        rpd = [[i1.area, i1.mean_intensity * i1.area, i1.mean_intensity, i1.coords.mean(axis=0)] for i1 in rp]
        indices = [i1.label for i1 in rp]
        indices = pd.Index(indices, name='cell_id')
        print('len rp=', len(rp))
        return pd.DataFrame(rpd, index=indices, columns=columns)

    def show_segmentation(self, z=None, contrast_stretch=True, figsize=None, seed=130):
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
            figsize = (6, 12)
        # self.grid_plot(self.image_stack,  z=z, contrast_stretch=contrast_stretch, figsize=figsize)
        _, ax = plt.subplots(1, 2, figsize=figsize)

        # show seeds on the original image
        if z is None:
            # plot the maximum intensity projection
            mip = self.image_stack.max(axis=0)
        else:
            mip = self.image_stack[z]
        if contrast_stretch:
            ax[0].imshow(self._contrast_stretch(mip), cmap=plt.cm.viridis)
        else:
            ax[0].imshow(mip, cmap=plt.cm.viridis)

        ax[0].set_title('maximum intensity projection')
        ax[0].plot(self.peaks[:, 2], self.peaks[:, 1], 'xr')
        ax[0].set_xlim(self.peaks[:, 2].min() - 20, self.peaks[:, 2].max() + 20)
        ax[0].set_ylim(self.peaks[:, 1].min() - 20, self.peaks[:, 1].max() + 20)

        # show watershed. Do it on the dataframe since may have deleted cells
        if z is None:
            ax[1].imshow(self.ws.max(axis=0), cmap=self.myrandom_cmap(seed=seed))
        else:
            ax[1].imshow(self.ws[z], cmap=self.myrandom_cmap(seed=seed))
        ax[1].plot(self.peaks[:, 2], self.peaks[:, 1], 'xr')
        ax[1].set_xlim(self.peaks[:, 2].min() - 20, self.peaks[:, 2].max() + 20)
        ax[1].set_ylim(self.peaks[:, 1].min() - 20, self.peaks[:, 1].max() + 20)

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
            self.df.hist(color='k', bins=50, alpha=0.6)
            plt.suptitle('before selection of nuclei')
            self.df[self.df.good_nuclei].hist(color='k', bins=50, alpha=0.6)
            plt.suptitle('after selection of nuclei')

            _, ax = plt.subplots()
            w2 = self.ws.copy()  # note the copy here, otherwise next line also affects self.ws
            # w2[~np.in1d(self.ws, np.array(self.df[self.good_nuclei].index)).reshape(self.ws.shape)] = 0 # set to 0
            w2[~np.in1d(self.ws, np.array(self.df[self.df['good_nuclei']].index)).reshape(self.ws.shape)] = 0 # set to 0
            self.mask_selected = w2 > 0
            # all elements that are NOT in the good nuclei

            # show watershed after selections.
            if z is None:
                ax.imshow(w2.max(axis=0), cmap=self.myrandom_cmap(seed=seed), origin='lower')
            else:
                ax.imshow(w2[z], cmap=self.myrandom_cmap(seed=seed), origin='lower')
            ax.set_title('after selection of good nuclei')

    def apply_to_channels(self, filename, channel_id, remove_background=True):
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
            im = self.remove_background(im)
        assert self.ws.shape == im.shape
        # self.channels[channel_id] = self._regionprops_to_dataframe(self.ws, im)
        channel_df = self._regionprops_to_dataframe(self.ws, im)
        channel_df.columns = ['area', channel_id, 'mean_intensity', 'centroid']

        try:
            self.df.drop(channel_id, axis=1, inplace=1)
        except ValueError:  # does not exist
            pass

        self.df = pd.concat([self.df, channel_df[channel_id]], axis=1)
        self.channels_image[channel_id] = im

    def find_center_of_colony(self):
        """
        find center of mass
        """

        self.center = np.array(center_of_mass(self.image_stack))

    def radial_intensity(self, channel_id, only_selected_nuclei=False, plot=True):
        """
        Get radial intensity either for all nuclei or only selected ones. This uses the pixel information,
        not the segmentation.

        :param channel_id:
        :param only_selected_nuclei:
        :param plot:
        :return:
        """

        data = self.channels_image[channel_id]

        if only_selected_nuclei:
            mask = self.mask_selected
        else:
            mask = self.mask
        data[~mask] = 0  # set all False values to 0

        x, y = np.indices(data.shape[1:])  # note changed NON-inversion of x and y
        r = np.sqrt((x - self.center[1]) ** 2 + (y - self.center[2]) ** 2)
        r = r.astype(np.int)
        # now make 3d
        r = np.tile(r, (data.shape[0], 1, 1))

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
            print("WARNING: there were empty bins, i.e. at some radii there seem to be no cells.")
            radialprofile[np.isnan(radialprofile)] = 0.  # set these to 0

        if plot:
            fig, ax = plt.subplots()
            ax.plot(np.arange(radialprofile.shape[0])*self.xy_scale, radialprofile)
            ax.set_ylim([0., ax.get_ylim()[1]])
            ax.set_xlim([0., ax.get_xlim()[1]/np.sqrt(2)])  # plot sqrt(2) less far because this is in the corners
            ax.set_xlabel('distance ($\mu m$)', fontsize=self.fontsize)
            ax.set_ylabel(str(channel_id) + ' intensity', fontsize=self.fontsize)
            # where there is no colony anyway
            self.nice_spines(ax)

        return np.arange(radialprofile.shape[0])*self.xy_scale, radialprofile

    def _get_indices(self, only_selected_cells):
        if only_selected_cells:
            return self.df.good_nuclei
        else:
            return self.df.index  # all

    def dot_plot(self, channel_id, colormap_cutoff=0.5, only_selected_cells=False):
        """
        Dot-plot as in Warmflash et al.

        :param channel_id:
        :param colormap_cutoff: percentage of maximum for cutoff. Makes smaller differences more visible.
        :param only_selected_cells:
        :return:
        """

        index = self._get_indices(only_selected_cells)

        fig, ax = plt.subplots()
        cax = ax.scatter(np.vstack(self.df.centroid[index].values.flat)[:, 1], np.vstack(
            self.df.centroid[index].values.flat)[:, 2], c=self.df[channel_id][index].values,
                         s=40, edgecolors='none', cmap=plt.cm.viridis, vmax=colormap_cutoff*self.df[
                channel_id][index].values.max())
        self.nice_spines(ax)
        ax.autoscale(tight=1)
        ax.set_aspect('equal')
        fig.colorbar(cax)

    def copy_data_to_clipboard(self):
        """
        copy data to clipboard

        """
        self.df.to_clipboard()

    def radial_profile_per_cell(self, channel_id, nbins=30, plot=True, only_selected_cells=False):
        """

        :param channel_id:
        :param nbins: number of bins
        :param plot:
        :param only_selected_cells:
        :return:
        """
        index = self._get_indices(only_selected_cells)

        x = np.vstack(self.df.centroid[index].values.flat)[:, 1]
        y = np.vstack(self.df.centroid[index].values.flat)[:, 2]
        i = self.df[channel_id][index].values

        r = np.sqrt((x-self.center[1])**2+(y-self.center[2])**2)
        # r = np.round(r).astype(np.int)

        # n = np.bincount(r, i)
        # n2 = np.bincount(r)
        # xn = np.arange(r.min(),r.max())
        n, xn = np.histogram(r, bins=nbins, weights=i)
        n2, _ = np.histogram(r, bins=nbins)

        if plot:
            _, ax = plt.subplots()
            ax.step(xn[:-1] - xn[0], n/n2, where='mid')
            ax.fill_between(xn[:-1] - xn[0], n/n2, alpha=0.2, step='mid')
            ax.set_xlabel('r', fontsize=self.fontsize)
            ax.set_ylabel(channel_id, fontsize=self.fontsize)
            # ax.set_xlim([0, ax.get_xlim()[1]])
            self.nice_spines(ax)
        return xn, n/n2

    def coexpression_per_cell(self, channel_id1, channel_id2, only_selected_cells=False):
        """
        Scatter plot visualizing co-expression of two channels, with each datapoint the intensity of one cell.

        :param channel_id1:
        :param channel_id2:
        :param only_selected_cells:
        :return:
        """

        index = self._get_indices(only_selected_cells)

        ch1 = self.df[channel_id1][index].values
        ch2 = self.df[channel_id2][index].values

        fig, ax = plt.subplots()
        ax.scatter(ch1, ch2, edgecolors='none', c='k', alpha=0.8)
        self.nice_spines(ax)
        ax.set_xlabel(channel_id1, fontsize=self.fontsize)
        ax.set_ylabel(channel_id2, fontsize=self.fontsize)
        ax.autoscale(tight=1)

    def coexpression_per_pixel(self, channel_id1, channel_id2, downsample=10, only_selected_cells=False):

        """
        Scatter plot visualizing co-expression of two channels, with each datapoint the intensity of one nuclear pixel.

        :param channel_id1:
        :param channel_id2:
        :param downsample: Usually have a lot of point, so can only use ever downsample'th point.
        :param only_selected_cells:
        :return:
        """

        ch1 = self.channels_image[channel_id1]
        ch2 = self.channels_image[channel_id2]

        if only_selected_cells:
            mask = self.mask_selected
        else:
            mask = self.mask

        fig, ax = plt.subplots()
        ax.scatter(ch1[mask > 0][::downsample], ch2[mask > 0][::downsample], marker='o', edgecolors='none', c='k',
                   alpha=0.8, s=4)
        self.nice_spines(ax)
        ax.set_xlabel(channel_id1, fontsize=self.fontsize)
        ax.set_ylabel(channel_id2, fontsize=self.fontsize)
        ax.autoscale(tight=1)

    # static methods
    @staticmethod
    def remove_background(im, n=1000):
        """
        basic method to remove background, returns background subtracted image

        :param im: image
        :param n: lowest non-zero n pixels are used to estimate background
        :return: background subtracted image
        """
        imm = im.mean(axis=0)
        # following in comments has a bug
        # im -= np.partition(imm[imm.nonzero()], n)[:n].mean().astype(im.dtype)
        # im[im<0.] = 0.  # set negatives to 0
        # return im
        return im - np.partition(imm[imm.nonzero()], n)[:n].mean()

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

        if return_darker:
            return mpl.colors.ListedColormap(random_array), mpl.colors.ListedColormap(random_array2)
        else:
            return mpl.colors.ListedColormap(random_array)

    @staticmethod
    def nice_spines(ax, grid=True):

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
            ax.spines[spine].set_color(almost_black)

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
