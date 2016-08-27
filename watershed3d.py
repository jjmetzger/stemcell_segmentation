import sys
import os
import re
import h5py
import imutils
import matplotlib as mpl
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
import skimage
from scipy.ndimage.filters import maximum_filter, gaussian_filter
# from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from skimage.morphology import ball, binary_erosion
from skimage.segmentation import random_walker
import numpy as np
import skimage.io as io
import matplotlib.pylab as plt
import mahotas as mh

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

        self.filename = imagefile
        self.image_stack = io.imread(imagefile)
        self.z_size, self.x_size, self.y_size = self.image_stack.shape

        self.xy_scale = xy_scale
        self.z_scale = z_scale

        self.peaks = None
        self.peak_array = None
        self.probability_map = None
        self.ws = None
        self.probability_map_filename = re.sub('\.tif$', '_Probabilities.h5', self.filename, re.IGNORECASE)
        if not os.path.isfile(self.probability_map_filename):
            print('ERROR: file', self.probability_map_filename, 'not found. Did you do the Ilastik classification?')
        self.mask = None

        # try to locate object prediction
        self.filename_op = re.sub('\.tif$', '_Object Predictions.h5', self.filename, re.IGNORECASE)
        self.op = None
        self.have_op = True

        if not os.path.isfile(self.filename_op):
            print('No object prediction file ({:s}) found - segmenting without.'.format(self.filename_op))
            self.have_op = False

    def load_mask(self, prob=0.5, foreground_index=1):
        """
        Load the probability mask and object prediction if it exists.

        :param prob: Set the probability for which
        :param foreground_index: Set the index that contains the foreground (i.e. did you use the first or the
        second label in Ilastik for the background? Default is that the first label (i.e. 0) is background and 1
        corresponds to foreground.
        """

        with h5py.File(self.probability_map_filename, 'r') as h:  # r+: read/write, file must exist
            self.probability_map = h['exported_data'][:][:, :, :, foreground_index]  # index 1 is the nuclei

        self.mask = self.probability_map > prob
        print('loaded probability map')
        # self.probability_map[self.probability_map < prob] = 0.

        # load object prediction if there
        if self.have_op:
            with h5py.File(re.sub('\.tif$', '_Object Predictions.h5', self.filename, re.IGNORECASE), 'r') as h:
                self.op = np.swapaxes(np.squeeze(h['exported_data']), 2, 0)  # object prediction
            print('loaded object prediction')

    def plot_probability_map(self, z=None, contrast_stretch=False, figsize=None):

        if self.probability_map is None:
            print('probability map not loaded, cannot plot - use load_mask first')
            return

        self.grid_plot(self.probability_map, z, contrast_stretch, figsize)

    def grid_plot(self, image_to_plot, z=None, contrast_stretch=False, figsize=None, cmap=None):
        """
        Plot all z-slices

        :param image_to_plot:
        :param z: can select a specific z to plot
        :param contrast_stretch: enhance contrast
        :param figsize: set the size of the figure when plotting a single z-slice, default is (10,10)
        :return:
        """

        try:
            assert image_to_plot.ndim == 3
        except AssertionError:
            print('Error: image to plot must be a z-stack, i.e. must have dimension .ndim == 3')
            return
        except AttributeError:
            print('Error: image to plot is not in the correct format - needs to be a numpy array.')
            return

        if cmap is None:
            cmap = plt.cm.plasma
        # cmap = plt.cm.Greys
        if figsize is None:
            figsize = (10, 10)
        if z is not None:
            fig, axes = plt.subplots(figsize=figsize)
            if contrast_stretch:
                axes.imshow(imutils.contrast_stretch(image_to_plot[z]), cmap=cmap)
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
                    axes[i, j].imshow(imutils.contrast_stretch(image_to_plot[n]), interpolation='None', cmap=cmap)
                else:
                    ax = axes[i, j].imshow(image_to_plot[n], interpolation='None', cmap=cmap)
                    ax.set_clim(0, image_to_plot.max())

            # Remove empty plots
            for ax in axes.ravel():
                if not (len(ax.images)):
                    fig.delaxes(ax)
            fig.tight_layout()

    def histogram_plot(self):

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

        try:
            assert len(sigma) == 3
        except AssertionError:
            print('sigma needs to have 3 elements for the three dimensions.')
            return
        except TypeError:
            print('setting sigma to default (2, 6, 6)')
            sigma = (2, 6, 6)

        pm = self.probability_map
        pm = gaussian_filter(pm, sigma=sigma)
        # normalize back to [0,1]
        pm[pm > 1] = 1.
        pm[pm < 0.05] = 0.
        return pm

    def segment(self, min_distance=None, sigma=None, do_not_use_object_classifier=False):
        """
        Segment the image.

        :param min_distance:
        :param sigma:
        :param do_not_use_object_classifier:
        :return:
        """

        if self.probability_map is None:
            print('probability map not loaded, cannot segment - use load_mask first')

        # have object classifier or don't
        if self.have_op and not do_not_use_object_classifier:
            raise NotImplementedError

        # NO object classifier, use smoothed map and find peaks in probability map
        else:
            if min_distance is None:
                min_distance = 2

            # get smoothed probability map
            pm = self.filter_probability_map(sigma=sigma)
            # self.peaks = peak_local_max(pm, min_distance=min_distance)

            # distance = ndi.distance_transform_edt(pm)
            # self.peak_array = peak_local_max(distance, min_distance=min_distance, indices=False)

            self.peak_array = peak_local_max(pm, min_distance=min_distance, indices=False)
            self.peaks = np.transpose(np.nonzero(self.peak_array))  # same as above with indices True, but need that too

            markers = ndi.label(self.peak_array)[0]

            self.ws = skimage.morphology.watershed(self.image_stack, markers, mask=self.mask)

            print('segmentation done, found', self.peaks.shape[0], 'cells')

    def show_segmentation(self, z=None, contrast_stretch=True, figsize=None, seed=123):
        """
        Show segmentation on the maximum intensity projection.

        :param z:
        :param contrast_stretch:
        :param figsize:
        :return:
        """

        if self.peaks is None:
            print('do not have cell positions, run segment first')

        if figsize is None:
            figsize = (6,12)
        # self.grid_plot(self.image_stack,  z=z, contrast_stretch=contrast_stretch, figsize=figsize)
        _, ax = plt.subplots(1,2,figsize=figsize)

        if z is None:
            # plot the maximum intensity projection
            mip = self.image_stack.max(axis=0)
        else:
            mip = self.image_stack[z]
        if contrast_stretch:
            ax[0].imshow(imutils.contrast_stretch(mip), cmap=plt.cm.viridis)
        else:
            ax[0].imshow(mip, cmap=plt.cm.viridis)

        ax[0].set_title('maximum intensity projection')
        ax[0].plot(self.peaks[:, 2], self.peaks[:, 1], 'xr');
        ax[0].set_xlim(self.peaks[:, 2].min() - 20, self.peaks[:, 2].max() + 20)
        ax[0].set_ylim(self.peaks[:, 1].min() - 20, self.peaks[:, 1].max() + 20)

        if z is None:
            ax[1].imshow(self.ws.max(axis=0), cmap=self.myrandom_cmap(seed=seed))
        else:
            ax[1].imshow(self.ws[z], cmap=self.myrandom_cmap(seed=seed))
        ax[1].plot(self.peaks[:, 2], self.peaks[:, 1], 'xr');
        ax[1].set_xlim(self.peaks[:, 2].min() - 20, self.peaks[:, 2].max() + 20)
        ax[1].set_ylim(self.peaks[:, 1].min() - 20, self.peaks[:, 1].max() + 20)

    def detect_peaks(self, neighborhood=None):
        # http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array

        """
        Takes an image and detect the peaks usingthe local maximum filter.
        Returns a boolean mask of the peaks (i.e. 1 when
        the pixel's value is the neighborhood maximum, 0 otherwise)
        """

        # define an 8-connected neighborhood
        if neighborhood is None:
            neighborhood = generate_binary_structure(len(self.probability_map.shape), 2)

        # apply the local maximum filter; all pixel of maximal value
        # in their neighborhood are set to 1
        local_max = maximum_filter(self.probability_map, footprint=neighborhood) == self.probability_map
        # local_max is a mask that contains the peaks we are
        # looking for, but also the background.
        # In order to isolate the peaks we must remove the background from the mask.

        # we create the mask of the background
        background = (self.probability_map == 0)

        # a little technicality: we must erode the background in order to
        # successfully subtract it form local_max, otherwise a line will
        # appear along the background border (artifact of the local maximum filter)
        eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

        # we obtain the final mask, containing only peaks,
        # by removing the background from the local_max mask
        detected_peaks = local_max - eroded_background

        return detected_peaks

    def myrandom_cmap(self, seed=None, return_darker=False, n=1024):

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
