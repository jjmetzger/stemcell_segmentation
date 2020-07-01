# find colonies and crop them
# 2016-2020, Jakob Metzger, jmetzger@rockefeller.edu

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage import exposure
import skimage
import os
import re
import tifffile as tiff
import numpy as np
import pylab as pl
from IPython import display


def crop_colonies(files, show_colonies=True, save_colonies=False, desired_shape=380, downsample_to_8_bit=True, image_bits=16, save_dir='', downsample=1):

    if tiff.imread(files[0]).shape[0] > 10:
        raise RuntimeError('First image dimension should be number of channels, but is > 10')
    if len(tiff.imread(files[0]).shape)!=3:
        raise RuntimeError('Image dimension should be 3 (channels, x, y')

    bit_factor = int(2**image_bits/2**8)
    pattern = re.compile('(\S+).tif')
    save_dir_trunk = save_dir  # directory to save to

    if show_colonies:
        pl.figure(figsize=(3, 3))

    # go through all the files in the list "files" from above
    for i11, f in enumerate(files):
        # print(i11, f)
        line_name = pattern.findall(f)[0]

        save_dir = os.path.join(line_name, 'crop')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_dir = os.path.join(save_dir_trunk, line_name)
        colonies = np.load(os.path.join(save_dir, 'colony_bounds.npy'))

        img_all = tiff.imread(f).transpose(1, 2, 0)  # assuming color is first index
        for color in range(img_all.shape[-1])[:1]:
            img = img_all[:, :, color]

            for cntr, colony in enumerate(colonies):
                xindx, yindx, success = crop(colony, desired_shape)
                xindx = np.array(xindx)
                yindx = np.array(yindx)
                xindx = xindx.astype(np.int)
                yindx = yindx.astype(np.int)
                # print(np.diff(xindx), np.diff(yindx))

                # TODO: Some cropping return negative x and y values. Fix this.
                if (xindx[0] < 0 or xindx[1] < 0 or yindx[0] < 0 or yindx[0] < 0):
                    success = False
                if success:  # could crop, i.e. not too large
                    cropped_img = img[xindx[0]:xindx[1], yindx[0]:yindx[1]]
                    #                 print('crop', cropped_img.shape)
                    if xindx[1] > img.shape[0]:
                        cropped_img = np.pad(cropped_img, ((0, xindx[1] - img.shape[0]), (0, 0)),
                                             'minimum')  # pad using minimum value
                    if xindx[1] < desired_shape:
                        cropped_img = np.pad(cropped_img, ((desired_shape - xindx[1], 0), (0, 0)),
                                             'minimum')  # pad using minimum value
                    if yindx[1] > img.shape[1]:
                        cropped_img = np.pad(cropped_img, ((0, 0), (0, yindx[1] - img.shape[1])),
                                             'minimum')  # pad using minimum value
                    if yindx[1] < desired_shape:
                        cropped_img = np.pad(cropped_img, ((0, 0), (desired_shape - yindx[1], 0)),
                                             'minimum')  # pad using minimum value

                    assert cropped_img.shape[0] == desired_shape
                    assert cropped_img.shape[1] == desired_shape
                    if not cropped_img.shape == (desired_shape, desired_shape):
                        print('---WARNING----')
                        print(os.path.join(line_name, 'crop', 'C' + str(color+1) + '_' + str(cntr) + '.tif'))
                        print('dimensions incorrect', cropped_img.shape, xindx, yindx)

                    if show_colonies:
                        pl.subplot(1, 1, 1)
                        pl.imshow(cropped_img)
                        pl.xlim(0, desired_shape)
                        pl.ylim(0, desired_shape)
                        pl.xlim(0, desired_shape)
                        pl.ylim(0, desired_shape)
                        display.clear_output(wait=True)
                        display.display(pl.gcf())
                        plt.gca().cla()

                    if save_colonies:
                        filename = os.path.join(line_name, 'crop', 'C' + str(color+1) + '_' + str(cntr) + '.tif')

                        if downsample_to_8_bit:
                            to_save = (cropped_img[::downsample, ::downsample] // bit_factor).astype(np.uint8)
                        else:
                            to_save = cropped_img[::downsample, ::downsample]
                        tiff.imsave(filename, to_save)


def get_colony_outlines(files, channel_for_segmentation, plot=True, lower_threshold=0.5, min_area=22000, min_aspect_ratio=0.7, save_directory=''):

    pattern = re.compile('(\S+).tif')  # finds all tiff files and prepares for removal of .tif
    save_dir_trunk = save_directory  # where to put the directories with colony outlines

    channel_for_segmentation = channel_for_segmentation - 1

    for f in files:
        print('processing file', f, flush=True)
        line_name = pattern.findall(f)[-1]
        save_dir = os.path.join(save_dir_trunk, line_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        img = tiff.imread(f)[channel_for_segmentation] #read tif file
        colonies = find_colonies(img, lower_threshold=lower_threshold, min_area=min_area, aspect_ratio_min=min_aspect_ratio, plot=plot) # on DAPI, index=0
        colonies = np.array(colonies)
        np.save(os.path.join(save_dir, 'colony_bounds.npy'), colonies)
        print('saved to', os.path.join(save_dir, 'colony_bounds.npy'))


# get number of channels and channel names from files
def get_channels(f):
    condition = os.path.split(f)[1].split('_')[0]
    channels = os.path.split(f)[1].split('_')[1]
    added_identifiers = os.path.split(f)[1].split('_')[2:]

    if not added_identifiers:
        channels, extension = os.path.splitext(channels)
    else:
        if isinstance(added_identifiers, list):
            added_identifiers = ''.join(added_identifiers)
        added_identifiers, extension = os.path.splitext(added_identifiers)

    channels = channels.split('-')

    return condition, channels, added_identifiers, extension


def save_colony_outlines(files, channel_for_segmentation='DAPI', lower_threshold=0.5, downsample=None, min_area=22000,
                         max_area=1e16):
    # save colony outlines, usually better to use the DAPI files
    # this makes a folder for each .tif file and saves the outlines of the colonies in a compressed .npy file
    # it also makes a plot of all the colonies
    # if the colonies don't get properly identified, try with a different lower_threshold

    pattern = re.compile('(\S+).tif')  # finds all tiff files and prepares for removal of .tif
    save_dir_trunk = ''  # where to put the directories with colony outlines

    #     color_for_segmentation = 4 #DAPI
    for f in files:
        # find the index corresponding to the channel used for segmentation
        index_for_segmentation = next(
            i for i, v in enumerate(get_channels(f)[1]) if v.lower() == channel_for_segmentation.lower())
        print(f, flush=True)
        line_name = pattern.findall(f)[-1]
        save_dir = os.path.join(save_dir_trunk, line_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        #     img = np.pad(imread(f)[color_for_segmentation,::2,::2],((1,),(1,)),'constant') #read tif file
        img = tiff.imread(f)[index_for_segmentation, ::downsample, ::downsample]  # read tif file
        colonies = find_colonies(img, lower_threshold=lower_threshold, min_area=min_area, max_area=max_area,
                                 aspect_ratio_min=0.7)  # on DAPI, index=0
        colonies = np.array(colonies)
        np.save(os.path.join(save_dir, 'colony_bounds.npy'), colonies)
        print('saved to', os.path.join(save_dir, 'colony_bounds.npy'))


def find_colonies(image, lower_threshold=0.5, min_area=100, max_area=1e15, aspect_ratio_min=None, plot=True):
    """
    find colonies

    :param image:
    :param lower_threshold: amount by which to lower Otsu threshold for safer identification
    :param min_area: minimum area in pixels
    :param max_area: maximum area in pixels
    :param aspect_ratio_min: minimum aspect ratio
    :param plot: plot segmentation, can also specify size of image here
    :return: list of coordinates of each colony
    """

    if not isinstance(plot, bool):
        if len(plot) !=2:
            raise RuntimeError('plot argument must be True, False, or image size of length 2')

    image = exposure.equalize_adapthist(image)
    image = skimage.filters.gaussian(image, sigma=10)
    thresh = threshold_otsu(image)*lower_threshold
    bw = closing(image > thresh, square(3))

    # remove artifacts connected to image border
    cleared = clear_border(bw)

    # label image regions
    label_image = label(cleared)
    image_label_overlay = label2rgb(label_image, image=image)

    if plot:
        if not isinstance(plot, bool):
            figsize = plot
        else:
            figsize = (5,5)
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(image_label_overlay)

    colony_list = []
    for region in regionprops(label_image):
        # take regions with large enough areas

        if min_area < region.area < max_area:

            # draw rectangle around segmented colonies
            minr, minc, maxr, maxc = region.bbox
            aspect_ratio = (maxc - minc)/(maxr - minr)
            # print(aspect_ratio)
            if aspect_ratio_min is not None:
                if (aspect_ratio < aspect_ratio_min) or (1./aspect_ratio < aspect_ratio_min):
                    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                              fill=False, edgecolor='red', linewidth=2)
                else:
                    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                              fill=False, edgecolor='green', linewidth=2)
                    colony_list.append([minr, minc, maxr, maxc])

            else:
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='green',
                                          linewidth=2)
                colony_list.append([minr, minc, maxr, maxc])

            if plot:
                ax.add_patch(rect)

    if plot:
        ax.set_axis_off()
        plt.tight_layout()

    return colony_list


def crop(colony, desired_shape):

    # shp = 160  # want this shape
    shp = desired_shape
    success = True
    xlen, ylen = colony[2] - colony[0], colony[3] - colony[1]
    # pad until have shp
    if xlen > shp:
        print('WARNING: colony bigger than desired shape!')
        xindx = (colony[0], colony[2])
        success = False
    else:
        if shp - xlen % 2 == 0:  # even
            xindx = (colony[0] - (shp - xlen) / 2, colony[2] + (shp - xlen) / 2)
        else:
            xindx = (colony[0] - (shp - xlen - 1) / 2, colony[2] + (shp - xlen - 1) / 2 + 1)
    if ylen > shp:
        print('WARNING: colony bigger than desired shape!')
        yindx = (colony[1], colony[3])
        success = False
    else:
        if shp - ylen % 2 == 0:  # even
            yindx = (colony[1] - (shp - ylen) / 2, colony[3] + (shp - ylen) / 2)
        else:
            yindx = (colony[1] - (shp - ylen - 1) / 2, colony[3] + (shp - ylen - 1) / 2 + 1)

    return xindx, yindx, success


def segment_colonies(files, channels, desired_shape=380, make8bit=False, save_colonies=True, show_colonies=False):
    # set to True if you want to show the colonies, False if not (faster)
    # when the eccentricity of the rosette ellipse is too large, will be discarded (red box)

    if not isinstance(channels, list):
        channels = [channels]

    pattern = re.compile('(\S+).tif')
    save_dir_trunk = ''  # directory to save to

    desired_shape = desired_shape  # shape in pixels of the outline around the colonies
    downsample1 = 1  # when reading
    downsample = 2

    bit_factor = 0

    # go through all the files in the list "files" from above
    for i11, f in enumerate(files):
        print(i11, f)

        # find the index corresponding to the channel used for segmentation
        index_for_segmentation = [next(
            i for i, v in enumerate(get_channels(f)[1]) if v.lower() == chch.lower()) for chch in channels]

        # print('get indices', index_for_segmentation)

        line_name = pattern.findall(f)[0]

        save_dir = os.path.join(line_name, 'crop')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_dir = os.path.join(save_dir_trunk, line_name)
        colonies = np.load(os.path.join(save_dir, 'colony_bounds.npy'))

        img_all = tiff.imread(f).transpose(1, 2, 0)[::downsample1, ::downsample1, :]
        for color, color_name in zip(index_for_segmentation, channels):
            img = img_all[:, :, color]

            if make8bit:
                if img.dtype == np.uint8:
                    bit_factor = 1
                elif img.dtype == np.uint16:
                    if img.max() < 2**12:
                        print('Warning: bit depth of image file is 16, but appears to be only 12')
                        bit_factor = 2**(12-8)
                    else:
                        bit_factor = 2**(16-8)
                else:
                    raise NotImplementedError('downsampling to 8 bit only implemented for uint8 and uint16')

            for cntr, colony in enumerate(colonies[:]):
                xindx, yindx, success = crop(colony, desired_shape)
                xindx = np.array(xindx)
                yindx = np.array(yindx)
                xindx = xindx.astype(np.int)
                yindx = yindx.astype(np.int)
                # print(np.diff(xindx), np.diff(yindx))

                # Some cropping return negative x and y values
                if xindx[0] < 0 or xindx[1] < 0 or yindx[0] < 0 or yindx[0] < 0:
                    success = False
                if success:  # could crop, i.e. not too large
                    cropped_img = img[xindx[0]:xindx[1], yindx[0]:yindx[1]]
                    #                 print('crop', cropped_img.shape)
                    if xindx[1] > img.shape[0]:
                        cropped_img = np.pad(cropped_img, ((0, xindx[1] - img.shape[0]), (0, 0)),
                                             'minimum')  # pad using minimum value
                    if xindx[1] < desired_shape:
                        cropped_img = np.pad(cropped_img, ((desired_shape - xindx[1], 0), (0, 0)),
                                             'minimum')  # pad using minimum value
                    if yindx[1] > img.shape[1]:
                        cropped_img = np.pad(cropped_img, ((0, 0), (0, yindx[1] - img.shape[1])),
                                             'minimum')  # pad using minimum value
                    if yindx[1] < desired_shape:
                        cropped_img = np.pad(cropped_img, ((0, 0), (desired_shape - yindx[1], 0)),
                                             'minimum')  # pad using minimum value

                    assert cropped_img.shape[0] == desired_shape
                    assert cropped_img.shape[1] == desired_shape
                    if not cropped_img.shape == (desired_shape, desired_shape):
                        print('---WARNING----')
                        print(os.path.join(line_name, 'crop', 'C' + str(color) + '_' + str(cntr) + '.tif'))
                        print('dimensions incorrect', cropped_img.shape, xindx, yindx)

                    if show_colonies:
                        pl.subplot(1, 1, 1)
                        pl.imshow(cropped_img)
                        pl.xlim(0, desired_shape)
                        pl.ylim(0, desired_shape)
                        pl.xlim(0, desired_shape)
                        pl.ylim(0, desired_shape)

                        display.clear_output(wait=True)
                        display.display(pl.gcf())
                        plt.gca().cla()
                    #                     time.sleep(0.01) # time between frames in seconds

                    if save_colonies:
                        filename = os.path.join(line_name, 'crop', str(color_name) + '_' + str(cntr) + '.tif')

                        if not make8bit:
                            to_save = (cropped_img[::downsample, ::downsample])
                        else:
                            to_save = (cropped_img[::downsample, ::downsample] // bit_factor).astype(np.uint8)

                        tiff.imsave(filename, to_save)
