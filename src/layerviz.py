'''
viewprobe creates visualizations for a certain eval.
'''

import os
import re
import numpy
import upsample
import loadseg
from scipy.misc import imread, imresize, imsave
from loadseg import normalize_label
import expdir
import tightcrop

class LayerViz:
    def __init__(self, ed, blob, ds=None):
        net_info = ed.load_info()
        if ds is None:
            ds = loadseg.SegmentationData(net_info.dataset)
        info = ed.load_info(blob=blob)
        self.shape = info.shape
        self.fieldmap = info.fieldmap
        self.input_dim = net_info.input_dim
        # Load the raw activation data
        self.blobdata = ed.open_mmap(blob=blob, shape=self.shape, mode='r')
        # Load the blob quantile data and grab thresholds
        self.quantdata = ed.open_mmap(blob=blob, part='quant-*',
                shape=(self.shape[1], -1), mode='r')
        # And load imgmax
        self.imgmax = ed.open_mmap(blob=blob, part='imgmax',
                shape=(ds.size(), self.shape[1]), mode='r')
        # Figure out tally level that was used.
        self.level = ed.glob_number(
                'tally-*.mmap', blob=blob, decimal=True)
        # Figure the top-acivated images for each unit
        self.top_indexes = self.imgmax.argsort(
                axis=0)[::-1,:].transpose()
        # Save away the dataset
        self.ds = ds

    def instance_data(self, i, normalize=True):
        record, shape = self.ds.resolve_segmentation(
                self.ds.metadata(i), categories=None)
        if normalize:
            default_shape = (1, ) + shape
            record = dict((cat, normalize_label(dat, default_shape))
                    for cat, dat in record.items())
        return record, shape

    # Generates a mask at the "lp.level" quantile, upsampled.
    def activation_mask(self, unit, index, shape=None):
        if shape is None:
            shape = self.input_dim
        blobdata = self.blobdata
        fieldmap = self.fieldmap
        quantdata = self.quantdata
        threshold = quantdata[unit, int(round(quantdata.shape[1] * self.level))]
        up = upsample.upsampleL(
                fieldmap, blobdata[index:index+1, unit],
                shape=self.input_dim, scaleshape=shape)[0]
        mask = up > threshold
        return mask

    # Makes an iamge using the mask
    def activation_visualization(self, unit, index, alpha=0.2):
        image = imread(self.ds.filename(index))
        mask = self.activation_mask(unit, index, image.shape[:2])
        return (mask[:, :, numpy.newaxis] * (1 - alpha) + alpha) * image

    # The key thing: generates a unit visualization image array
    # that can be saved using imsave if desired.
    def unit_visualization(self, unit, count=1, shape=None,
            gap=3, gridwidth=None, tight=False, saveas=None):
        '''
        Returns a visualization for the given unit in the layer,
        consisting of count subimages of the given shape,
        separated by margins of "gap" size, arranged in a
        horizontal line, or a grid of width gridwidth.
        '''
        if shape is None:
            shape = self.input_dim
        if gridwidth is None:
            gridwidth = count
            gridheight = 1
        else:
            gridheight = (count + gridwidth - 1) // gridwidth
        tiled = numpy.full(
            ((shape[0] + gap) * gridheight - gap,
             (shape[1] + gap) * gridwidth - gap, 3), 255, dtype='uint8')
        for x, index in enumerate(self.top_indexes[unit][:count]):
            row = x // gridwidth
            col = x % gridwidth
            vis = self.activation_visualization(unit, index)
            if tight:
                vis = tightcrop.best_tightcrop(vis, shape)
            elif not numpy.array_equal(vis.shape, shape):
                vis = imresize(vis, shape)
            tiled[row*(shape[0]+gap):row*(shape[0]+gap)+shape[0],
                  col*(shape[1]+gap):col*(shape[1]+gap)+shape[1],:] = vis
        if saveas is not None:
            imsave(saveas, tiled)
        return tiled

    # TODO: consider returning segmentation visualizations etc.

if __name__ == '__main__':
    import sys
    import traceback
    import argparse

    parser = argparse.ArgumentParser(description=
        'Visualize a specific unit of a probed model.')
    parser.add_argument(
            '--directory',
            default='.', required=True,
            help='directory containing probed units')
    parser.add_argument(
            '--blob', required=True,
            help='layer to visualize')
    parser.add_argument(
            '--unit', type=int, required=True,
            help='unit number to visualize')
    parser.add_argument(
            '--output', required=True,
            help='output filename for visualization')
    parser.add_argument(
            '--count',
            type=int, default=1,
            help='number of images to show in the strip')
    parser.add_argument(
            '--gap',
            type=int, default=3,
            help='white pixels between images')
    parser.add_argument(
            '--height',
            type=int, default=None,
            help='the image height to use')
    parser.add_argument(
            '--gridwidth',
            type=int, default=None,
            help='number of images to show in a single row')
    parser.add_argument(
            '--tight',
            type=int, default=0,
            help='set to 1 to apply tight cropping and zooming')
    args = parser.parse_args()

    ed = expdir.ExperimentDirectory(args.directory)
    lv = LayerViz(ed, args.blob)
    lv.unit_visualization(args.unit,
            count=args.count,
            shape=args.height and (args.height, args.height),
            gap=args.gap,
            gridwidth=args.gridwidth,
            tight=args.tight,
            saveas=args.output)

