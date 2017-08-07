#!/usr/bin/env python

import os
import numpy
import glob
import shutil
import codecs
import time
import sys

os.environ['GLOG_minloglevel'] = '2'
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
from scipy.misc import imresize, imread
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import zoom
from tempfile import NamedTemporaryFile
from contextlib import contextmanager
from collections import namedtuple
import upsample
import rotate
import expdir
from vecquantile import QuantileVector

caffe.set_mode_gpu()
caffe.set_device(0)

def create_sample(
        directory, dataset, definition, weights, mean, blobs, quantiles,
        output_mat=True,
        resolution=8 * 1024, buffersize=1024,
        resolution_boost=2, colordepth=3,
        stride=1,
        rotation_seed=None, rotation_power=1,
        limit=None, split=None,
        batch_size=16, ahead=4,
        cl_args=None, verbose=True):
    # If we're already done, skip it!
    ed = expdir.ExperimentDirectory(directory)
    if all(ed.has_mmap(blob=b, part='sample') for b in blobs) and all(
            ed.has_mmap(blob=b, part='count') for b in blobs):
        return

    '''
    directory: where to place the probe_conv5.mmap files.
    data: the AbstractSegmentation data source to draw upon
    definition: the filename for the caffe prototxt
    weights: the filename for the caffe model weights
    mean: to use to normalize rgb values for the network
    blobs: ['conv3', 'conv4', 'conv5'] to probe
    '''
    if verbose:
        print 'Opening dataset', dataset
    ds = loadseg.SegmentationData(args.dataset)
    if verbose:
        print 'Opening network', definition
    np = caffe_pb2.NetParameter()
    with open(definition, 'r') as dfn_file:
        text_format.Merge(dfn_file.read(), np)
    net = caffe.Net(definition, weights, caffe.TEST)
    input_blob = net.inputs[0]
    input_dim = net.blobs[input_blob].data.shape[2:]
    data_size = ds.size(split)
    if limit is not None:
        data_size = min(data_size, limit)
    # Organize labels into categories
    categories = ds.category_names()
    primarycat = ds.primary_categories_per_index()
    num_classes = len(primarycat)

    # Make sure we have a directory to work in
    ed.ensure_dir()

    if verbose:
         print 'Creating new sample.'
    sample_all = {}
    sample_cat = {}
    sample_match = {}
    fieldmap = {}
    smap = {}
    cmap = {}
    rot = None
    if rotation_seed is not None:
        rot = {}
    # buffersize = min(buffersize, resolution)

    # We need to initialize quantiles lazily to avoid fork memory consumption
    def initialize_maps():
        if initialize_maps.done:
            return
        for blob in blobs:
            num_units = net.blobs[blob].data.shape[1]
            # Find the shortest path through the network to the target blob
            fieldmap[blob], _ = upsample.composed_fieldmap(np.layer, blob)
            sample_match[blob] = [
                QuantileVector(depth=num_units, buffersize=buffersize,
                    resolution=resolution, dtype='float32')
                        for _ in range(ds.label_size(None))]
            sample_cat[blob] = [
                QuantileVector(depth=num_units, buffersize=buffersize,
                    resolution=resolution * resolution_boost, dtype='float32')
                        for _ in categories]
            sample_all[blob] = (
                QuantileVector(depth=num_units, buffersize=buffersize,
                    resolution=resolution * resolution_boost, dtype='float32'))
            # Compute random rotation for each blob, if needed
            if rot is not None:
                rot[blob] = rotate.randomRotationPowers(
                    num_units, [rotation_power], rotation_seed)[0]
        initialize_maps.done = True
    initialize_maps.done = False

    # Periodically make write quantiles to snapshots
    def make_snapshot():
        print 'Writing snapshot'
        qlist = numpy.linspace(0, 1, num=quantiles)
        for blob in blobs:
            # Gather quantiles into a single vector
            qvlist = sample_match[blob] + sample_cat[blob] + [sample_all[blob]]
            if blob not in cmap:
                cmap[blob] = ed.open_mmap(
                    blob=blob, part='count', mode='w+', shape=len(qvlist))
            counts = cmap[blob]
            if blob not in smap:
                smap[blob] = ed.open_mmap(
                    blob=blob, part='sample', mode='w+',
                    shape=(len(qvlist), qvlist[0].depth, len(qlist)))
            samples = smap[blob]
            # Save counts and compute quantiles
            for row, qv in enumerate(qvlist):
                counts[row] = qv.size
                samples[row, :, :] = qv.quantiles(qlist)
        if output_mat:
            print 'Saving mat files'
            from scipy.io import savemat
            savemat(ed.filename('%s-count.mat' % blob), {
                'count': cmap[blob]
                })
            savemat(ed.filename('%s-sample.mat' % blob), {
                'sample': smap[blob]
                })

    # The main loop
    if verbose:
         print 'Beginning work.'
    pf = loadseg.SegmentationPrefetcher(ds, categories=['image'] + categories,
            split=split, once=True, batch_size=batch_size, ahead=ahead)
    start_time = time.time()
    last_batch_time = start_time
    batch_size = 0
    index = 0
    for batch in pf.batches():
        initialize_maps() # Defer intialization to save mem for fork.
        batch_time = time.time()
        rate = index / (batch_time - start_time + 1e-15)
        batch_rate = batch_size / (batch_time - last_batch_time + 1e-15)
        last_batch_time = batch_time
        if verbose:
            print 'netsample index', index, 'items per sec', batch_rate, rate
            sys.stdout.flush()
        inp = numpy.array([
            loadseg.normalize_image(rec['image'], mean)
            for rec in batch])
        batch_size = len(inp)
        if limit is not None and index + batch_size > limit:
            # Truncate last if limited
            batch_size = limit - index
            inp = inp[:batch_size]
        if colordepth == 1:
            inp = numpy.mean(inp, axis=1, keepdims=True)
        net.blobs[input_blob].reshape(*(inp.shape))
        net.blobs[input_blob].data[...] = inp
        result = net.forward(blobs=blobs)
        if rot is not None:
            for key in out.keys():
                result[key] = numpy.swapaxes(numpy.tensordot(
                        rot[key], result[key], axes=((1,), (1,))), 0, 1)
        offset = stride // 2
        # print 'Computation done'
        # Handle each image individually on cpu
        for i, rec in enumerate(batch):
            sw, sh = [(rec[k] + stride-offset-1) // stride for k in ['sw', 'sh']]
            for blob in blobs:
                up = upsample.upsampleL(fieldmap[blob], result[blob][i],
                    shape=input_dim, scaleshape=(sh, sw))
                up.shape = (up.shape[0], numpy.prod(up.shape[1:]))
                pixact = up.transpose() # (linearized pixels, units)
                has_cat = numpy.zeros(len(categories), 'bool')
                qvm = sample_match[blob]
                qvc = sample_cat[blob]
                qva = sample_all[blob]
                for cat in categories:
                    label_group = rec[cat]
                    if len(numpy.shape(label_group)) % 2 == 0:
                        label_group = [label_group]
                    for label in label_group:
                        if len(numpy.shape(label)) == 0:
                            # The whole-image-is-one-label case
                            if label > 0:
                                has_cat[primarycat[label]] = True
                                # print 'Adding for label %d' % label
                                qvm[label].add(pixact)
                        else:
                            pixlabel = label[offset::stride, offset::stride
                                    ].ravel()
                            for c in numpy.bincount(pixlabel).nonzero()[0]:
                                if c > 0:
                                    has_cat[primarycat[c]] = True
                                    # print 'Adding for label %d' % c
                                    qvm[c].add(pixact[pixlabel == c, :])
                for c in has_cat.nonzero()[0]:
                    # print 'Adding for category %d' % c
                    qvc[c].add(pixact)
                # print 'Adding for all'
                qva.add(pixact)
            if index + i in [100, 1000, 10000]:
                make_snapshot()
        index += batch_size
        if index >= data_size:
            break

    assert index == data_size, (
            "Data source should return evey item once %d %d." %
            (index, data_size))
    make_snapshot()
    print 'Finalizing mmaps'
    for blob in blobs:
        ed.finish_mmap(cmap[blob])
        ed.finish_mmap(smap[blob])


def write_readme_file(args, ed, verbose):
    '''
    Writes a README.txt that describes the settings used to geenrate the ds.
    '''
    with codecs.open(ed.filename('README.txt'), 'w', 'utf-8') as f:
        def report(txt):
            f.write('%s\n' % txt)
            if verbose:
                print txt
        title = '%s network probe' % ed.basename()
        report('%s\n%s' % (title, '=' * len(title)))
        for key, val in args:
            if key == 'cl_args':
                if val is not None:
                    report('Command-line args:')
                    for ck, cv in vars(val).items():
                        report('    %s: %r' % (ck, cv))
            report('%s: %r' % (key, val))
        report('\ngenerated at: %s' % time.strftime("%Y-%m-%d %H:%M"))
        try:
            label = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
            report('git label: %s' % label)
        except:
            pass

if __name__ == '__main__':
    import sys
    import traceback
    import argparse
    try:
        import loadseg

        parser = argparse.ArgumentParser(description=
            'Probe a caffe network and save results in a directory.')
        parser.add_argument(
                '--directory',
                default='.',
                help='output directory for the net probe')
        parser.add_argument(
                '--blobs',
                nargs='*',
                help='network blob names to sample')
        parser.add_argument(
                '--definition',
                help='the deploy prototext defining the net')
        parser.add_argument(
                '--weights',
                help='the caffemodel file of weights for the net')
        parser.add_argument(
                '--mean',
                nargs='*', type=float,
                help='mean values to subtract from input')
        parser.add_argument(
                '--dataset',
                help='the directory containing the dataset to use')
        parser.add_argument(
                '--split',
                help='the split of the dataset to use')
        parser.add_argument(
                '--limit',
                type=int, default=None,
                help='limit dataset to this size')
        parser.add_argument(
                '--batch_size',
                type=int, default=8,
                help='the batch size to use')
        parser.add_argument(
                '--ahead',
                type=int, default=4,
                help='number of batches to prefetch')
        parser.add_argument(
                '--rotation_seed',
                type=int, default=None,
                help='the seed for the random rotation to apply')
        parser.add_argument(
                '--rotation_power',
                type=float, default=1.0,
                help='the power of hte random rotation')
        parser.add_argument(
                '--colordepth',
                type=int, default=3,
                help='set to 1 for grayscale')
        parser.add_argument(
                '--quantiles',
                type=int, default=1001,
                help='number of quantiles to estimate')
        parser.add_argument(
                '--output_mat',
                type=int, default=0,
                help='1 to output mat files')
        parser.add_argument(
                '--resolution',
                type=int, default=4 * 1024,
                help='resolution of quantile estimators')
        parser.add_argument(
                '--stride',
                type=int, default=1,
                help='stride for downsampling of label pixels')
        parser.add_argument(
                '--buffersize',
                type=int, default=None,
                help='chunk size for quantile estimators')
        args = parser.parse_args()
        
        create_sample(
            args.directory, args.dataset, args.definition, args.weights,
            numpy.array(args.mean, dtype=numpy.float32), args.blobs,
            quantiles=args.quantiles, resolution=args.resolution,
            buffersize=args.buffersize,
            output_mat=args.output_mat,
            batch_size=args.batch_size, ahead=args.ahead, limit=args.limit,
            colordepth=args.colordepth,
            rotation_seed=args.rotation_seed,
            rotation_power=args.rotation_power,
            split=args.split, cl_args=args, verbose=True)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
