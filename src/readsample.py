from expdir import ExperimentDirectory
from loadseg import SegmentationData
import numpy
import os

def calculate_corr(ed, ds, blob):
    info = ed.load_info(blob=blob)
    num_units = info.shape[1]
    num_classes = ds.label_size()
    num_cats = len(ds.category_names())
    primary_cat = ds.primary_categories_per_index()

    fsamp = ed.open_mmap(blob=blob, part='sample',
            shape=(num_classes + num_cats + 1, info.shape[1], -1))
    fcount = ed.open_mmap(blob=blob, part='count')
    # Shape: (classes, units, buckets)
    sample = fsamp[:num_classes]
    count = fcount[:num_classes]
    sample_cat = fsamp[ds.label_size():-1]
    count_cat = fcount[ds.label_size():-1]
    sample_all = fsamp[-1]
    count_all = fcount[-1]
    eps = 1e-100

    # First, take the mean and std of each unit (per-unit, per-category)
    sample_cat_mean = numpy.mean(sample_cat, axis=2)[
            [primary_cat[c] for c in range(num_classes)], :]
    sample_cat_std = numpy.std(sample_cat, axis=2)[
            [primary_cat[c] for c in range(num_classes)], :]

    # Now take the mean and std of each groundtruth (per-class)
    count_denom = count_cat[[primary_cat[c] for c in range(num_classes)]]
    count_mean = count / count_denom
    count_std = numpy.sqrt((count * (1.0 - count_mean) ** 2 +
            (count_denom - count) * count_mean ** 2) / count_denom)
    # Patch no-label
    count_std[0] = 1.0
    ground_true = (1.0 - count_mean) / count_std
    ground_false = -count_mean / count_std

    # Now figure the correlation
    # Two terms:
    # (1) a = Expected value of normed activation * normed groundtruth (true)
    #         (mean(sample) - sample_cat_mean) / sample_cat_std * ground_true
    # (2) b = Expected value of normed activation * normed groundtruth (false)
    #         mean(out) = mean(cat) * count_cat - mean(sample) * count /
    #               (count_cat - count_in)
    #         b = (mean(out) - sample_cat_mean) / sample_cat_std * ground_false
    # (3) Combined as a * count + b * (cat - count) / cat
    inside_mean = numpy.mean(sample, axis=2)
    terma = (inside_mean - sample_cat_mean) / sample_cat_std * (
            ground_true[:,None])
    outside_mean = (sample_cat_mean * count_denom[:,None] -
            inside_mean * count[:,None]) / (count_denom - count)[:,None]
    termb = (outside_mean - sample_cat_mean) / sample_cat_std * (
            ground_true[:,None])

    result = (terma * count[:,None] + termb * (count_denom - count)[:,None]) / (
            count_denom[:,None])
    return result

def calculate_iou(ed, ds, blob, thresholds):
    info = ed.load_info(blob=blob)
    num_units = info.shape[1]
    num_classes = ds.label_size()
    num_cats = len(ds.category_names())
    primary_cat = ds.primary_categories_per_index()

    fsamp = ed.open_mmap(blob=blob, part='sample',
            shape=(num_classes + num_cats + 1, info.shape[1], -1))
    fcount = ed.open_mmap(blob=blob, part='count')
    sample = fsamp[:num_classes]
    count = fcount[:num_classes]
    sample_cat = fsamp[ds.label_size():-1]
    count_cat = fcount[ds.label_size():-1]
    sample_all = fsamp[-1]
    count_all = fcount[-1]
    eps = 1e-100

    ls = numpy.linspace(0.0, 1.0, sample_all.shape[1])

    result = numpy.zeros((len(thresholds), num_units, num_classes),
            dtype='float32')

    levels = numpy.zeros((num_units, len(thresholds)))
    for u in range(num_units):
        levels[u] = levels_at(sample_all[u], ls, thresholds)

    for i, threshold in enumerate(thresholds):
        level = levels[:, i]
        intersection = numpy.array([[
            fraction_over(sample[c, u], ls, level[u]) * count[c]
                    for c in range(num_classes)]
                for u in range(num_units)])
        intersection_cat = numpy.array([[
            fraction_over(sample_cat[c, u], ls, level[u]) * count_cat[c]
                    for c in range(num_cats)]
                for u in range(num_units)])
        union = numpy.array([[
            count[c] + intersection_cat[u, primary_cat[c]] - intersection[u, c]
                    for c in range(num_classes)]
                for u in range(num_units)])
        iou = (intersection + eps) / (union + eps)
        result[i] = iou
    return result

def binary_mutual_information(x, y, joint):
    """
    Returns mutual information of two binomial events given their
    marginal probabilities (x, y), and their joint probability (joint)
    """
    # Avoid recomputing common differences
    negx = 1.0 - x
    negy = 1.0 - y
    x_minus_joint = x - joint
    y_minus_joint = y - joint
    eps = 1e-100
    # Sum over four terms
    return sum(
            jchoice * numpy.log((jchoice + eps) / (xchoice * ychoice + eps))
        for xchoice, ychoice, jchoice in [
            (x, y, joint),
            (x, negy, x_minus_joint),
            (negx, y, y_minus_joint),
            (negx, negy, negx - y_minus_joint)])

def calculate_bmi(ed, ds, blob, thresholds):
    info = ed.load_info(blob=blob)
    num_units = info.shape[1]
    num_classes = ds.label_size()
    num_cats = len(ds.category_names())
    primary_cat = ds.primary_categories_per_index()

    fsamp = ed.open_mmap(blob=blob, part='sample',
            shape=(num_classes + num_cats + 1, info.shape[1], -1))
    fcount = ed.open_mmap(blob=blob, part='count')
    sample = fsamp[:num_classes]
    count = fcount[:num_classes]
    sample_cat = fsamp[ds.label_size():-1]
    count_cat = fcount[ds.label_size():-1]
    sample_all = fsamp[-1]
    count_all = fcount[-1]

    ls = numpy.linspace(0.0, 1.0, sample_all.shape[1])

    result = numpy.zeros((len(thresholds), num_units, num_classes),
            dtype='float32')

    levels = numpy.zeros((num_units, len(thresholds)))
    for u in range(num_units):
        levels[u] = levels_at(sample_all[u], ls, thresholds)
    # compute groundtruth fractions for each label
    truth = numpy.array([[
            count[c] / count_cat[primary_cat[c]]
                for c in range(num_classes)]]) 

    for i, threshold in enumerate(thresholds):
        print 'Computing mutual information at threshold %.4f' % threshold
        level = levels[:, i]
        # we divide differently for each category, so compute category-specific
        # activation fractions, same across all labels in the category
        acts_by_cat = numpy.array([[
            fraction_over(sample_cat[c, u], ls, level[u])
                    for c in range(num_cats)]
                for u in range(num_units)])
        acts = acts_by_cat[:, [primary_cat[c] for c in range(num_classes)]]
        # compute joint probabilities
        joint = numpy.array([[
            fraction_over(sample[c, u], ls, level[u]) * truth[0, c]
                    for c in range(num_classes)]
                for u in range(num_units)])
        mutual_information = binary_mutual_information(acts, truth, joint)
        result[i] = mutual_information
    return result


def binary_entropy(x):
    return sum(-xchoice * numpy.log(xchoice) for xchoice in [x, 1.0 - x])

def calculate_redundancy(ed, ds, blob, thresholds):
    info = ed.load_info(blob=blob)
    num_units = info.shape[1]
    num_classes = ds.label_size()
    num_cats = len(ds.category_names())
    primary_cat = ds.primary_categories_per_index()

    fsamp = ed.open_mmap(blob=blob, part='sample',
            shape=(num_classes + num_cats + 1, info.shape[1], -1))
    fcount = ed.open_mmap(blob=blob, part='count')
    sample = fsamp[:num_classes]
    count = fcount[:num_classes]
    sample_cat = fsamp[ds.label_size():-1]
    count_cat = fcount[ds.label_size():-1]
    sample_all = fsamp[-1]
    count_all = fcount[-1]

    ls = numpy.linspace(0.0, 1.0, sample_all.shape[1])

    result = numpy.zeros((len(thresholds), num_units, num_classes),
            dtype='float32')

    levels = numpy.zeros((num_units, len(thresholds)))
    for u in range(num_units):
        levels[u] = levels_at(sample_all[u], ls, thresholds)
    # compute groundtruth fractions for each label
    truth = numpy.array([[
            count[c] / count_cat[primary_cat[c]]
                for c in range(num_classes)]]) 

    for i, threshold in enumerate(thresholds):
        print 'Computing redundancy at threshold %.4f' % threshold
        level = levels[:, i]
        # we divide differently for each category, so compute category-specific
        # activation fractions, same across all labels in the category
        acts_by_cat = numpy.array([[
            fraction_over(sample_cat[c, u], ls, level[u])
                    for c in range(num_cats)]
                for u in range(num_units)])
        acts = acts_by_cat[:, [primary_cat[c] for c in range(num_classes)]]
        # compute joint probabilities
        joint = numpy.array([[
            fraction_over(sample[c, u], ls, level[u]) * truth[0, c]
                    for c in range(num_classes)]
                for u in range(num_units)])
        mutual_information = binary_mutual_information(acts, truth, joint)
        entropy = binary_entropy(truth)
        result[i] = mutual_information / entropy
    return result

def binary_variation_of_information(x, y, joint):
    """
    Returns mutual information of two binomial events given their
    marginal probabilities (x, y), and their joint probability (joint)
    """
    # Avoid recomputing common differences
    negx = 1.0 - x
    negy = 1.0 - y
    x_minus_joint = x - joint
    y_minus_joint = y - joint
    one_minus_all = negx - y_minus_joint
    eps = 1e-50
    # Sum over four terms
    terms = list(
            jchoice * numpy.log((xchoice * ychoice + eps) /
                                (jchoice * jchoice + eps))
        for xchoice, ychoice, jchoice in [
            (x, y, joint),
            (x, negy, x_minus_joint),
            (negx, y, y_minus_joint),
            (negx, negy, one_minus_all)])
    bestlab = numpy.argmin(sum(term[0] for term in terms))
    print 'terms for %d' % bestlab,
    for t in range(4):
        print terms[t][0,bestlab],
    print
    return sum(terms)

def calculate_voi(ed, ds, blob, thresholds):
    info = ed.load_info(blob=blob)
    num_units = info.shape[1]
    num_classes = ds.label_size()
    num_cats = len(ds.category_names())
    primary_cat = ds.primary_categories_per_index()

    fsamp = ed.open_mmap(blob=blob, part='sample',
            shape=(num_classes + num_cats + 1, info.shape[1], -1))
    fcount = ed.open_mmap(blob=blob, part='count')
    sample = fsamp[:num_classes]
    count = fcount[:num_classes]
    sample_cat = fsamp[ds.label_size():-1]
    count_cat = fcount[ds.label_size():-1]
    sample_all = fsamp[-1]
    count_all = fcount[-1]

    ls = numpy.linspace(0.0, 1.0, sample_all.shape[1])

    result = numpy.zeros((len(thresholds), num_units, num_classes),
            dtype='float32')

    levels = numpy.zeros((num_units, len(thresholds)))
    for u in range(num_units):
        levels[u] = levels_at(sample_all[u], ls, thresholds)
    # compute groundtruth fractions for each label
    truth = numpy.array([[
            count[c] / count_cat[primary_cat[c]]
                for c in range(num_classes)]]) 

    for i, threshold in enumerate(thresholds):
        print 'Computing variation of information at threshold %.4f' % threshold
        level = levels[:, i]
        # we divide differently for each category, so compute category-specific
        # activation fractions, same across all labels in the category
        acts_by_cat = numpy.array([[
            fraction_over(sample_cat[c, u], ls, level[u])
                    for c in range(num_cats)]
                for u in range(num_units)])
        acts = acts_by_cat[:, [primary_cat[c] for c in range(num_classes)]]
        # compute joint probabilities
        joint = numpy.array([[
            fraction_over(sample[c, u], ls, level[u]) * truth[0, c]
                    for c in range(num_classes)]
                for u in range(num_units)])
        variation = binary_variation_of_information(acts, truth, joint)
        result[i] = variation
    return result

def make_graphs(directory, dataset, blob):
    ed = ExperimentDirectory(directory)
    ds = SegmentationData(dataset)
    ed.ensure_dir('html/graph')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fractions = numpy.concatenate((
            numpy.logspace(-3, -2, 5, endpoint=False),
            numpy.linspace(0.01, 0.5, 40),))
    thresholds = 1.0 - fractions
    netname = os.path.basename(directory)
    corr = calculate_corr(ed, ds, blob)

    if True:
        num_units = corr.shape[1]
        # each unit has a few best labels at different ranges
        for u in range(num_units):
            best_class = numpy.nanargmax(corr[:, u])
            label = ds.name(None, best_class)
            print 'unit', (u+1), best_class, label, corr[best_class, u]

    if False:
        # iou is (num_thresholds, num_units, num_classes)
        iou = calculate_iou(ed, ds, blob, thresholds)
        num_units = iou.shape[1]
        # each unit has a few best labels at different ranges
        best_classes_per_thresh = numpy.nanargmax(iou, axis=2)
        for u in range(num_units):
            best_classes = numpy.unique(best_classes_per_thresh[:,u])
            # Figure 1: unweighted
            fig = plt.figure(num=None, figsize=(7,4), dpi=100)
            for c in best_classes:
                label = ds.name(None, c)
                plt.plot(fractions, iou[:,u,c], label=label)
                print 'unit', (u+1), label
            plt.legend(loc='lower right', fancybox=True, framealpha=0.7)
            plt.title('%s %s unit %d' % (netname, blob, u+1))
            plt.savefig(ed.filename('html/graph/samp-%s-%04d-iou.png' %
                (blob, u)))
            plt.close(fig)

    if False:
        # voi is (num_thresholds, num_units, num_classes)
        voi = calculate_voi(ed, ds, blob, thresholds)
        num_units = voi.shape[1]
        # Eliminate NaNs
        voi[numpy.isnan(voi)] = numpy.inf
        # now repeat for information
        # each unit has a few best labels at different ranges
        # note: exclude "none" class from consideration
        best_classes_per_thresh = numpy.nanargmin(voi[:,:,1:], axis=2) + 1
        for u in range(num_units):
            best_classes = numpy.unique(best_classes_per_thresh[:,u])
            # Figure 1: unweighted
            fig = plt.figure(num=None, figsize=(7,4), dpi=100)
            for c in best_classes:
                label = ds.name(None, c) or 'none'
                plt.plot(fractions, voi[:,u,c], label=label)
                print 'unit', (u+1), label
            plt.legend(loc='lower right', fancybox=True, framealpha=0.7)
            plt.title('%s %s unit %d' % (netname, blob, u+1))
            plt.savefig(ed.filename('html/graph/samp-%s-%04d-voi.png' %
                (blob, u)))
            plt.close(fig)


    if False:
        # bmi is (num_thresholds, num_units, num_classes)
        bmi = calculate_bmi(ed, ds, blob, thresholds)
        num_units = bmi.shape[1]
        # Eliminate NaNs
        bmi[numpy.isnan(bmi)] = -numpy.inf
        # now repeat for information
        # each unit has a few best labels at different ranges
        best_classes_per_thresh = numpy.nanargmax(bmi, axis=2)
        for u in range(num_units):
            best_classes = numpy.unique(best_classes_per_thresh[:,u])
            # Figure 1: unweighted
            fig = plt.figure(num=None, figsize=(7,4), dpi=100)
            for c in best_classes:
                label = ds.name(None, c) or 'none'
                plt.plot(fractions, bmi[:,u,c], label=label)
                print 'unit', (u+1), label
            plt.legend(loc='lower right', fancybox=True, framealpha=0.7)
            plt.title('%s %s unit %d' % (netname, blob, u+1))
            plt.savefig(ed.filename('html/graph/samp-%s-%04d-bmi.png' %
                (blob, u)))
            plt.close(fig)

    if False:
        # red is (num_thresholds, num_units, num_classes)
        red = calculate_redundancy(ed, ds, blob, thresholds)
        num_units = red.shape[1]
        # Eliminate NaNs
        red[numpy.isnan(red)] = -numpy.inf
        # now repeat for information
        # each unit has a few best labels at different ranges
        best_classes_per_thresh = numpy.nanargmax(red, axis=2)
        for u in range(num_units):
            best_classes = numpy.unique(best_classes_per_thresh[:,u])
            # Figure 1: unweighted
            fig = plt.figure(num=None, figsize=(7,4), dpi=100)
            for c in best_classes:
                label = ds.name(None, c) or 'none'
                plt.plot(fractions, red[:,u,c], label=label)
                print 'unit', (u+1), label
            plt.legend(loc='lower right', fancybox=True, framealpha=0.7)
            plt.title('%s %s unit %d' % (netname, blob, u+1))
            plt.savefig(ed.filename('html/graph/samp-%s-%04d-red.png' %
                (blob, u)))
            plt.close(fig)

#    best_label = numpy.nanargmax(iou, axis=1)
#    best_score = iou[numpy.arange(len(best_label)), best_label]
#    order = numpy.argsort(best_score)[::-1]
#    for u in order:
#        print 'unit %u: %s %f' % (
#                u+1, ds.name(None, best_label[u]), best_score[u])

def levels_at(array, ls, quant):
    """
    Given an array shape=(quantiles) of sampled quantiles
       (e.g., smallest first, largest last, median at midpoint)
    and a linspace shape=(quantiles) describing the quantile of each sample
    and a quant shape=(outputs) of desired quantiles,
    returns shape=(outputs) of estimated values at desired quantiles
    """
    return numpy.interp(quant, ls, array)

def fraction_under(array, ls, level):
    """
    Given an array shape=(quantiles) of sampled quantiles
       (e.g., largest first, smallest last, median at midpoint)
    and a linspace shape=(quantiles) describing the quantile of each sample
    and a level scalar to query
    returns estimated portion of values 0 <= result <= 1 under the given level
    """
    return numpy.interp([level], array, ls)[0]

def fraction_over(array, ls, level):
    """
    Given an array shape=(quantiles) of sampled quantiles
       (e.g., largest first, smallest last, median at midpoint)
    and a linspace shape=(quantiles) describing the quantile of each sample
    and a level scalar to query
    returns estimated portion of values 0 <= result <= 1 over the given level
    """
    return 1.0 - fraction_under(array, ls, level)

if __name__ == '__main__':
    import sys
    import traceback
    import argparse
    try:
        import loadseg

        parser = argparse.ArgumentParser(description=
            'Visualize a sampled network.')
        parser.add_argument(
                '--directory',
                default='.',
                help='output directory for the net probe')
        parser.add_argument(
                '--dataset',
                help='dataset directory')
        parser.add_argument(
                '--blobs',
                nargs='*',
                help='network blob names to sample')
#        parser.add_argument(
#                '--threshold',
#                type=float, default=.995,
#                help='threshold to apply')
        args = parser.parse_args()

        for blob in args.blobs:
            make_graphs(args.directory, args.dataset, blob)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
