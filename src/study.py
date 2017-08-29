'''
For generating a visualization that compares several representations.
'''

import os
import re
import numpy
import upsample
import loadseg
from scipy.misc import imread, imresize, imsave
from loadseg import normalize_label
import expdir
import layerviz
import bargraph

def choose_concept_units(layerlist, conceptlist):
    # Chooses a list of units in the layer for visualization
    concept_ranking = dict((c, []) for c in conceptlist)
    for directory, layer in layerlist:
        ed = expdir.ExperimentDirectory(directory)
        netname = ed.basename()
        for record in ed.load_csv(blob=layer, part='result'):
            label = record['label']
            if label in conceptlist:
                concept_ranking[label].append((
                    float(record['score']),
                    netname, layer,
                    int(record['unit']) - 1))
    for c in conceptlist:
        concept_ranking[c].sort()
        concept_ranking[c].reverse()
    return concept_ranking

def generate_study(od,
            layerlist,
            concepts=None,
            categories=None,
            top_n=5,
            imsize=None,
            imscale=72,
            imcount=1,
            threshold=0.04,
            barscale=None,
            include_hist=True,
            verbose=True):
    htmlfn = od.filename('html/study.html')
    print 'Generating html summary', htmlfn
    od.ensure_dir('html','image')
    html = [html_prefix]
    max_label_count = 0
    # First, precount the maximum number of unique labels
    for directory, layer in layerlist:
        ed = expdir.ExperimentDirectory(directory)
        records = ed.load_csv(blob=layer, part='result')
        max_label_count = max(max_label_count,
            len(set(record['label'] for record in records
                    if float(record['score']) >= threshold)))
    if barscale is None:
        barscale = imscale * top_n
    barwidth = float(barscale) / max_label_count
    for directory, layer in layerlist:
        print 'processing', directory, layer
        ed = expdir.ExperimentDirectory(directory)
        lv = layerviz.LayerViz(ed, layer)
        records = ed.load_csv(blob=layer, part='result')
        records.sort(key=lambda record: -float(record['score']))
        html.append('<div class="layer">')
        html.append('<div class="layergrid">')
        # Reoreder records to put unique labels first
        seen_count = {}
        if not categories:
            categories = []
        for record in records:
            seen = seen_count.get(record['label'], 0)
            record['seen_before'] = seen
            seen_count[record['label']] = seen + 1
        records.sort(key=lambda r: (
            (float(r['score']) < threshold),
            r['seen_before'],
            categories.index(r['category'])
                if r['category'] in categories else -1,
            -float(r['score'])))
        # Resort by IoU
        records = sorted(records[:top_n], key=lambda r: -float(r['score']))
        # Run through records in this sorted order.
        for record in records:
            print 'unit', record['unit'], record['label']
            unit = int(record['unit']) - 1
            imfn = 'image/%s-%s-%04d-comp.jpg' % (
                    expdir.fn_safe(os.path.basename(directory.rstrip('/'))),
                    expdir.fn_safe(layer),
                    unit)
            imsave(od.filename('html/' + imfn),
                    lv.unit_visualization(unit, tight=True))
            html.append('<div class="unit">')
            html.append('<img src="%s" height="%d">' % (imfn, imscale))
            html.append('<div class="unitnum">%d</div>'
                    % record['unit'])
            html.append('<div class="unitlabel">%s</div>'
                    % fix(record['label']))
            html.append('<div class="iou">%.2f</div>'
                    % float(record['score']))
            html.append('</div>')
        html.append('<div class="layername">%s %s</div>' %
                (fix(directory), fix(layer)))
        html.append('</div>')
        barfn = 'image/%s-%s-bargraph.svg' % (
                expdir.fn_safe(os.path.basename(directory.rstrip('/'))),
                expdir.fn_safe(layer))
        bargraph.bar_graph_svg(ed, layer, barheight=imscale,
                barwidth=barwidth, threshold=threshold,
                save=od.filename('html/' + barfn))
        html.append('<div class="layerhist">')
        html.append('<img src="%s">' % barfn)
        html.append('</div>')
        html.append('</div>')
        
    html.append(html_suffix);
    with open(htmlfn, 'w') as f:
        f.write('\n'.join(html))

replacements = [(re.compile(r[0]), r[1]) for r in [
    (r'-[sc]$', ''),
    (r'_', ' '),
    ]]

def fix(s):
    for pattern, subst in replacements:
        s = re.sub(pattern, subst, s)
    return s

html_prefix = '''\
<!doctype html>
<html>
<head>
<style>
body {
  font-family: Arial;
  font-size: 15px;
}
.unit {
  position: relative;
  display: inline-block;
  margin-bottom: 3px;
  margin-right: 5px;
}
.layergrid, .layerhist {
  display: inline-block;
  white-space: nowrap;
  vertical-align: top;
}
.layername {
  display: none;
}
.layerhist {
  margin-bottom: -80px;
}
.unitlabel {
  font-weight: bold;
  font-size: 150%;
  text-align: center;
  line-height: 1;
  width: 144px;
}
.unitnum, .iou {
  color: white;
  position: absolute;
  left: 3px;
  text-shadow: 0px 0px 5px black;
}
.unitnum {
  top: 0;
}
.unitnum::before {
  content: 'unit '
}
.iou {
  top: 125px;
}
.iou::before {
  content: 'IoU ';
}
.layer {
  white-space: nowrap;
}
</style>
</head>
<body class="unitviz">
<div class="container-fluid">
'''

html_suffix = '''
</div>
</body>
</html>
'''

replacements = [(re.compile(r[0]), r[1]) for r in [
    (r'-[sc]$', ''),
    (r'_', ' '),
    (r'/output', '')
    ]]

def fix(s):
    for pattern, subst in replacements:
        s = re.sub(pattern, subst, s)
    return s


if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser(
            description='Generate visualization for probed activation data.')
        parser.add_argument(
                '--layer',
                nargs=2, metavar=('directory', 'blob'), action='append',
                help='directory/blob pairs evaluate')
        parser.add_argument(
                '--concepts',
                nargs='+',
                help='concepts to seek')
        parser.add_argument(
                '--categories',
                nargs='+',
                help='categories to seek')
        parser.add_argument(
                '--top_n', type=int, default=5,
                help='number of units to show')
        parser.add_argument(
                '--imcount', type=int, default=1,
                help='image samples per unit')
        parser.add_argument(
                '--outdir', default=None,
                help='set to dirname to create an html page')
        parser.add_argument(
                '--replace', default=[],
                nargs=2, metavar=('pattern', 'replacement'), action='append',
                help='string replacements for creating webpage')
        parser.add_argument(
                '--css',
                help='filename for css styles')
        parser.add_argument(
                '--imsize',
                type=int, default=224,
                help='thumbnail dimensions')
        parser.add_argument(
                '--imscale',
                type=int, default=144,
                help='thumbnail dimensions')
        parser.add_argument(
                '--barscale',
                type=int, default=800,
                help='thumbnail dimensions')
        args = parser.parse_args()
        od = expdir.ExperimentDirectory(args.outdir)
        generate_study(od, args.layer,
                concepts=args.concepts,
                categories=args.categories,
                top_n=args.top_n,
                imsize=args.imsize, imscale=args.imscale,
                barscale=args.barscale,
                imcount=args.imcount,
                include_hist=True,
                verbose=True)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
