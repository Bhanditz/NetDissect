import os
import numpy
import csv
import colorname
from loadseg import AbstractSegmentation
from skimage.io import imread

# Files in the MINC-S segmentation test set:
# DIR/minc/minc-s/categories.txt
# DIR/minc/minc-s/test-segments.txt
# DIR/minc/minc-s/segments/[photoid]_[segid].png
# DIR/photo_orig/[0-9]/[photoid].jpg

class MincSegmentation(AbstractSegmentation):
    def __init__(self, directory=None, supply=None):
        directory = os.path.expanduser(directory)
        self.directory = directory
        self.supply = supply
        # Process open surfaces labels: open categories.txt
        material_name_map = {}
        with open(os.path.join(directory, 'minc', 'minc-s', 'categories.txt')
                ) as f:
            for z, line in enumerate(f.readlines()):
                material_name_map[line.strip()] = z + 1
        self.material_names = ['-'] * (1 + max(material_name_map.values()))
        for k, v in material_name_map.items():
            # Treat the catch-all label as no-label.
            if k == 'other':
                k = '-'
            self.material_names[v] = k
        # Process segment information: open test-segments.txt
        photo_segments = {}
        with open(os.path.join(directory, 'minc', 'minc-s',
                'test-segments.txt')) as f:
            for ms, ph, ss in csv.reader(f):
                material = int(ms)
                photo = int(ph)
                segment = int(ss)
                if photo not in photo_segments:
                    photo_segments[photo] = []
                photo_segments[photo].append((material, segment))
        # Now collate all the segments together in a canonical order
        self.images = [{
            'photo': 'photo_orig/%d/%09d.jpg' % (photo % 10, photo),
            'segments': [
                (material,
                    'minc/minc-s/segments/%09d_%09d.png' % (photo, segment))
                for material, segment in sorted(photo_segments[photo])]
            } for photo in sorted(photo_segments.keys())]

    def all_names(self, category, j):
        if j == 0:
            return []
        if category == 'color':
            return [colorname.color_names[j - 1] + '-c']
        if category == 'material':
            return [self.material_names[j]]
        return []

    def size(self):
        '''Returns the number of images in this dataset.'''
        return len(self.images)

    def filename(self, i):
        '''Returns the filename for the nth dataset image.'''
        return os.path.join(
                self.directory, self.images[i]['photo'])

    def metadata(self, i):
        '''Returns an object that can be used to create all segmentations.'''
        row = self.images[i]
        return (self.directory, row, self.supply)

    @classmethod
    def resolve_segmentation(cls, m, categories=None):
        directory, row, supply = m
        fnjpg = os.path.join(directory, row['photo'])
        result = {}
        if wants('material', categories) and wants('material', supply):
            mats = {}
            for material, fn in row['segments']:
                mask = imread(os.path.join(directory, fn)) != 0
                if material not in mats:
                    mats[material] = mask
                else:
                    mats[material] |= mask
            allmats = numpy.concatenate([
                material * fn[None,:,:]
                for material, fn in sorted(mats.items(), key=lambda x: x[0])])
            result['material'] = allmats
        if wants('color', categories) and wants('color', supply):
            result['color'] = colorname.label_major_colors(imread(fnjpg)) + 1
        arrs = [a for a in result.values() if numpy.shape(a) >= 2]
        shape = arrs[0].shape[-2:] if arrs else (1, 1)
        return result, shape

def wants(what, option):
    if option is None:
        return True
    return what in option
