"""Generate synthetic training data."""

import logging
import random
from collections import OrderedDict

from tools.frames.rectangle import Rectangle
from tools.files.file_utils import FileUtils

class FrameEmbedderError(Exception):
    pass

class FrameEmbedder(object):
    def __init__(self, embed_to_image):
        self.embed_to_image = embed_to_image
        self.width = self.embed_to_image.width
        self.height = self.embed_to_image.height
        self.edgePad = 50 # px
        self.minWH = 50 # px
        self.existingRects = []

    def can_add_det(self, det_image):
        """Attempt to add det and if possible to add, return rect where det can
        be added. If no det can be added, return None"""
        rectToAdd = None
        detImgRect = Rectangle.rectangle_from_dimensions(det_image.width, det_image.height)
        detImgRectRatio = (1.0 * detImgRect.width) / detImgRect.height

        numTries = 0
        while numTries < 10000:
            numTries += 1
            # choose a random sized box within the image
            x0 = random.randint(self.edgePad, self.width - (self.edgePad + self.minWH))
            y0 = random.randint(self.edgePad, self.height - (self.edgePad + self.minWH))
            if detImgRectRatio > 1.0:
                minX2 = int(x0 + detImgRectRatio * self.minWH)
                maxX2 = self.width - self.edgePad
                if minX2 >= maxX2:
                    continue
                x2 = random.randint(minX2, maxX2)
                y2 = int(y0 + (x2 - x0) / detImgRectRatio)
            else:
                minY2 = int(y0 + detImgRectRatio * self.minWH)
                maxY2 = self.height - self.edgePad
                if minY2 >= maxY2:
                    continue
                y2 = random.randint(minY2, maxY2)
                x2 = int(x0 + (y2 - y0) * detImgRectRatio)
            rect = Rectangle.rectangle_from_endpoints(x0, y0, x2, y2)
            rect = rect.get_randomly_perturbed_rectangle()
            rectEncl = rect.get_enclosing_rectangle()
            # if less than minWH or out of bounds, then invalid
            if ((rectEncl.width < self.minWH) or (rectEncl.height < self.minWH) or
                (rectEncl.x2 > (self.width - self.edgePad)) or
                (rectEncl.y2 > (self.height - self.edgePad))):
                continue
            # if intersects with existing rects, then invalid
            intersectsWithExisting = False
            for existingRect in self.existingRects:
                if existingRect.intersects(rectEncl):
                    intersectsWithExisting = True
                    break
            if intersectsWithExisting:
                continue
            # this is an acceptable candidate
            rectToAdd = rect
            break
        return rectToAdd

    def add_det(self, det_image, background_image, anno_rect, texture_image):
        """Add det at given rect to frame"""
        self.existingRects.append(anno_rect.get_enclosing_rectangle())
        self.embed_to_image.embedImage(det_image, background_image, anno_rect, texture_image)

    def calculate_det_area_fraction(self):
        """Calculate the fraction of image occupied by dets"""
        detFrac = 0.0
        for existingRect in self.existingRects:
            detFrac += (existingRect.width * existingRect.height)
        detFrac /= (self.width * self.height)
        return detFrac
