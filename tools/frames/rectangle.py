import random
import cv2
import numpy as np
from shapely.geometry import Polygon
from matplotlib.path import Path
import matplotlib.patches as patches


class Rectangle(Polygon):
    """General purpose rectangle"""

    def __init__(self, polyArray):
        """Initialize class"""
        Polygon.__init__(self, polyArray)
        self.width = int(self.bounds[2] - self.bounds[0])
        self.height = int(self.bounds[3] - self.bounds[1])
        polyCenter = self.centroid
        self.centerX = int(polyCenter.x)
        self.centerY = int(polyCenter.y)
        self.x0 = self.bounds[0]
        self.y0 = self.bounds[1]
        self.x2 = self.bounds[2]
        self.y2 = self.bounds[3]
        b = np.asarray(self.exterior)
        self.angle = 0
        if (b[1][0] - b[0][0]) != 0:
            self.angle = np.rad2deg(np.arctan((b[1][1] - b[0][1])/(b[1][0] - b[0][0])))

    def get_smaller_rectangle(self, pixelPadding):
        """Get a new rectangle with pixelPadding smaller dimension
        Returns new rectangle
        """
        b = np.asarray(self.exterior)
        return Rectangle([
            (b[0][0] + pixelPadding, b[0][1] + pixelPadding),
            (b[1][0] - pixelPadding, b[1][1] + pixelPadding),
            (b[2][0] - pixelPadding, b[2][1] - pixelPadding),
            (b[3][0] + pixelPadding, b[2][1] - pixelPadding)
        ])

    def get_enclosing_rectangle(self):
        """Get enclosing rectangle"""
        return Rectangle.rectangle_from_endpoints(
            self.x0, self.y0, self.x0 + self.width, self.y0 + self.height
        )

    def get_scaled_rectangle(self, scaleFactor):
        """Get a new rectangle scaled according to given scale factor
        Returns new rectangle
        """
        origRect = Rectangle([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])
        scaledRect = Rectangle([
            (0,                       0),
            (int(1000 * scaleFactor), 0),
            (int(1000 * scaleFactor), int(1000 * scaleFactor)),
            (0,                       int(1000 * scaleFactor))
        ])
        return self.get_transformed_rectangle(origRect, scaledRect)

    def get_sheared_rectangle(self, pt1LR, pt1UD, pt2LR, pt2UD, pt3LR, pt3UD, pt4LR, pt4UD):
        """Get a new rectangle sheared according to given shear angle
        Returns new rectangle
        """
        if (pt1LR >= 1) or (pt1UD >= 1) or (pt2LR >= 1) or (pt2UD >= 1) or \
                (pt3LR >= 1) or (pt3UD >= 1) or (pt4LR >= 1) or (pt4UD >= 1):
            raise RuntimeError("Rectangle: Incoorrect shear request")
        origRect = Rectangle([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])
        shearedRect = Rectangle([
            (int(0 + 1000 * pt1LR),    int(0 + 1000 * pt1UD)),
            (int(1000 + 1000 * pt2LR), int(0 + 1000 * pt2UD)),
            (int(1000 + 1000 * pt3LR), int(1000 + 1000 * pt3UD)),
            (int(0 + 1000 * pt4LR),    int(1000 + 1000 * pt4UD))
        ])
        return self.get_transformed_rectangle(origRect, shearedRect)

    def get_randomly_perturbed_rectangle(self):
        """Get a new rectangle that is randomly sheared in one or more vertices"""
        maxDelta = 0.1
        return self.get_sheared_rectangle(
            random.uniform(-1 * maxDelta, maxDelta), random.uniform(-1 * maxDelta, maxDelta),
            random.uniform(-1 * maxDelta, maxDelta), random.uniform(-1 * maxDelta, maxDelta),
            random.uniform(-1 * maxDelta, maxDelta), random.uniform(-1 * maxDelta, maxDelta),
            random.uniform(-1 * maxDelta, maxDelta), random.uniform(-1 * maxDelta, maxDelta)
        )

    def get_changed_aspect_ratio_rectangle(self, targetAspectRect):
        """Get a new rectangle that is transformed to fit the aspect ratio of
        targetAspectRect"""
        extRect = self.get_enclosing_rectangle()
        rectRatio = (1.0 * extRect.width) / extRect.height
        targetRectRatio = (1.0 * targetAspectRect.width) / targetAspectRect.height

        scaledRect = None
        if rectRatio >= targetRectRatio:
            scaledRect = Rectangle.rectangle_from_centers(
                extRect.centerX, extRect.centerY,
                int(extRect.width * targetRectRatio), extRect.height)
        else:
            scaledRect = Rectangle.rectangle_from_centers(
                extRect.centerX, extRect.centerY,
                extRect.width, int(extRect.height / targetRectRatio))
        # transform and warp embed image to fit in container
        return self.get_transformed_rectangle(extRect, scaledRect)

    def get_transformed_rectangle(self, baseOrigAnnoRect, newOrigAnnoRect):
        """Get a new rectangle that effectively transforms this rectangle with the same
        transformation as is required to take baseOrigAnnoRect to newOrigAnnoRect"""
        mat = Rectangle.get_perspective_transform_matrix(
            baseOrigAnnoRect, newOrigAnnoRect)
        return Rectangle.apply_perspective_transform_matrix(self, mat)

    def numpy_format(self):
        """Convert shapely polygon to numpy array"""
        ext = np.asarray(self.exterior)
        return np.array(ext[0:4, :], np.float32)

    def cv2_format(self):
        """Convert shapely polygon to cv2 polygon points"""
        ext = np.asarray(self.exterior)
        b = np.array(ext[0:4, :], np.int32)
        return b.reshape((-1, 1, 2))

    def matplotlib_format(self):
        """Convert shapely polygon to matplotlib patch"""
        b = np.asarray(self.exterior)
        verts = [
            (b[3][0], b[3][1]), (b[0][0], b[0][1]), (b[1][0], b[1][1]),
            (b[2][0], b[2][1]), (b[3][0], b[3][1])
        ]
        codes = [Path.MOVETO, Path.LINETO,
                 Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
        path = Path(verts, codes)
        patch = patches.PathPatch(path, edgecolor='red', lw=1, fill=False)
        return patch

    def json_format(self):
        """Convert shapely polygon to json format"""
        return {
            'x': self.x0,
            'y': self.y0,
            'width': self.width,
            'height': self.height
        }

    def bbox_format(self):
        """Convert shapely polygon to bbox format"""
        b = np.asarray(self.exterior)
        return {
            'x0': int(b[0][0]), 'y0': int(b[0][1]),
            'x1': int(b[1][0]), 'y1': int(b[1][1]),
            'x2': int(b[2][0]), 'y2': int(b[2][1]),
            'x3': int(b[3][0]), 'y3': int(b[3][1])
        }

    @staticmethod
    def rectangle_from_endpoints(x0, y0, x2, y2):
        """Get rectangle based on end points"""
        rect = Rectangle([(x0, y0), (x2, y0), (x2, y2), (x0, y2)])
        return rect

    @staticmethod
    def rectangle_from_points(x0, y0, x1, y1, x2, y2, x3, y3):
        """Get rectangle based on end points"""
        rect = Rectangle([(x0, y0), (x1, y1), (x2, y2), (x3, y3)])
        return rect

    @staticmethod
    def rectangle_from_dimensions(width, height):
        """Get rectangle based on dimension starting at 0,0"""
        rect = Rectangle([(0, 0), (width, 0), (width, height), (0, height)])
        return rect

    @staticmethod
    def rectangle_from_json(bbox):
        """Get rectangle based on bbox json"""
        rect = Rectangle([
            (int(bbox['x']), int(bbox['y'])),
            (int(bbox['x']) + int(bbox['width']), int(bbox['y'])), (
                int(bbox['x']) + int(bbox['width']),
                int(bbox['y']) + int(bbox['height'])
            ), (int(bbox['x']), int(bbox['y']) + int(bbox['height']))
        ])
        return rect

    @staticmethod
    def rectangle_from_bbox(bbox):
        """Get rectangle based on bbox"""
        rect = Rectangle([
            (int(bbox['x0']), int(bbox['y0'])),
            (int(bbox['x1']), int(bbox['y1'])),
            (int(bbox['x2']), int(bbox['y2'])),
            (int(bbox['x3']), int(bbox['y3']))
        ])
        return rect

    @staticmethod
    def rectangle_from_centers(centerX, centerY, width, height, largerRect=None):
        """Get rectangle based on dimension centered on X,Y
        Optionally, specify largerRect if returned rectangle needs to be within
        """
        x0 = x3 = int(centerX - width / 2)
        x1 = x2 = int(centerX + width / 2)
        y0 = y1 = int(centerY - height / 2)
        y2 = y3 = int(centerY + height / 2)

        if x0 < 0:
            x1 = x2 = int(centerX + width / 2) + abs(x0)
            x0 = x3 = 0
        if y0 < 0:
            y2 = y3 = int(centerY + height / 2) + abs(y0)
            y0 = y1 = 0

        if largerRect != None:
            if x1 > largerRect.width:
                x1 = x2 = largerRect.width
            if y2 > largerRect.height:
                y2 = y3 = largerRect.height

        # if desired output is square, return back square
        if width == height:
            minWH = min((x1 - x0), (y2 - y0))
            x1 = x2 = x0 + minWH
            y2 = y3 = y0 + minWH

        rect = Rectangle([(x0, y0), (x1, y1), (x2, y2), (x3, y3)])
        return rect

    @staticmethod
    def apply_perspective_transform_matrix(srcShape, transformMatrix):
        """Apply transformation matrix to srcShape.
        Return new rectangle
        """
        src = np.array([srcShape.numpy_format()])
        dst = cv2.perspectiveTransform(src, transformMatrix)
        numpyBox = dst[0]
        bbox = Rectangle([
            (int(numpyBox[0][0]), int(numpyBox[0][1])),
            (int(numpyBox[1][0]), int(numpyBox[1][1])),
            (int(numpyBox[2][0]), int(numpyBox[2][1])),
            (int(numpyBox[3][0]), int(numpyBox[3][1]))
        ])
        return bbox

    @staticmethod
    def get_perspective_transform_matrix(srcShape, dstShape):
        """Get transformation matrix to take srcShape to dstShape.
        Return transformMatrix
        """
        src = srcShape.numpy_format()
        dst = dstShape.numpy_format()
        return cv2.getPerspectiveTransform(src, dst)

    def __str__(self):
        b = self.bounds
        return "X: " + str(int(b[0])) + \
            ", Y: " + str(int(b[1])) + \
            ", W: " + str(int(b[2] - b[0])) + \
            ", H: " + str(int(b[3] - b[1]))
