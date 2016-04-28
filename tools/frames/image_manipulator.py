import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

from tools.frames.rectangle import Rectangle

class ImageManipulatorError(Exception):
    pass

class ImageManipulator(object):
    """Manipulate single image using opencv and plt tools"""

    def __init__(self, imageFileName):
        """Initialization"""
        self.image = cv2.imread(imageFileName)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        self.colorMap = cm.ScalarMappable(norm=norm, cmap=cm.jet)
        self.heatmapVis = 0.5
        self.height, self.width, self.channels = self.image.shape

    def addPixelMap(self, pixelMap):
        """Overlay pixelMap as heatmap on top of image"""
        heatmap = self.colorMap.to_rgba(pixelMap, bytes=True)
        # opencv uses BGR but matplotlib users rgb
        r, g, b, a = cv2.split(heatmap)
        heatmap_bgr = cv2.merge([b, g, r])
        self.image = cv2.addWeighted(
            heatmap_bgr, self.heatmapVis, self.image, 1 - self.heatmapVis, 0)

    def addLabeledBbox(self, bbox, label):
        """Overlay bboxes with labels"""
        rect = Rectangle.rectangle_from_endpoints(
            bbox.x0, bbox.y0, bbox.x2, bbox.y2)
        self.addLabeledRect(rect, label)

    def addLabeledRect(self, rect, label):
        """Overlay rects with labels"""
        textColor = (256, 256, 256)
        colorForeground = (0, 0, 256)
        colorBackground = (256, 256, 256)
        pts = rect.cv2_format()
        cv2.polylines(self.image, [pts - 1], True, colorForeground)
        cv2.polylines(self.image, [pts], True, colorBackground)
        cv2.polylines(self.image, [pts + 1], True, colorForeground)
        cv2.putText(
            self.image, label, (pts[0][0][0] + 5, pts[0][0][1] + 20),
            self.font, 0.8, textColor, 2)

    def show(self):
        """Show current image state"""
        # opencv uses BGR but matplotlib users rgb
        b, g, r = cv2.split(self.image)
        rgb_image = cv2.merge([r, g, b])
        plt.imshow(rgb_image)

    def extract_patch(self, bbox, outputPatchName, patchWidth, patchHeight):
        """Extract patch from image at specified bbox location"""
        rect = Rectangle.rectangle_from_endpoints(
            bbox.x0, bbox.y0, bbox.x2, bbox.y2)
        tX0 = rect.x0
        tY0 = rect.y0
        tW = rect.x0 + rect.width
        tH = rect.y0 + rect.height
        patch = self.image[tY0:tH, tX0:tW].copy()
        patch = cv2.resize(patch, (patchWidth, patchHeight))
        cv2.imwrite(outputPatchName, patch)

    def embedFrameNumber(self, frameNumber):
        """Embeds frame number using color coded squares"""
        # Explanation of color<->frame_number can be found in kheer issue
        # https://github.com/zigvu/kheer/issues/10
        colors = self.getEmbedColors(frameNumber)
        squareWH = 10
        sqDem = [
            [[0, 0], [squareWH, squareWH]],
            [[1280 - squareWH, 0], [1280, squareWH]],
            [[0, 720 - squareWH], [squareWH, 720]],
            [[1280 - squareWH, 720 - squareWH], [1280, 720]],
        ]
        for idx, sq in enumerate(sqDem):
            # opencv uses bgr
            bgrColor = (colors[idx][2], colors[idx][1], colors[idx][0])
            cv2.rectangle(
                self.image, (sq[0][0], sq[0][1]), (sq[1][0], sq[1][1]), bgrColor, -1)

    def embedImage(self, embImg, bkgrndImage, containerRect, textureImage = None):
        """Adds texture to an image and embeds it within given Rectangle.
        embdImg: ImageManipulator image to embed. Assume has white background.
        bkgrndImage: ImageManipulator image where to embed.
        containerRect: Rectangle poly describing shape of final embedding.
        textureImage: ImageManipulator image that will be overlaid as texture on embed.
        Note: self, bkgrndImage and textureImage images need to be of same dimension."""
        # check for dimension
        if (self.width != bkgrndImage.width) or (self.height != bkgrndImage.height):
            raise ImageManipulatorError("All images need to be of same dimension")
        if textureImage != None:
            if (self.width != textureImage.width) or (self.height != textureImage.height):
                raise ImageManipulatorError("All images need to be of same dimension")
        # transformation needed for embedding image
        embImgRect = Rectangle.rectangle_from_dimensions(embImg.width, embImg.height)
        trMatForTransImg = Rectangle.get_perspective_transform_matrix(embImgRect, containerRect)
        transImg = np.zeros(shape=(self.width, self.height), dtype="uint8")
        transImg = cv2.warpPerspective(
            embImg.image, trMatForTransImg, (self.width, self.height),
            transImg, cv2.INTER_CUBIC, cv2.BORDER_CONSTANT,
            (255, 255, 255, 255))

        # create the inverse mask of embImg to clear out white pixels
        # of embImg in bkgrndImage
        transImgGray = cv2.cvtColor(transImg, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(transImgGray, 250, 255, cv2.THRESH_BINARY_INV)
        maskInv = cv2.bitwise_not(mask)
        bkgrndImageBg = cv2.bitwise_and(bkgrndImage.image, bkgrndImage.image, mask = maskInv)
        transImgFg = cv2.bitwise_and(transImg, transImg, mask = mask)
        # embed with background and modify the main image
        if textureImage != None:
            textureImageFg = cv2.bitwise_and(textureImage.image, textureImage.image, mask = mask)
            transImgFg = cv2.addWeighted(transImgFg, 0.7, textureImageFg, 0.3, 0)
        dst = cv2.add(bkgrndImageBg, transImgFg)
        cropPoly = containerRect.get_enclosing_rectangle().get_smaller_rectangle(-2).numpy_format()

        self.image[
            int(cropPoly[0][1]):int(cropPoly[2][1]),
            int(cropPoly[0][0]):int(cropPoly[2][0])
        ] = dst[
            int(cropPoly[0][1]):int(cropPoly[2][1]),
            int(cropPoly[0][0]):int(cropPoly[2][0])
        ]

    def getEmbedColors(self, frameNumber):
        """Get color for squares that represent binary value of frameNumber"""
        # Assumes that there are four rectangles to be embedded in image
        fnBin = [int(x) for x in bin(frameNumber)[2:]]
        fnBin = [int(0) for x in range(0, 12 - len(fnBin))] + fnBin
        f = [255 * x for x in fnBin]
        return [(f[0], f[1], f[2]), (f[3], f[4], f[5]), (f[6], f[7], f[8]),
                (f[9], f[10], f[11])]

    def resize_image(self, scale):
        """Resize image to new scale"""
        self.image = cv2.resize(self.image, (0, 0), fx=scale, fy=scale)

    def resize_to_rect(self, rect):
        """Resize image to the bounds of new rect"""
        pass

    def getImage(self):
        """Return current image state"""
        return self.image

    def saveImage(self, outputFileName):
        """Save current image state"""
        cv2.imwrite(outputFileName, self.image)
