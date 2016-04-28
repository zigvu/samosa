"""Generate synthetic training data."""

import logging
import os
import glob
import random
import json
from collections import OrderedDict

from tools.frames.rectangle import Rectangle
from tools.frames.image_manipulator import ImageManipulator
from tools.synthetic.frame_embedder import FrameEmbedder
from tools.files.file_utils import FileUtils

class DataGeneratorError(Exception):
    pass

class DataGenerator(object):
    def __init__(self, texture_path, embed_to_path, det_path, output_path):
        self.textures = self.read_images(texture_path)
        self.embedToFileNames = glob.glob(os.path.join(embed_to_path,"*.jpg"))
        self.dets = OrderedDict()
        for path in glob.glob(os.path.join(det_path,"*")):
            for pth, img in self.read_images(path).iteritems():
                self.dets[pth] = img
        self.output_path = output_path
        self.clipsPath = os.path.join(self.output_path, 'clips')
        FileUtils.mkdir_p(self.clipsPath)
        self.annotationsPath = os.path.join(self.output_path, 'annotations')
        FileUtils.mkdir_p(self.annotationsPath)
        self.maxNumDetsToEmbed = 10
        self.maxDetAreaFraction = 0.3
        self.numOfFramesInClip = 1025

    def read_images(self, image_path):
        # format:
        # {image_path: ImageManipulator}
        images = OrderedDict()
        for path in glob.glob(os.path.join(image_path,"*.jpg")):
            images[path] = ImageManipulator(path)
        if len(images.keys()) == 0:
            raise DataGeneratorError("No image in: {}".format(image_path))
        return images

    def create_clip(self, clip_id):
        annotations = OrderedDict()
        tempFramesPath = os.path.join(self.output_path, 'temp_frames')
        FileUtils.rm_rf(tempFramesPath)
        FileUtils.mkdir_p(tempFramesPath)
        frameCounter = 0
        # to have annotated frames in mod 5, insert blank first few frames
        for x in xrange(3):
            outFileName = os.path.join(tempFramesPath, '{}.jpg'.format(frameCounter))
            FileUtils.symlink(self.embedToFileNames[3], outFileName)
            frameCounter += 1
        while frameCounter < self.numOfFramesInClip:
            # two frames before annotated frame
            embedToKey = self.embedToFileNames[frameCounter % len(self.embedToFileNames)]
            for x in xrange(2):
                outFileName = os.path.join(tempFramesPath, '{}.jpg'.format(frameCounter))
                FileUtils.symlink(embedToKey, outFileName)
                frameCounter += 1
            # embedded frame
            outFileName = os.path.join(tempFramesPath, '{}.jpg'.format(frameCounter))
            embedToImage = ImageManipulator(embedToKey)
            createdFrame = self.create_frame(embedToImage)
            createdFrame['frame_embedder'].embed_to_image.saveImage(outFileName)
            embedDetails = OrderedDict()
            embedDetails['embed_to'] = embedToKey
            embedDetails['embedded_dets'] = createdFrame['embedded_dets']
            annotations[frameCounter] = embedDetails
            frameCounter += 1
            # two frames after annotated frame
            for x in xrange(2):
                outFileName = os.path.join(tempFramesPath, '{}.jpg'.format(frameCounter))
                FileUtils.symlink(embedToKey, outFileName)
                frameCounter += 1
        os.system("ffmpeg -i {}/%d.jpg -c:v libx264 -r 25 -pix_fmt yuv420p {}.mp4".format(
            tempFramesPath, os.path.join(self.clipsPath, str(clip_id))
        ))
        with open(os.path.join(self.annotationsPath, "{}.json".format(clip_id)), "w") as f:
            json.dump(annotations, f, indent = 4)
        # delete temp files
        FileUtils.rm_rf(tempFramesPath)

    def create_frame(self, embed_to_image):
        embeddedDets = OrderedDict()
        frameEmbedder = FrameEmbedder(embed_to_image)
        while (len(embeddedDets.keys()) < self.maxNumDetsToEmbed and
                frameEmbedder.calculate_det_area_fraction() < self.maxDetAreaFraction):
            detKey = None
            anno_rect = None
            detKeys = self.dets.keys()
            random.shuffle(detKeys)
            for dk in detKeys:
                detKey = dk
                anno_rect = frameEmbedder.can_add_det(self.dets[detKey])
                # if found valid annotation, break
                if anno_rect != None:
                    break
            # if no valid annotation, break
            if anno_rect == None:
                break
            # embed det
            backgroundKey = random.sample(self.textures.keys(), 1)[0]
            background_image = self.textures[backgroundKey]
            textureKey = random.sample(self.textures.keys(), 1)[0]
            texture_image = self.textures[textureKey]
            # print ("Adding {} with background {} and texture {}".format(
            #     detKey, backgroundKey, textureKey
            # ))
            frameEmbedder.add_det(self.dets[detKey], background_image, anno_rect, texture_image)
            if not detKey in embeddedDets:
                embeddedDets[detKey] = []
            embedDetails = OrderedDict()
            embedDetails['anno_rect'] = anno_rect.bbox_format()
            embedDetails['background'] = backgroundKey
            embedDetails['texture'] = textureKey
            embeddedDets[detKey].append(embedDetails)
        # done adding all dets
        frameInfo = OrderedDict()
        frameInfo['frame_embedder'] = frameEmbedder
        frameInfo['embedded_dets'] = embeddedDets
        return frameInfo
