"""Extract frames from clip."""

import logging
import os
import glob
import subprocess
import shutil
from fractions import gcd

from tools.files.file_utils import FileUtils

class FrameExtractorError(Exception):
    pass

class FrameExtractor(object):
    def __init__(self, clip_path, output_path):
        self.clip_path = clip_path
        if not os.path.exists(self.clip_path):
            raise FrameExtractorError("Input clip couldn't be found at {}".format(self.clip_path))
        self.tempFramesPath = os.path.join(output_path, 'temp_frames')
        FileUtils.mkdir_p(self.tempFramesPath)
        self.framesPath = os.path.join(output_path, 'frames')
        FileUtils.mkdir_p(self.framesPath)
        self.frameNumbersFile = os.path.join(output_path, 'frame_numbers.txt')

    def extract_non_sequential(self, frame_numbers):
        # extract every frame
        self.fps = 1
        self._frame_extract()
        return self._rename_frames(sorted(frame_numbers))

    def extract_sequential(self, fps):
        self.fps = fps
        self._frame_extract()
        return self._rename_frames([])

    def _frame_extract(self):
        logging.debug("Extracting at {} FPS".format(self.fps))
        # ffmpeg -i long_video.mp4 -vf select='not(mod(n\,5))' \
        #  -f image2 -qscale 0 -vsync 0 frames/%04d.png
        subprocess.check_call([
            "ffmpeg", "-i", self.clip_path,
            "-vf", "select='not(mod(n\,{}))'".format(self.fps),
            "-f", "image2",
            "-q:v", "0",
            "-vsync", "0",
            "{}/%04d.png".format(self.tempFramesPath)
        ])

    def _rename_frames(self, frame_numbers):
        # format:
        # [[frameNumber, framePath], ]
        frameSorter = []
        for framePath in glob.glob("{}/*.png".format(self.tempFramesPath)):
            frameNumber = int(os.path.splitext(os.path.basename(framePath))[0])
            frameNumber -= 1 # make index 0 based since FFMPEG produces 1 as first file
            frameSorter.append([frameNumber, framePath])
        frameSorter.sort(key = lambda x: x[0])
        # once frames are sorted, move frames to correct location
        writtenFrameNumbers = []
        if len(frame_numbers) == 0:
            # sequential FPS based extraction
            curFrameNumber = 0
            for fr in frameSorter:
                fileName = os.path.join(self.framesPath, "{}.png".format(curFrameNumber))
                shutil.move(fr[1], fileName)
                writtenFrameNumbers.append(curFrameNumber)
                curFrameNumber += self.fps
        else:
            # non-sequential array based extraction
            fnIdx = 0
            for idx, fr in enumerate(frameSorter):
                if frame_numbers[fnIdx] == fr[0]:
                    fileName = os.path.join(self.framesPath, "{}.png".format(frame_numbers[fnIdx]))
                    shutil.move(fr[1], fileName)
                    writtenFrameNumbers.append(frame_numbers[fnIdx])
                    fnIdx += 1
                if fnIdx >= len(frame_numbers):
                    break
        # save frame numbers to file
        with open(self.frameNumbersFile, 'w') as f:
            for wfn in writtenFrameNumbers:
                f.write("%d " % wfn)
            f.write("\n")
        # clean up
        # FileUtils.rm_rf(self.tempFramesPath)
        return writtenFrameNumbers
