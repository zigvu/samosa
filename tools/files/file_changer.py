"""Regex in file."""

import logging
import os
import shutil

class FileChangerError(Exception):
    pass

class FileChanger(object):

    @staticmethod
    def int_array_reader(input_file):
        """Util to get an array of ints from file"""
        if not os.path.exists(input_file):
            raise FileChangerError("Input file not found: {}".format(input_file))
        frameNumbers = []
        with open(input_file, 'r') as f:
            for line in f.readlines():
                for num in line.split():
                    frameNumbers.append(int(num))
        return frameNumbers


    @staticmethod
    def regex(input_file, regex_hash, output_file):
        """Util to search and replace all regexed items"""
        with open(output_file, "wt") as fout:
            with open(input_file, "rt") as fin:
                for line in fin:
                    for search_term, replace_term in regex_hash.iteritems():
                        line = line.replace(search_term, str(replace_term))
                    fout.write(line)
        # done
