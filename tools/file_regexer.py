"""Common utilities for file operations."""

import logging
import os
import shutil

class FileRegexerError(Exception):
    pass

class FileRegexer(object):

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
