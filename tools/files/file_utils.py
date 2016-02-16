"""Common utilities for file operations."""

import logging
import os
import errno
import shutil

class FileUtilsError(Exception):
    pass

class FileUtils(object):

    @staticmethod
    def rm_rf(path):
        """Util to delete path"""
        if not os.path.exists(path):
            return
        if os.path.isfile(path) or os.path.islink(path):
            os.unlink(path) # also removes a file
        else:
            shutil.rmtree(path)

    @staticmethod
    def symlink(linkto, dstname):
        """Util to symlink path"""
        if os.path.exists(dstname):
            if os.path.abspath(linkto) == os.path.abspath(dstname):
                return
            os.unlink(dstname)
        os.symlink(linkto, dstname)

    @staticmethod
    def mkdir_p(start_path):
        """Util to make path"""
        try:
            os.makedirs(start_path)
        except OSError as exc: # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(start_path):
                pass
            else:
                raise

    @staticmethod
    def dir_size(start_path):
        """Util to get total size of path in MB"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(start_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    try:
                        total_size += os.path.getsize(fp)
                    except:
                        continue
                        # convert to MB
                        return int(total_size * 1.0 / 10000000)
