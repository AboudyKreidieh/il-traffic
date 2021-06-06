#!/usr/bin/env python3
# flake8: noqa
"""Setup script for the repository."""
from os.path import dirname
from os.path import realpath
from setuptools import find_packages
from setuptools import setup
from setuptools import Distribution
import setuptools.command.build_ext as _build_ext
import subprocess
from il_traffic.version import __version__


def _read_requirements_file():
    req_file_path = '%s/requirements.txt' % dirname(realpath(__file__))
    with open(req_file_path) as f:
        return [line.strip() for line in f]


class BuildExt(_build_ext.build_ext):
    """External build commands."""

    def run(self):
        """Install traci wheels."""
        subprocess.check_call(['pip', 'install', 'ray[tune]'])


class BinaryDistribution(Distribution):
    """See parent class."""

    @staticmethod
    def has_ext_modules():
        """Return True for external modules."""
        return True


setup(
    name='il-traffic',
    version=__version__,
    distclass=BinaryDistribution,
    cmdclass={"build_ext": BuildExt},
    packages=find_packages(),
    install_requires=_read_requirements_file(),
    description='Learning energy-efficient driving behaviors by imitating '
                'experts',
    author='Aboudy Kreidieh',
    url='https://github.com/AboudyKreidieh/il-traffic',
    author_email='aboudy@berkeley.edu',
    zip_safe=False,
)
