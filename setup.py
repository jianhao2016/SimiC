#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This is the setup file required by setuptools to packaging python code.
"""

from setuptools import setup, find_packages


with open('README.md', 'r') as f:
    long_description = f.read()

setup(
        name = 'simiclasso',
        version = '0.0.2',
        author = 'Jianhao Peng',
        author_email = 'jianhao2@illinois.edu',
        long_description = long_description,
        long_description_content_type = 'text/markdown',
        # url='https://github.com/jianhao2016/simicLASSO_git',
        packages = ['simiclasso',],
        package_dir = {'': 'code'},
        # packages = find_packages(),
        classifiers = [
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS independent',
            ],
        python_requires='>=3.6',
        )
