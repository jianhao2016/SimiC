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

from setuptools import setup


setup(
        name = 'simiclasso',
        version = '0.1.1',
        author = 'Jianhao Peng',
        author_email = 'jianhao2@illinois.edu',
        description = 'initial package for simiclasso',
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
        zip_safe = False
        )
