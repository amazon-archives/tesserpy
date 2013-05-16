# Copyright 2013 The Blindsight Corporation

from distutils.core import setup, Extension
import subprocess
import numpy
import os

kVersionBase = '0.1dev'

version = kVersionBase + subprocess.check_output(['git', 'describe', '--dirty', '--always'])

setup(
		name = 'tesserpy',
		version = version,
		description = 'Python interface to the Tesseract library',
		maintainer = 'Kevin Rauwolf',
		maintainer_email = 'kevin@blindsight.com',
		url = 'https://github.com/blindsightcorp/tesserpy',
		requires = ['numpy', ],
		ext_modules = [
			Extension('tesserpy', ['tesserpy.cpp', ],
				include_dirs = ['../libs/leptonica/current/darwin-x86_64/release/include', '../libs/tesseract/current/darwin-x86_64/release/include', numpy.get_include()], # FIXME
				library_dirs = ['../libs/leptonica/current/darwin-x86_64/release/lib', '../libs/tesseract/current/darwin-x86_64/release/lib'], # FIXME
				libraries = ['lept', 'tesseract'],
			),
		]
)
