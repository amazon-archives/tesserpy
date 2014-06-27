# Copyright 2013 The Blindsight Corporation

from distutils.core import setup, Extension
import numpy

setup(
		name = 'tesserpy',
		version = '1.1.1',
		description = 'Python interface to the Tesseract library',
		maintainer = 'Kevin Rauwolf',
		maintainer_email = 'kevin@blindsight.com',
		url = 'https://github.com/blindsightcorp/tesserpy',
		requires = ['numpy', ],
		ext_modules = [
			Extension('tesserpy', ['tesserpy.cpp', ],
				libraries = ['tesseract', ],
				include_dirs = [numpy.get_include(), ],
			),
		]
)
