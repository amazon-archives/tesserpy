# Copyright 2015 The Blindsight Corporation

from distutils.core import setup, Extension, Command
import numpy

class PyTest(Command):
	user_options = list()
	def initialize_options(self):
		pass

	def finalize_options(self):
		pass

	def run(self):
		import sys, subprocess
		errno = subprocess.call([sys.executable, 'runtests.py'])
		raise SystemExit(errno)

setup(
		name = 'tesserpy',
		version = '1.1.2',
		description = 'Python interface to the Tesseract library',
		long_description = 'Python interface to the Tesseract library',
		maintainer = 'Kevin Rauwolf',
		maintainer_email = 'kevin@blindsight.com',
		url = 'https://github.com/blindsightcorp/tesserpy',
		license = 'LGPLv2',
		requires = ['numpy', ],
		ext_modules = [
			Extension('tesserpy', ['tesserpy.cpp', ],
				libraries = ['tesseract', ],
				include_dirs = [numpy.get_include(), ],
			),
		],
		keywords = ['tesseract', 'ocr', ],
		cmdclass = {'test': PyTest, },
		classifiers = [
			'Development Status :: 5 - Production/Stable',
			'Intended Audience :: Science/Research',
			'License :: OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)',
			'Natural Language :: English',
			'Programming Language :: Python :: 2',
			'Programming Language :: Python :: 3',
			'Topic :: Multimedia :: Graphics :: Graphics Conversion',
			'Topic :: Scientific/Engineering :: Image Recognition',
			'Topic :: Utilities',
		],
)
