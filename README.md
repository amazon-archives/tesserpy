tesserpy
========

A Python API for Tesseract

Building
--------
It's the usual distutils dance -- run `python setup.py` for more details.

If your Tesseract installation's files are not in the standard system paths,
you may need to create a `setup.cfg` with the following contents:

	[build_ext]
	include-dirs=/path/to/tesseract/include
	library-dirs=/path/to/tesseract/lib
