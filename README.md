tesserpy
========

A Python API for Tesseract

Requirements
------------
* Python >= 2.7 or >= 3.2
* NumPy >= 1.6
* Tesseract >= 3.02

Building
--------
It's the usual distutils dance -- run `python setup.py` for more details.

If your Tesseract installation's files are not in the standard system paths, you may need to create a `setup.cfg` with the following contents:

```ini
[build_ext]
include-dirs=/path/to/tesseract/include
library-dirs=/path/to/tesseract/lib
```

Example
-------
Here's a simple example that requires OpenCV:

```python
import cv2
import tesserpy

tess = tesserpy.Tesseract("/path/to/tessdata/prefix", language="eng")
# Anything exposed by SetVariable / GetVariableAsString is an attribute
tess.tessedit_char_whitelist = """'"!@#$%^&*()_+-=[]{};,.<>/?`~abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"""
image = cv2.imread('/path/to/image.png')
tess.set_image(image)
page_info = tess.orientation()
print(page_info.textline_order == tesserpy.TEXTLINE_ORDER_TOP_TO_BOTTOM)
print("#####")
print(tess.get_utf8_text())
print("#####")
print("Word\tConfidence\tBounding box coordinates")
for word in tess.words():
	bb = word.bounding_box
	print("{}\t{}\tt:{}; l:{}; r:{}; b:{}".format(word.text, word.confidence, bb.top, bb.left, bb.right, bb.bottom))
```
