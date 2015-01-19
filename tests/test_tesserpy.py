import tesserpy
import pytest
import numpy as np
import os.path

kTessdataPrefix = '/usr/share/tesseract-ocr/'
kTesseractLanguage = 'eng'
kSampleFile = 'sample.npy'
kSampleText = 'This is line one\nNow this is line two\n\n'

@pytest.fixture
def tesseract():
	tesseract = tesserpy.Tesseract(kTessdataPrefix)
	tesseract.tessedit_char_whitelist = """abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"""
	return tesseract

@pytest.fixture(scope='module')
def image():
	return np.load(os.path.join(os.path.dirname(__file__), kSampleFile))

def test_init_empty():
	with pytest.raises(TypeError):
		tesserpy.Tesseract()

def test_init_bad_datapath():
	with pytest.raises(EnvironmentError):
		tesserpy.Tesseract('/xx/yy/zz/aa')

def test_init_good_datapath(tesseract):
	assert tesseract is not None

def test_init_language():
	tesseract = tesserpy.Tesseract(kTessdataPrefix, language=kTesseractLanguage)
	assert tesseract is not None

def test_defaults(tesseract):
	assert int(tesseract.textord_blob_size_bigile) != 85
	assert int(tesseract.classify_debug_level) == 0

def test_set_get_variable(tesseract):
	old_strength = int(tesseract.classify_cp_cutoff_strength)
	new_strength = old_strength + 5
	tesseract.classify_cp_cutoff_strength = new_strength
	assert int(tesseract.classify_cp_cutoff_strength) == new_strength

def test_init_config():
	tesseract = tesserpy.Tesseract(kTessdataPrefix, configs=(os.path.join(os.path.dirname(__file__), 'tesseract.ini'), ))
	assert int(tesseract.textord_blob_size_bigile) == 85
	assert int(tesseract.classify_debug_level) == 2

def test_init_only_non_debug():
	tesseract = tesserpy.Tesseract(kTessdataPrefix, configs=(os.path.join(os.path.dirname(__file__), 'tesseract.ini'), ), set_only_non_debug_params=True)
	assert int(tesseract.classify_debug_level) == 0

def test_set_image(tesseract, image):
	tesseract.set_image(image)

def test_get_page_info(tesseract, image):
	tesseract.set_image(image)
	page_info = tesseract.orientation()
	assert page_info.orientation == tesserpy.ORIENTATION_PAGE_UP
	assert page_info.writing_direction == tesserpy.WRITING_DIRECTION_LEFT_TO_RIGHT
	assert page_info.textline_order == tesserpy.TEXTLINE_ORDER_TOP_TO_BOTTOM

def test_get_utf8_text(tesseract, image):
	tesseract.set_image(image)
	text = tesseract.get_utf8_text()
	assert text == kSampleText

def test_get_words(tesseract, image):
	def words():
		for word in kSampleText.split():
			yield word

	tesseract.set_image(image)
	tesseract.get_utf8_text() # FIXME: shouldn't be required
	word_list = words()
	for word in tesseract.words():
		assert word.text == word_list.next()

def test_get_text_lines(tesseract, image):
	def lines():
		for line in [ll for ll in kSampleText.split('\n') if ll]:
			yield line

	tesseract.set_image(image)
	tesseract.get_utf8_text() # FIXME: shouldn't be required
	line_list = lines()
	for line in tesseract.text_lines():
		assert line.text.strip() == line_list.next()

def test_get_mean_text_conf(tesseract, image):
	tesseract.set_image(image)
	assert tesseract.mean_text_conf() > 80
