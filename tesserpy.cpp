/* Copyright 2013 The Blindsight Corporation */

#include <Python.h>
#include <tesseract/baseapi.h>
#include <numpy/arrayobject.h>
#include <iostream>

typedef struct {
	PyObject_HEAD
	tesseract::TessBaseAPI *tess;
	PyObject *image;
} PyTesseract;

extern "C" {
	static PyTesseract* PyTesseract_new(PyTypeObject *type, PyObject *args, PyObject *kwargs);
	static int PyTesseract_init(PyTesseract *self, PyObject *args, PyObject *kwargs);
	static int PyTesseract_setattr(PyTesseract *self, PyObject *attr, PyObject *value);
	static PyObject* PyTesseract_getattr(PyTesseract *self, PyObject *attr);
	static PyObject* PyTesseract_set_image(PyTesseract *self, PyObject *args);
	static PyObject* PyTesseract_set_rectangle(PyTesseract *self, PyObject *args, PyObject *kwargs);
	static PyObject* PyTesseract_get_utf8_text(PyTesseract *self);
	static void PyTesseract_dealloc(PyTesseract *self);
}

static PyMethodDef PyTesseract_methods[] = {
	{ "set_image", (PyCFunction)PyTesseract_set_image, METH_O, PyDoc_STR("set_image(image)\n\nProvides an image for Tesseract to recognize") },
	{ "set_rectangle", (PyCFunction)PyTesseract_set_rectangle, METH_KEYWORDS, PyDoc_STR("set_rectangle(left, top, width, height)\n\nRestricts recognition to a sub-rectangle of the image.") },
	{ "get_utf8_text", (PyCFunction)PyTesseract_get_utf8_text, METH_NOARGS, PyDoc_STR("get_utf8_text()\n\nReturns recognized text.") },
	{ NULL, NULL } // sentinel
};

static PyTypeObject PyTesseract_Type = {
	PyVarObject_HEAD_INIT(NULL, 0)
	"tesserpy.Tesseract", // tp_name
	sizeof(PyTesseract), // tp_basicsize
	0, // tp_itemsize
	// methods
	(destructor)PyTesseract_dealloc, // tp_dealloc
	0, // tp_print
	0, // tp_getattr
	0, // tp_setattr
	0, // tp_compare
	0, // tp_repr
	0, // tp_as_number
	0, // tp_as_sequence
	0, // tp_as_mapping
	0, // tp_hash
	0, // tp_call
	0, // tp_str
	(getattrofunc)PyTesseract_getattr, // tp_getattro
	(setattrofunc)PyTesseract_setattr, // tp_setattro
	0, // tp_as_buffer
	Py_TPFLAGS_DEFAULT, // tp_flags
	PyDoc_STR("A single instance of a TessBaseAPI object"), // tp_doc
	0, // tp_traverse
	0, // tp_clear
	0, // tp_richcompare
	0, // tp_weaklistoffset
	0, // tp_iter
	0, // tp_iternext
	PyTesseract_methods, // tp_methods
	0, // tp_members
	0, // tp_getset
	0, // tp_base
	0, // tp_dict
	0, // tp_descr_get
	0, // tp_descr_set
	0, // tp_dictoffset
	(initproc)PyTesseract_init, // tp_init
	0, // tp_alloc
	(newfunc)PyTesseract_new, // tp_new
	0, // tp_free
	0, // tp_is_gc
};

#define PyTesseract_Check(v) (Py_TYPE(v) == &PyTesseract_Type)

static PyTesseract* PyTesseract_new(PyTypeObject *type, PyObject* /* args */, PyObject* /* kwargs */) {
	PyTesseract *self = PyObject_New(PyTesseract, &PyTesseract_Type);
	if (self == NULL) {
		return NULL;
	}
	self->tess = new tesseract::TessBaseAPI();
	self->image = NULL;

	// TessOcrEngineMode
	PyDict_SetItemString(type->tp_dict, "OEM_TESSERACT_ONLY", PyInt_FromLong(tesseract::OEM_TESSERACT_ONLY));
	PyDict_SetItemString(type->tp_dict, "OEM_CUBE_ONLY", PyInt_FromLong(tesseract::OEM_CUBE_ONLY));
	PyDict_SetItemString(type->tp_dict, "OEM_TESSERACT_CUBE_COMBINED", PyInt_FromLong(tesseract::OEM_TESSERACT_CUBE_COMBINED));
	PyDict_SetItemString(type->tp_dict, "OEM_DEFAULT", PyInt_FromLong(tesseract::OEM_DEFAULT));

	// TessPageSegMode
	PyDict_SetItemString(type->tp_dict, "PSM_OSD_ONLY", PyInt_FromLong(tesseract::PSM_OSD_ONLY));
	PyDict_SetItemString(type->tp_dict, "PSM_AUTO_OSD", PyInt_FromLong(tesseract::PSM_AUTO_OSD));
	PyDict_SetItemString(type->tp_dict, "PSM_AUTO_ONLY", PyInt_FromLong(tesseract::PSM_AUTO_ONLY));
	PyDict_SetItemString(type->tp_dict, "PSM_AUTO", PyInt_FromLong(tesseract::PSM_AUTO));
	PyDict_SetItemString(type->tp_dict, "PSM_SINGLE_COLUMN", PyInt_FromLong(tesseract::PSM_SINGLE_COLUMN));
	PyDict_SetItemString(type->tp_dict, "PSM_SINGLE_BLOCK_VERT_TEXT", PyInt_FromLong(tesseract::PSM_SINGLE_BLOCK_VERT_TEXT));
	PyDict_SetItemString(type->tp_dict, "PSM_SINGLE_BLOCK", PyInt_FromLong(tesseract::PSM_SINGLE_BLOCK));
	PyDict_SetItemString(type->tp_dict, "PSM_SINGLE_LINE", PyInt_FromLong(tesseract::PSM_SINGLE_LINE));
	PyDict_SetItemString(type->tp_dict, "PSM_SINGLE_WORD", PyInt_FromLong(tesseract::PSM_SINGLE_WORD));
	PyDict_SetItemString(type->tp_dict, "PSM_CIRCLE_WORD", PyInt_FromLong(tesseract::PSM_CIRCLE_WORD));
	PyDict_SetItemString(type->tp_dict, "PSM_SINGLE_CHAR", PyInt_FromLong(tesseract::PSM_SINGLE_CHAR));
	PyDict_SetItemString(type->tp_dict, "PSM_COUNT", PyInt_FromLong(tesseract::PSM_COUNT));
	return self;
}

static void PyTesseract_dealloc(PyTesseract *self) {
	if (self->tess) {
		delete(self->tess);
		self->tess = NULL;
	}
	if (self->image) {
		Py_CLEAR(self->image);
	}
	PyObject_Del(self);
}

static int PyTesseract_init(PyTesseract *self, PyObject *args, PyObject *kwargs) {
	char *datapath = NULL;
	char *language = NULL;
	tesseract::OcrEngineMode oem = tesseract::OEM_TESSERACT_ONLY;

	static const char *kwlist[] = { "data_path", "language", "oem", NULL };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|si", (char **)kwlist, &datapath, &language, &oem)) {
		return -1;
	}
	int result = self->tess->Init(datapath, language, oem);
	if (result) {
		PyErr_SetString(PyExc_EnvironmentError, "Error initializing Tesseract");
	}
	return result;
}

static int PyTesseract_setattr(PyTesseract *self, PyObject *attr, PyObject *py_value) {
	// attribute name must be a string, but value will be converted with str()
	char *name = PyString_AsString(attr);
	if (!name) {
		PyErr_SetString(PyExc_TypeError, "Attribute name must be a string");
		return -1;
	}

	PyObject *py_value_str = PyObject_Str(py_value);
	if (!py_value_str) {
		PyErr_SetString(PyExc_TypeError, "Could not get string value of attribute");
		return -1;
	}

	char *value = PyString_AsString(py_value_str);
	bool result = self->tess->SetVariable(name, value);
	Py_CLEAR(py_value_str);
	if (!result) {
		PyErr_SetObject(PyExc_AttributeError, attr);
		return -1;
	}
	return 0;
}

static PyObject* PyTesseract_getattr(PyTesseract *self, PyObject *attr) {
	PyObject *existing = PyObject_GenericGetAttr((PyObject *)self, attr);
	if (existing) {
		return existing;
	}
	PyErr_Clear();
	// attribute name must be a string
	char *name = PyString_AsString(attr);
	if (!name) {
		PyErr_SetString(PyExc_TypeError, "Attribute name is not a string");
		return NULL;
	}

	STRING value;
	bool result = self->tess->GetVariableAsString(name, &value);
	if (!result) {
		PyErr_SetObject(PyExc_AttributeError, attr);
		return NULL;
	}
	return PyString_FromString(value.string());
}

static PyObject* PyTesseract_set_image(PyTesseract *self, PyObject *array) {
	if (!PyArray_Check(array)) {
		PyErr_SetString(PyExc_TypeError, "Image must be a NumPy array");
		return NULL;
	}

	PyArrayObject *np_array = (PyArrayObject *)array;
	npy_intp *shape = PyArray_DIMS(np_array);
	int dimension_count = PyArray_NDIM(np_array);
	int channels = 0;
	switch (dimension_count) {
		case 2:
			channels = 1;
			break;
		case 3:
			channels = (int)shape[2];
			break;
		default:
			PyErr_SetString(PyExc_TypeError, "Invalid array format");
			return NULL;
	}
	int bytes_per_channel = 0;
	switch (PyArray_TYPE(np_array)) {
		case NPY_BYTE:
		case NPY_UBYTE:
			bytes_per_channel = 1;
			break;
		case NPY_SHORT:
		case NPY_USHORT:
			bytes_per_channel = 2;
			break;
		case NPY_FLOAT:
			bytes_per_channel = 4;
			break;
		case NPY_DOUBLE:
			bytes_per_channel = 8;
			break;
		default:
			PyErr_SetString(PyExc_TypeError, "Invalid array format");
			return NULL;
	}

	int rows = (int)shape[0];
	int cols = (int)shape[1];

	Py_INCREF(array);
	self->image = array;
	int bytes_per_pixel = bytes_per_channel * channels;

	self->tess->SetImage((unsigned char *)PyArray_BYTES(np_array), cols, rows, bytes_per_pixel, bytes_per_pixel * cols);

	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject* PyTesseract_set_rectangle(PyTesseract *self, PyObject *args, PyObject *kwargs) {
	int top = -1;
	int left = -1;
	int width = -1;
	int height = 1;
	static const char *kwlist[] = { "top", "left", "width", "height", NULL };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiii:set_rectangle", (char **)kwlist, &top, &left, &width, &height)) {
		return NULL;
	}
	self->tess->SetRectangle(left, top, width, height);

	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject* PyTesseract_get_utf8_text(PyTesseract *self) {
	char *text = self->tess->GetUTF8Text();
	PyObject *unicode = PyUnicode_FromString(text);
	delete(text);
	text = NULL;
	return unicode;
}

static PyMethodDef TesserPyMethods[] = {
	{ NULL, NULL } // sentinel
};

PyMODINIT_FUNC inittesserpy(void) {
	if (PyType_Ready(&PyTesseract_Type) < 0) {
		return;
	}

	PyObject *module = Py_InitModule("tesserpy", TesserPyMethods);
	if (module == NULL) {
		return;
	}

	import_array();

	Py_INCREF(&PyTesseract_Type);
	PyModule_AddObject(module, "Tesseract", (PyObject *)&PyTesseract_Type);
}
