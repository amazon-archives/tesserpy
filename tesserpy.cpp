/* Copyright 2013 The Blindsight Corporation */

#include <Python.h>
#include <leptonica/allheaders.h>
#include <tesseract/baseapi.h>
#include <numpy/arrayobject.h>

typedef struct {
	PyObject_HEAD
	tesseract::TessBaseAPI *tess;
} PyTesseract;

// TODO: module-level constants for PSM, OEM, etc.

extern "C" {
	static PyTesseract* PyTesseract_new(PyTypeObject *type, PyObject *args, PyObject *kwargs);
	static int PyTesseract_init(PyTesseract *self, PyObject *args, PyObject *kwargs);
	static int PyTesseract_setattr(PyTesseract *self, PyObject *attr, PyObject *value);
	static PyObject* PyTesseract_getattr(PyTesseract *self, PyObject *attr);
	static PyObject* PyTesseract_set_image(PyTesseract *self, PyObject *args);
	static void PyTesseract_dealloc(PyTesseract *self);
}

static PyMethodDef PyTesseract_methods[] = {
	{ "set_image", (PyCFunction)PyTesseract_set_image, METH_VARARGS, PyDoc_STR("set_image(image)\n\nProvides an image for Tesseract to recognize") },
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
	0, // tp_doc
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
	return self;
}

static void PyTesseract_dealloc(PyTesseract *self) {
	delete(self->tess);
	self->tess = NULL;
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
		PyErr_SetString(PyExc_TypeError, "Attribute name is not a string");
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

static PyObject* PyTesseract_set_image(PyTesseract *self, PyObject *args) {
	// TODO: image to PIX
	Py_INCREF(Py_None);
	return Py_None;
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
