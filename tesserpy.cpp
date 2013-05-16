/* Copyright 2013 The Blindsight Corporation */

#include <Python.h>
#include <leptonica/allheaders.h>
#include <tesseract/baseapi.h>
#include <numpy/arrayobject.h>

typedef struct {
	PyObject_HEAD
	tesseract::TessBaseAPI *tess;
} PyTesseract;

extern "C" {
	static PyTesseract* PyTesseract_new(PyObject *type, PyObject *args, PyObject *kw);
	static void PyTesseract_dealloc(PyTesseract *self);
}

static PyMethodDef PyTesseract_methods[] = {
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
	0, // tp_getattro
	0, // tp_setattro
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
	0, // tp_init
	0, // tp_alloc
	(newfunc)PyTesseract_new, // tp_new
	0, // tp_free
	0, // tp_is_gc
};

#define PyTesseract_Check(v) (Py_TYPE(v) == &PyTesseract_Type)

static PyTesseract* PyTesseract_new(PyObject *type, PyObject *args, PyObject* /* kw */) {
	char *tessdata_prefix = NULL;
	if (!PyArg_ParseTuple(args, "s;Constructor requires the TESSDATA_PREFIX path", &tessdata_prefix)) {
		return NULL;
	}
	setenv("TESSDATA_PREFIX", tessdata_prefix, 1);
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
