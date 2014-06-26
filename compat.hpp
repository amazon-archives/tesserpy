/* Compatibility changes for compiling against Python 3.x */

#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#define Py_TPFLAGS_HAVE_ITER 0
#define PyInt_FromLong PyLong_FromLong
#define PyString_FromString PyUnicode_FromString
#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION >= 3
#define PyString_AsString(STR) PyUnicode_AsUTF8((STR))
#else
/* XXX reference leak below */
#define PyString_AsString(STR) PyBytes_AsString(PyUnicode_AsUTF8String((STR)))
#endif
#endif
