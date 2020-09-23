#include <Python.h>
#include <stdio.h>

// https://docs.scipy.org/doc/numpy/reference/c-api/types-and-structures.html
typedef struct {
    PyObject_HEAD
    uint8_t *data;
    int nd;
    int64_t *dimensions;
    int64_t *strides;
} PyArrayObject;

#define min(a, b) ((a) < (b))?(a):(b)
#define max(a, b) ((a) > (b))?(a):(b)

static void draw(const PyArrayObject *oframe, const PyArrayObject *iframe)
{
    int64_t iw = iframe->dimensions[1], ih = iframe->dimensions[0];
    int64_t ow = oframe->dimensions[1], oh = oframe->dimensions[0];

    int64_t x_pad = (ow - iw) / 2;
    int64_t y_pad = oh / 2 - ih;

    int64_t is0 = iframe->strides[0], is1 = iframe->strides[1];
    int64_t os0 = oframe->strides[0], os1 = oframe->strides[1];
    uint8_t *data = oframe->data + os0 * (oh - ih);

    for(int y=0; y<ih; y++)
    {
        int64_t i_pad = max(iw / 2 - y_pad - y, 0);
        int64_t o_pad = i_pad + x_pad;
        int64_t n = iw - 2 * i_pad;
        for(int x=0; x<n; x++)
        {
            data[y * os0 + (x + o_pad) * os1 + 0] = iframe->data[y * is0 + (x + i_pad) * is1 + 0];
            data[y * os0 + (x + o_pad) * os1 + 1] = iframe->data[y * is0 + (x + i_pad) * is1 + 1];
            data[y * os0 + (x + o_pad) * os1 + 2] = iframe->data[y * is0 + (x + i_pad) * is1 + 2];
        }
    }
}

static void rot90(PyArrayObject *m)
{
    int64_t h = m->dimensions[0], w = m->dimensions[1];
    int64_t s0 = m->strides[0], s1 = m->strides[1];
    m->dimensions[0] = w;
    m->dimensions[1] = h;
    m->data += s1 * (w - 1);
    m->strides[0] = -s1;
    m->strides[1] = s0;
}

// https://python3-cookbook.readthedocs.io/zh_CN/latest/chapters/p15_c_extensions.html
static PyObject* py_draw3d(PyObject *self, PyObject *args)
{
    PyArrayObject *oframe, *w, *s, *a, *d;
    if (!PyArg_ParseTuple(args, "OOOOO", &oframe, &w, &s, &a, &d))
        return NULL;
    // 下
    draw(oframe, w);
    // 左
    rot90(oframe);
    draw(oframe, a);
    // 上
    rot90(oframe);
    draw(oframe, s);
    // 右
    rot90(oframe);
    draw(oframe, d);
    // 复原
    rot90(oframe);
    Py_RETURN_NONE;
}

/* Module method table */
static PyMethodDef SampleMethods[] = {
  {"draw3d", py_draw3d, METH_VARARGS, "draw3d"},
  { NULL, NULL, 0, NULL}
};

/* Module structure */
static struct PyModuleDef samplemodule = {
  PyModuleDef_HEAD_INIT,
  "sample",           /* name of module */
  "A sample module",  /* Doc string (may be NULL) */
  -1,                 /* Size of per-interpreter state or -1 */
  SampleMethods       /* Method table */
};

/* Module initialization function */
PyMODINIT_FUNC
PyInit_draw3d(void) {
    return PyModule_Create(&samplemodule);
}
