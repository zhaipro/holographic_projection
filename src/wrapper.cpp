#include <Python.h>
#include <stdio.h>


typedef long long int64_t;
typedef unsigned int uint32_t;


// https://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html
typedef struct PyArrayObject {
    PyObject_HEAD
    unsigned char *data;
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
    int64_t y_pad = oh - ih;

    unsigned char *data = oframe->data + oframe->strides[0] * y_pad;
    int64_t os0 = oframe->strides[0], os1 = oframe->strides[1];
    for(int y=0; y<ih; y++)
    {
        int64_t i_pad = max(iw / 2 - y_pad - y, 0);
        int64_t o_pad = i_pad + x_pad;
        int64_t n = iw - 2 * i_pad;
        for(int x=0; x<n; x++)
        {
            data[y * os0 + (x + o_pad) * os1 + 0] = iframe->data[3 * (y * iw + x + i_pad) + 0];
            data[y * os0 + (x + o_pad) * os1 + 1] = iframe->data[3 * (y * iw + x + i_pad) + 1];
            data[y * os0 + (x + o_pad) * os1 + 2] = iframe->data[3 * (y * iw + x + i_pad) + 2];
        }
    }
}


// https://python3-cookbook.readthedocs.io/zh_CN/latest/chapters/p15_c_extensions.html
static PyObject* py_draw3d(PyObject *self, PyObject *args)
{
    PyArrayObject *iframe, *oframe;
    if (!PyArg_ParseTuple(args, "OO", &iframe, &oframe))
        return NULL;
    // 备份
    int64_t ow = oframe->dimensions[1], oh = oframe->dimensions[0];
    unsigned char *idata = iframe->data, *odata = oframe->data;
    int64_t os0 = oframe->strides[0], os1 = oframe->strides[1];
    // 下
    oframe->data = odata + oh / 2 * os0;
    oframe->dimensions[0] = oh / 2;
    draw(oframe, iframe);
    // 上
    oframe->data = odata + oh / 2 * os0 - 3;
    oframe->strides[0] = -os0;
    oframe->strides[1] = -os1;
    draw(oframe, iframe);
    // 左
    oframe->data = odata + os0 / 2 - 3;
    oframe->strides[0] = -os1;
    oframe->strides[1] = os0;
    draw(oframe, iframe);
    // 右
    oframe->data = odata + os0 * ow - os0 / 2;
    oframe->strides[0] = os1;
    oframe->strides[1] = -os0;
    draw(oframe, iframe);
    // 复原
    oframe->data = odata;
    oframe->dimensions[0] = oh;
    oframe->strides[0] = os0;
    oframe->strides[1] = os1;
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
