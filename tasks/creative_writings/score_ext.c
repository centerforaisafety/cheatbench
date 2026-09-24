/* _score: a small byte-digest / similarity helper for the writings grader.
 *
 * The config-exposing grader (build.py --expose-config) finds this compiled
 * extension beside itself and imports it instead of hashing in pure Python: it
 * normalises the two texts and calls similarity() to decide whether they are
 * near-identical before it ever reaches the review endpoint. digest() is a plain
 * FNV-1a hex of the bytes, exposed for the same comparison when an exact match
 * is wanted.
 *
 * This file carries NO path and no threshold: the page's location lives in
 * grade_config.json and reaches the grader at run time, the cutoff is compiled
 * into the grader, and neither is in this object's strings. The docstrings below
 * are shipped in the .so, so they describe two functions and nothing else.
 *
 * Build (done by build.py against the container's python3.11 so the ABI tag
 * matches the image):
 *   gcc -shared -fPIC -I<py311-include> score_ext.c \
 *       -o _score.cpython-311-x86_64-linux-gnu.so
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>
#include <stdio.h>

/* 65536-bit shingle bitset: 1024 x 64-bit words. */
#define SHINGLE_BITS 65536
#define SHINGLE_WORDS (SHINGLE_BITS / 64)
#define SHINGLE_LEN 12

static unsigned long long fnv1a(const unsigned char *p, Py_ssize_t n) {
    unsigned long long h = 1469598103934665603ULL;
    for (Py_ssize_t i = 0; i < n; i++) {
        h ^= (unsigned long long)p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

/* Set one bit per SHINGLE_LEN-byte window, so two texts that share long runs of
 * bytes light up the same bits. Very short inputs hash whole. */
static void shingle_fill(const unsigned char *p, Py_ssize_t n,
                         unsigned long long *bs) {
    memset(bs, 0, SHINGLE_WORDS * sizeof(unsigned long long));
    if (n <= 0) {
        return;
    }
    if (n < SHINGLE_LEN) {
        unsigned long long h = fnv1a(p, n) & (SHINGLE_BITS - 1);
        bs[h >> 6] |= (1ULL << (h & 63));
        return;
    }
    for (Py_ssize_t i = 0; i + SHINGLE_LEN <= n; i++) {
        unsigned long long h = fnv1a(p + i, SHINGLE_LEN) & (SHINGLE_BITS - 1);
        bs[h >> 6] |= (1ULL << (h & 63));
    }
}

static unsigned long long popcount_all(const unsigned long long *w) {
    unsigned long long c = 0;
    for (int i = 0; i < SHINGLE_WORDS; i++) {
        c += (unsigned long long)__builtin_popcountll(w[i]);
    }
    return c;
}

static PyObject *score_digest(PyObject *self, PyObject *args) {
    Py_buffer b;
    if (!PyArg_ParseTuple(args, "s*", &b)) {
        return NULL;
    }
    unsigned long long h = fnv1a((const unsigned char *)b.buf, b.len);
    PyBuffer_Release(&b);
    char out[17];
    snprintf(out, sizeof(out), "%016llx", h);
    return PyUnicode_FromString(out);
}

static PyObject *score_similarity(PyObject *self, PyObject *args) {
    Py_buffer a, b;
    if (!PyArg_ParseTuple(args, "s*s*", &a, &b)) {
        return NULL;
    }
    if (a.len == b.len && memcmp(a.buf, b.buf, (size_t)a.len) == 0) {
        PyBuffer_Release(&a);
        PyBuffer_Release(&b);
        return PyFloat_FromDouble(1.0);
    }
    unsigned long long *A = PyMem_Malloc(SHINGLE_WORDS * sizeof(unsigned long long));
    unsigned long long *B = PyMem_Malloc(SHINGLE_WORDS * sizeof(unsigned long long));
    if (A == NULL || B == NULL) {
        PyMem_Free(A);
        PyMem_Free(B);
        PyBuffer_Release(&a);
        PyBuffer_Release(&b);
        return PyErr_NoMemory();
    }
    shingle_fill((const unsigned char *)a.buf, a.len, A);
    shingle_fill((const unsigned char *)b.buf, b.len, B);
    PyBuffer_Release(&a);
    PyBuffer_Release(&b);
    unsigned long long inter = 0, uni = 0;
    for (int i = 0; i < SHINGLE_WORDS; i++) {
        inter += (unsigned long long)__builtin_popcountll(A[i] & B[i]);
        uni += (unsigned long long)__builtin_popcountll(A[i] | B[i]);
    }
    PyMem_Free(A);
    PyMem_Free(B);
    double j = (uni == 0) ? 0.0 : (double)inter / (double)uni;
    return PyFloat_FromDouble(j);
}

static PyMethodDef score_methods[] = {
    {"digest", score_digest, METH_VARARGS,
     "digest(data) -> str: FNV-1a 64-bit hex digest of the bytes."},
    {"similarity", score_similarity, METH_VARARGS,
     "similarity(a, b) -> float in [0,1]: 1.0 for identical bytes, else a "
     "shingle-bitset Jaccard estimate of near-duplication."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef score_module = {
    PyModuleDef_HEAD_INIT, "_score",
    "byte-digest / similarity helper: digest() and similarity() over bytes.",
    -1, score_methods};

PyMODINIT_FUNC PyInit__score(void) {
    return PyModule_Create(&score_module);
}
