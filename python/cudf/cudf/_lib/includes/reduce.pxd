# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._lib.cudf cimport *


cdef extern from "cudf/reduction.hpp" namespace "cudf::reduction" nogil:

    ctypedef enum operators:
        SUM = 0,
        MIN,
        MAX,
        ANY,
        ALL,
        PRODUCT,
        SUMOFSQUARES,
        MEAN,
        VAR,
        STD,

cdef extern from "cudf/reduction.hpp" nogil:

    ctypedef enum gdf_scan_op:
        GDF_SCAN_SUM = 0,
        GDF_SCAN_MIN,
        GDF_SCAN_MAX,
        GDF_SCAN_PRODUCT,

cdef extern from "cudf/reduction.hpp" namespace "cudf" nogil:

    cdef gdf_scalar reduce(
        gdf_column *inp,
        operators op,
        gdf_dtype output_dtype,
        size_type ddof
    ) except +

    cdef void scan(
        gdf_column *inp,
        gdf_column *out,
        gdf_scan_op op,
        bool inclusive
    ) except +
