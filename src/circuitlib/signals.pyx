#cython: boundscheck=False, wraparound=False, cdivision=True
from libc.math cimport M_PI
from libc.math cimport sin as csin

cdef struct SignalParams:
    double t
    double value
    double period
    double width
    double x_offset
    double y_offset

cdef double DC(double t, double value, double period, double width, double x_offset, double y_offset) nogil:
    return value

cdef double sin(double t, double value, double period, double width, double x_offset, double y_offset) nogil:
    return value * csin((t - x_offset) * 2 * M_PI / period) + y_offset

cdef double sawtooth(double t, double value, double period, double width, double x_offset, double y_offset) nogil:
    cdef double tmod = modulo((t - x_offset), (period))
    if tmod < width * period:
        return (value * tmod / ((period / 2) * width) - 1) + y_offset
    else:
        return (value * ((period / 2) * (width + 1) - tmod) / ((period / 2) * (1 - width))) + y_offset

cdef double square(double t, double value, double period, double width, double x_offset, double y_offset) nogil:
    cdef double tmod = modulo((t - x_offset), (period))
    if tmod <= (period / 2):
        return value + y_offset
    else:
        return -value + y_offset

cdef double modulo(double n, double m) nogil:
    """Perform modulo in same way as Python."""
    return ((n % m) + m) % m

signals = {
    'DC': DC,
    'sin': sin,
    'sawtooth': sawtooth,
    'square': square
}
