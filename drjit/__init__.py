import sys
import os

if sys.version_info < (3, 8):
    raise ImportError("Dr.Jit requires Python >= 3.8")

if os.name == 'nt':
    # Specify DLL search path for windows (no rpath on this platform..)
    d = __file__
    for i in range(3):
        d = os.path.dirname(d)
    try: # try to use Python 3.8's DLL handling
        os.add_dll_directory(d)
    except AttributeError:  # otherwise use PATH
        os.environ['PATH'] += os.pathsep + d
    del d, i

del sys, os

# Native extension defining low-level arrays
import drjit.drjit_ext as drjit_ext  # noqa

self = vars()

# Install constants in global scope
import drjit.const as const  # noqa
for k, v in const.__dict__.items():
    if k.startswith('_'):
        continue
    self[k] = v

# ------------------------------------------------------------------------------

def sqr(arg, /):
    return arg * arg

def isnan(arg, /):
    """
    Performs an elementwise test for *NaN* (Not a Number) values

    Args:
        arg (object): A Dr.Jit array or other kind of numeric sequence type.

    Returns:
        :py:func:`mask_t(arg) <mask_t>`: A mask value describing the result of the test.
    """
    result = arg == arg
    if isinstance(result, bool):
        return not result
    else:
        return ~result


def isinf(arg, /):
    """
    Performs an elementwise test for positive or negative infinity

    Args:
        arg (object): A Dr.Jit array or other kind of numeric sequence type.

    Returns:
        :py:func:`mask_t(arg) <mask_t>`: A mask value describing the result of the test
    """
    return abs(arg) == float('inf')


def isfinite(arg, /):
    """
    Performs an elementwise test that checks whether values are finite and not
    equal to *NaN* (Not a Number)

    Args:
        arg (object): A Dr.Jit array or other kind of numeric sequence type.

    Returns:
        :py:func:`mask_t(arg) <mask_t>`: A mask value describing the result of the test
    """
    return abs(arg) < float('inf')


def all_nested(arg, /):
    """
    Iterates :py:func:`all` until the type of the return value no longer
    changes. This can be used to reduce a nested mask array into a single
    value.
    """
    while True:
        arg_t = type(arg)
        arg = all(arg)
        if type(arg) is arg_t:
            break;
    return arg


def any_nested(arg, /):
    """
    Iterates :py:func:`any` until the type of the return value no longer
    changes. This can be used to reduce a nested mask array into a single
    value.
    """
    while True:
        arg_t = type(arg)
        arg = any(arg)
        if type(arg) is arg_t:
            break;
    return arg


def allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False):
    r'''
    Returns ``True`` if two arrays are element-wise equal within a given error
    tolerance.

    The function considers both absolute and relative error thresholds. Specifically
    **a** and **b** are considered equal if all elements satisfy

    .. math::
        |a - b| \le |b| \cdot \mathrm{rtol} + \mathrm{atol}.

    Args:
        a (object): A Dr.Jit array or other kind of numeric sequence type.
        b (object): A Dr.Jit array or other kind of numeric sequence type.
        rtol (float): A relative error threshold. The default is :math:`10^{-5}`.
        atol (float): An absolute error threshold. The default is :math:`10^{-8}`.
        equal_nan (bool): If **a** and **b** *both* contain a *NaN* (Not a Number) entry,
                          should they be considered equal? The default is ``False``.

    Returns:
        bool: The result of the comparison.
    '''

    if is_array_v(a) or is_array_v(b):
        # No derivative tracking in the following
        a, b = detach(a), detach(b)

        if is_array_v(a):
            diff = a - b
        else:
            diff = b - a

        a = type(diff)(a)
        b = type(diff)(b)

        cond = abs(diff) <= abs(b) * rtol + atol

        if equal_nan:
            cond |= isnan(a) & isnan(b)

        return all_nested(cond)

    def safe_len(x):
        try:
            return len(x)
        except TypeError:
            return 0

    def safe_getitem(x, len_x, i):
        if len_x == 0:
            return x
        elif len_x == 1:
            return x[0]
        else:
            return x[i]

    len_a, len_b = safe_len(a), safe_len(b)
    len_ab = maximum(len_a, len_b)

    if len_a != len_ab and len_a > 1 or \
       len_b != len_ab and len_b > 1:
        raise RuntimeError('drjit.allclose(): incompatible sizes '
                           '(%i and %i)!' % (len_a, len_b))
    elif len_ab == 0:
        if equal_nan and isnan(a) and isnan(b):
            return True
        return abs(a - b) <= abs(b) * rtol + atol
    else:
        for i in range(len_ab):
            ia = safe_getitem(a, len_a, i)
            ib = safe_getitem(b, len_b, i)
            if not allclose(ia, ib, rtol, atol, equal_nan):
                return False
        return True


def clip(value, min, max):
    r'''
    Clip the provided input to the given interval.

    This function is equivalent to

    .. code-block::

        dr.maximum(dr.minimum(value, max), min)

    Args:
        value (int | float | drjit.ArrayBase): A Python or Dr.Jit type
        min (int | float | drjit.ArrayBase): A Python or Dr.Jit type
        max (int | float | drjit.ArrayBase): A Python or Dr.Jit type

    Returns:
        float | drjit.ArrayBase: Clipped input
    '''
    return maximum(minimum(value, max), min)


def power(x, y):
    r'''
    Raise the first input value to the power given as second input value.

    This function handles both the case of integer and floating-point exponents.
    Moreover, when the exponent is an array, the function will calculate the
    element-wise powers of the input values.

    Args:
        x (int | float | drjit.ArrayBase): A Python or Dr.Jit array type as input value
        y (int | float | drjit.ArrayBase): A Python or Dr.Jit array type as exponent

    Returns:
        int | float | drjit.ArrayBase: input value raised to the power
    '''
    if type(y) is int:
        n = abs(y)
        result = type(x)(1)
        x = type(x)(x)
        while n:
            if n & 1:
                result *= x
            x *= x
            n >>= 1
        return result if y >= 0 else rcp(result)
    else:
        return exp2(log2(x) * y)


# -------------------------------------------------------------------
#   "Safe" functions that avoid domain errors due to rounding
# -------------------------------------------------------------------


def safe_sqrt(a):
    r'''
    Safely evaluate the square root of the provided input avoiding domain errors.

    Negative inputs produce a ``0.0`` output value.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Square root of the input
    '''
    result = sqrt(maximum(a, 0))
    if is_diff_v(a) and grad_enabled(a):
        alt = sqrt(maximum(a, epsilon(a)))
        result = replace_grad(result, alt)
    return result


def safe_asin(a):
    r'''
    Safe wrapper around :py:func:`drjit.asin` that avoids domain errors.

    Input values are clipped to the :math:`(-1, 1)` domain.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Arcsine approximation
    '''
    result = asin(clip(a, -1, 1))
    if is_diff_v(a) and grad_enabled(a):
        alt = asin(clip(a, -one_minus_epsilon(a), one_minus_epsilon(a)))
        result = replace_grad(result, alt)
    return result


def safe_acos(a):
    r'''
    Safe wrapper around :py:func:`drjit.acos` that avoids domain errors.

    Input values are clipped to the :math:`(-1, 1)` domain.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Arccosine approximation
    '''
    result = acos(clip(a, -1, 1))
    if is_diff_v(a) and grad_enabled(a):
        alt = acos(clip(a, -one_minus_epsilon(a), one_minus_epsilon(a)))
        result = replace_grad(result, alt)
    return result


def meshgrid(*args, indexing='xy'):
    r'''
    Creates a grid coordinates based on the coordinates contained in the
    provided one-dimensional arrays.

    The indexing keyword argument allows this function to support both matrix
    and Cartesian indexing conventions. If given the string 'ij', it will return
    a grid coordinates with matrix indexing. If given 'xy', it will return a
    grid coordinates with Cartesian indexing.

    .. codeblock::

        import drjit as dr

        x, y = dr.meshgrid(
            dr.arange(dr.llvm.UInt, 4),
            dr.arange(dr.llvm.UInt, 4)
        )

        # x = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
        # y = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]

    Args:
        args (drjit.ArrayBase): Dr.Jit one-dimensional coordinate arrays

        indexing (str): Specifies the indexing conventions

    Returns:
        tuple: Grid coordinates

    '''
    if indexing != "ij" and indexing != "xy":
        raise Exception("meshgrid(): 'indexing' argument must equal"
                        " 'ij' or 'xy'!")

    if len(args) == 0:
        return ()
    elif len(args) == 1:
        return args[0]

    t = type(args[0])
    for v in args:
        if size_v(v) != Dynamic or \
           depth_v(v) != 1 or type(v) is not t:
            raise Exception("meshgrid(): consistent 1D dynamic arrays expected!")

    size = prod((len(v) for v in args))
    index = arange(uint32_array_t(t), size)

    result = []

    # This seems non-symmetric but is necessary to be consistent with NumPy
    if indexing == "xy":
        args = (args[1], args[0], *args[2:])

    for v in args:
        size //= len(v)
        index_v = index // size
        index = index - index_v * size
        result.append(gather(t, v, index_v))

    if indexing == "xy":
        result[0], result[1] = result[1], result[0]

    return tuple(result)
