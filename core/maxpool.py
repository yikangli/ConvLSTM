import theano
import numpy as np
import theano.tensor as T
from theano.tensor.signal.downsample import max_pool_2d
import numbers

# Could use fractions.gcd, but this works
def _gcd(num1, num2):
    """Calculate gcd(num1, num2), greatest common divisor, using euclid's
    algorithm"""
    while (num2 != 0):
        if num1 > num2:
            num1, num2 = num2, num1
        num2 -= (num2 // num1) * num1
    return num1


def _lcm(num1, num2):
    """Calculate least common multiple of num1 and num2"""
    return num1 * num2 / _gcd(num1, num2)


def fancy_max_pool(input_tensor, pool_shape, pool_stride,
                    ignore_border=False):
    """Using theano built-in maxpooling, create a more flexible version.
    Obviously suboptimal, but gets the work done."""

    if isinstance(pool_shape, numbers.Number):
        pool_shape = pool_shape,
    if isinstance(pool_stride, numbers.Number):
        pool_stride = pool_stride,

    if len(pool_shape) == 1:
        pool_shape = pool_shape * 2
    if len(pool_stride) == 1:
        pool_stride = pool_stride * 2

    lcmh, lcmw = [_lcm(p, s) for p, s in zip(pool_shape, pool_stride)]
    dsh, dsw = lcmh / pool_shape[0], lcmw / pool_shape[1]

    pre_shape = input_tensor.shape[:-2]
    length = T.prod(pre_shape)
    post_shape = input_tensor.shape[-2:]
    new_shape = T.concatenate([[length], post_shape])
    reshaped_input = input_tensor.reshape(new_shape, ndim=3)
    sub_pools = []
    for sh in range(0, lcmh, pool_stride[0]):
        sub_pool = []
        sub_pools.append(sub_pool)
        for sw in range(0, lcmw, pool_stride[1]):
            full_pool = max_pool_2d(reshaped_input[:, sh:,
                                                      sw:],
                                    pool_shape, ignore_border=ignore_border)
            ds_pool = full_pool[:, ::dsh, ::dsw]
            concat_shape = T.concatenate([[length], ds_pool.shape[-2:]])
            sub_pool.append(ds_pool.reshape(concat_shape, ndim=3))
    output_shape = (length,
                    T.sum([l[0].shape[1] for l in sub_pools]),
                    T.sum([i.shape[2] for i in sub_pools[0]]))
    output = T.zeros(output_shape)
    for i, line in enumerate(sub_pools):
        for j, item in enumerate(line):
            output = T.set_subtensor(output[:, i::lcmh / pool_stride[0],
                                               j::lcmw / pool_stride[1]],
                                     item)
    return output.reshape(T.concatenate([pre_shape, output.shape[1:]]),
                          ndim=input_tensor.ndim)

