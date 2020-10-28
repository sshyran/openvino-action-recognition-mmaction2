import ctypes
import random
import string
import ast

from mmcv import DictAction


def get_random_string(length=15):
    """Get random string with letters and digits.

    Args:
        length (int): Length of random string. Default: 15.
    """
    return ''.join(
        random.choice(string.ascii_letters + string.digits)
        for _ in range(length))


def get_thread_id():
    """Get current thread id."""
    # use ctype to find thread id
    thread_id = ctypes.CDLL('libc.so.6').syscall(186)
    return thread_id


def get_shm_dir():
    """Get shm dir for temporary usage."""
    return '/dev/shm'


class ExtendedDictAction(DictAction):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options should
    be passed as comma separated values, i.e KEY=V1,V2,V3
    """

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split('=', maxsplit=1)
            if '[' in val or '(' in val:
                val = ast.literal_eval(val)
            else:
                val = [self._parse_int_float_bool(v) for v in val.split(',')]
                if len(val) == 1:
                    val = val[0]
            options[key] = val

        setattr(namespace, self.dest, options)
