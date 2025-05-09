import numpy as np
from bitarray import bitarray
from typing import Any

class FeatureMapCompressorBase:
    def __init__(self):
        np.set_printoptions(threshold=np.inf)
        self.log_file: Any = None

    def log_write(self, msg: Any = ''):
        """Write to the log file, if there is a log file."""
        if self.log_file:
            for line in str(msg).split('\n'):
                self.log_file.write(4*self.indent*' ' + line + '\n')

    def log_as_str(self, obj: Any) -> str:
        if isinstance(obj, bitarray):
            return str(np.array(list(obj)))
        if isinstance(obj, float):
            return str(np.float32(obj))
        return str(obj)

    def log_indent(self):
        """Increase the indent in the log file."""
        self.indent += 1

    def log_deindent(self):
        """Decrease the indent in the log file."""
        self.indent -= 1
