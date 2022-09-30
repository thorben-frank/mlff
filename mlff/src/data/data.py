from typing import (Any, Dict, Sequence, Tuple)

Array = Any
DataTupleT = Tuple[Dict[str, Array], Dict[str, Any]]


class DataTuple:
    def __init__(self, input_keys: Sequence[str], target_keys: Sequence[str]):
        self.input_keys = input_keys
        self.target_keys = target_keys
        self.get_args = lambda data, args: {k: v for (k, v) in data.items() if k in args}

    def __call__(self, ds: Dict[str, Array]) -> DataTupleT:
        inputs = self.get_args(ds, self.input_keys)
        targets = self.get_args(ds, self.target_keys)
        return inputs, targets
