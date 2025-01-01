# -*- coding: utf-8 -*-

from abc import ABC
from dataclasses import dataclass, field
from typing import List, Union, Optional, Dict

import dataclasses
import json
import sys
from argparse import ArgumentParser
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, List, NewType, Optional, Tuple, Union


DataClass = NewType("DataClass", Any)
DataClassType = NewType("DataClassType", Any)

"""### Define Arguments"""
@dataclass
class DataArguments:
    dataset: str = field(default='dblp.v13',
        metadata={"help": 'Name/Path of the dataset.'})
    target: str = field(default='paper',
        metadata={"help": 'Targeted node type in the heterogeneous graph.'})
    num_classes: int = field(default=3,
        metadata={"help":"Number of classes for the classification task."})


@dataclass
class ModelArguments:
    model_name: str = field(default='hgt',
        metadata={"help": 'Name the Model.'})
    in_channels: int = field(default=6,
        metadata={"help":"input channels, which follows the dim of nodes features."})
    hidden_channels: int = field(default=128,
        metadata={"help":"hidden channels."})
    num_heads: int = field(default=4,
        metadata={"help":"Number of attention headers."})
    num_layers: int = field(default=2,
        metadata={"help":"Number of GNN layers."})
    readout: str = field(default='global_mean',
        metadata={"help": "readout from graph, option: all, global_mean, global_max"})
    use_RTE: bool = field(default=True,
        metadata={"help": "whether use Relational Temporal Encoding."})
    

@dataclass
class TrainingArguments:

    training_size: int = field(default=120000,
        metadata={"help":"Targeted node size for train, eval and test."})
    observation_points: str = field(default="2009=2011=2013",
        metadata={"help": "Observation points for evaluation."})
    train_proportion: float = field(default=0.8,
        metadata={"help": "The proportion of test data."})                   
    test_proportion: float = field(default=0.2,
        metadata={"help": "The proportion of test data."})
    
    task_name: str = field(default='task',
        metadata={"help": "Task name."})
    seed: int = field(default=42,
        metadata={"help": "set random state."})
    batch_size: int = field(default=256,
        metadata={"help": "Training batch size."})
    num_workers: int = field(default=8)
    sampler: str = field(default='neighbor',
        metadata={"help": "Sampler for training."})
    num_samples: int = field(default=20,
        metadata={"help": "Number of samples in each layer for DataSampler."})
    lr: float = field(default=0.005,
        metadata={"help": "Learning rate."})
    weight_decay: float = field(default=0.0001,
        metadata={"help": "Rate of weight decay."})
    patience: int = field(default=5,)
    scheduler: int = field(default=0,)
    num_epochs: int = field(default=20,
        metadata={"help": "Maximum Number of Epochs to Train the Model."})
    

class HfArgumentParser(ArgumentParser):
    """
    This subclass of `argparse.ArgumentParser` uses type hints on dataclasses to generate arguments. More details can be 
    found from HuggingFace's transformers library.
    The class is designed to play well with the native argparse. In particular, you can add more (non-dataclass backed)
    arguments to the parser after initialization and you'll get the output back after parsing as an additional
    namespace.
    """

    dataclass_types: Iterable[DataClassType]

    def __init__(self, dataclass_types: Union[DataClassType, Iterable[DataClassType]], **kwargs):

        super().__init__(**kwargs)
        if dataclasses.is_dataclass(dataclass_types):
            dataclass_types = [dataclass_types]
        self.dataclass_types = dataclass_types
        for dtype in self.dataclass_types:
            self._add_dataclass_arguments(dtype)

    def _add_dataclass_arguments(self, dtype: DataClassType):
        for field in dataclasses.fields(dtype):
            if not field.init:
                continue
            field_name = f"--{field.name}"
            kwargs = field.metadata.copy()
            # field.metadata is not used at all by Data Classes,
            # it is provided as a third-party extension mechanism.
            if isinstance(field.type, str):
                raise ImportError(
                    "This implementation is not compatible with Postponed Evaluation of Annotations (PEP 563),"
                    "which can be opted in from Python 3.7 with `from __future__ import annotations`."
                    "We will add compatibility when Python 3.9 is released."
                )
            typestring = str(field.type)
            for prim_type in (int, float, str):
                for collection in (List,):
                    if (
                        typestring == f"typing.Union[{collection[prim_type]}, NoneType]"
                        or typestring == f"typing.Optional[{collection[prim_type]}]"
                    ):
                        field.type = collection[prim_type]
                if (
                    typestring == f"typing.Union[{prim_type.__name__}, NoneType]"
                    or typestring == f"typing.Optional[{prim_type.__name__}]"
                ):
                    field.type = prim_type

            if isinstance(field.type, type) and issubclass(field.type, Enum):
                kwargs["choices"] = list(field.type)
                kwargs["type"] = field.type
                if field.default is not dataclasses.MISSING:
                    kwargs["default"] = field.default
            elif field.type is bool or field.type is Optional[bool]:
                if field.type is bool or (field.default is not None and field.default is not dataclasses.MISSING):
                    kwargs["action"] = "store_false" if field.default is True else "store_true"
                if field.default is True:
                    field_name = f"--no_{field.name}"
                    kwargs["dest"] = field.name
            elif hasattr(field.type, "__origin__") and issubclass(field.type.__origin__, List):
                kwargs["nargs"] = "+"
                kwargs["type"] = field.type.__args__[0]
                assert all(
                    x == kwargs["type"] for x in field.type.__args__
                ), "{} cannot be a List of mixed types".format(field.name)
                if field.default_factory is not dataclasses.MISSING:
                    kwargs["default"] = field.default_factory()
            else:
                kwargs["type"] = field.type
                if field.default is not dataclasses.MISSING:
                    kwargs["default"] = field.default
                elif field.default_factory is not dataclasses.MISSING:
                    kwargs["default"] = field.default_factory()
                else:
                    kwargs["required"] = True
            self.add_argument(field_name, **kwargs)

    def parse_args_into_dataclasses(
        self, args=None, return_remaining_strings=False, look_for_args_file=True, args_filename=None
    ) -> Tuple[DataClass, ...]:

        if args_filename or (look_for_args_file and len(sys.argv)):
            if args_filename:
                args_file = Path(args_filename)
            else:
                args_file = Path(sys.argv[0]).with_suffix(".args")

            if args_file.exists():
                fargs = args_file.read_text().split()
                args = fargs + args if args is not None else fargs + sys.argv[1:]
                # in case of duplicate arguments the first one has precedence
                # so we append rather than prepend.
        namespace, remaining_args = self.parse_known_args(args=args)
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: v for k, v in vars(namespace).items() if k in keys}
            for k in keys:
                delattr(namespace, k)
            obj = dtype(**inputs)
            outputs.append(obj)
        if len(namespace.__dict__) > 0:
            # additional namespace.
            outputs.append(namespace)
        if return_remaining_strings:
            return (*outputs, remaining_args)
        else:
            if remaining_args:
                raise ValueError(f"Some specified arguments are not used by the HfArgumentParser: {remaining_args}")

            return (*outputs,)

    def parse_json_file(self, json_file: str) -> Tuple[DataClass, ...]:

        data = json.loads(Path(json_file).read_text())
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype)}
            inputs = {k: v for k, v in data.items() if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)
        return (*outputs,)

    def parse_dict(self, args: dict) -> Tuple[DataClass, ...]:

        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype)}
            inputs = {k: v for k, v in args.items() if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)
        return (*outputs,)
