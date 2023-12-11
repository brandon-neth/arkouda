from __future__ import annotations

import json
import os
import random
from collections import UserDict
from typing import Callable, Dict, List, Optional, Union, cast
from warnings import warn

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from typeguard import typechecked

from arkouda.categorical import Categorical
from arkouda.client import generic_msg, maxTransferBytes
from arkouda.client_dtypes import BitVector, Fields, IPv4
from arkouda.dtypes import bool as akbool
from arkouda.dtypes import float64 as akfloat64
from arkouda.dtypes import int64 as akint64
from arkouda.dtypes import resolve_scalar_dtype
from arkouda.groupbyclass import GROUPBY_REDUCTION_TYPES
from arkouda.groupbyclass import GroupBy as akGroupBy
from arkouda.groupbyclass import unique
from arkouda.index import Index
from arkouda.join import inner_join
from arkouda.numeric import cast as akcast
from arkouda.numeric import cumsum, where
from arkouda.pdarrayclass import RegistrationError, pdarray
from arkouda.pdarraycreation import arange, array, create_pdarray, full, zeros
from arkouda.pdarraysetops import concatenate, in1d, intersect1d
from arkouda.row import Row
from arkouda.segarray import SegArray
from arkouda.series import Series
from arkouda.sorting import argsort, coargsort
from arkouda.strings import Strings
from arkouda.timeclass import Datetime, Timedelta


class DFIndexer:
  def __init__(self, name, obj):
    self.obj = obj
    self.name = name
  
  def ndim(self):
    ndim = self.obj.ndim
    if ndim > 2:
      raise ValueError("Indexers only support up to 2 dimensional data.")
    return ndim

class _AtIndexer(DFIndexer):

  def __getitem__(self, key):
    if not isinstance(key, tuple):
      raise ValueError(".at indexing requires scalar arguments for row and column.")
    return self.obj.loc[key]
  
class _iAtIndexer(DFIndexer):

  def __getitem__(self, key):
    if not isinstance(key, tuple):
      raise ValueError(".iat indexing requires integer arguments for row and column.")
    return self.obj.iloc[key]
  
class _LocIndexer(DFIndexer):

  def __getitem__(self, key):
    print("loc.__getitem__ ", key)
    if isinstance(key, tuple):
      if len(key) == 1:
        return self.obj[key]
      elif len(key) == 2:
        return self.obj[key[0]][key[1]]
    elif isinstance(key, list):
      if len(key) == 0:
        return self.obj[key]
      # if list of index values, get those rows
      index_type = self.obj.index.dtype
      key_types = {resolve_scalar_dtype(k) for k in key}
      if len(key_types) != 1:
        raise ValueError(".loc argument must have single data type")
      if resolve_scalar_dtype(key[0]) == index_type:
        #list of index values, so get those rows
        row_indices = [self.obj.index[k] for k in key]
        return self.obj.iloc[row_indices]
    elif isinstance(key, int):
      print("dataframe index object:", self.obj.index)
      idx = self.obj.index.lookup(key).argmax()
      print("lookup result:", idx)
      print("Index value for loc argument:", idx)
      return self.obj.iloc[int(idx)]
    return None
  
class _iLocIndexer(DFIndexer):

  def __getitem__(self, key):
    print("iloc.__getitem__ ", key)
    print("type of index:", type(key))
    if isinstance(key, tuple):
      if len(key) == 2:
        return self.obj[key[0]][self.obj.columns[key[1]]]
    if isinstance(key, list):
      if len(key) == 0:
        return self.obj[key]
      index_type = self.obj.index.dtype
      key_types = {resolve_scalar_dtype(k) for k in key}
      if len(key_types) != 1:
        raise ValueError(".iloc argument must have single data type")
      if resolve_scalar_dtype(key[0]) != akint64 and resolve_scalar_dtype(key[0]) != akbool:
        raise ValueError(".iloc argument must be list of integers or booleans")
      return self.obj[array(key)]
    return self.obj[key]