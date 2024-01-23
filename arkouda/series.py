from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from pandas._config import get_option  # type: ignore
from typeguard import typechecked
import json

import arkouda.dataframe
from arkouda.accessor import CachedAccessor, DatetimeAccessor, StringAccessor
from arkouda.alignment import lookup
from arkouda.categorical import Categorical
from arkouda.dtypes import float64, int64, SeriesDTypes, numpy_scalars, float_scalars, all_scalars, dtype
from arkouda.groupbyclass import GroupBy, groupable_element_type
from arkouda.index import Index, MultiIndex
from arkouda.numeric import cast as akcast
from arkouda.numeric import value_counts
from arkouda.pdarrayclass import argmaxk, create_pdarray, pdarray, RegistrationError, any
from arkouda.pdarraycreation import arange, array, zeros, full
from arkouda.pdarraysetops import argsort, concatenate, in1d, indexof1d
from arkouda.strings import Strings
from arkouda.util import convert_if_categorical, get_callback

from arkouda.client import generic_msg


# pd.set_option("display.max_colwidth", 65) is being called in DataFrame.py. This will resolve BitVector
# truncation issues. If issues arise, that's where to look for it.

__all__ = [
    "Series",
    "series_indexing"
]

import operator

def series_indexing(arg1, arg2):
    generic_msg(
            cmd="seriesIndexing",
            args={
                "arg1": arg1,
                "arg2": arg2,
                'assume_unique': False}
    );

def natural_binary_operators(cls):
    for name, op in {
        "__add__": operator.add,
        "__sub__": operator.sub,
        "__mul__": operator.mul,
        "__truediv__": operator.truediv,
        "__floordiv__": operator.floordiv,
        "__and__": operator.and_,
        "__or__": operator.or_,
        "__xor__": operator.xor,
        "__eq__": operator.eq,
        "__ge__": operator.ge,
        "__gt__": operator.gt,
        "__le__": operator.le,
        "__lshift__": operator.lshift,
        "__lt__": operator.lt,
        "__mod__": operator.mod,
        "__ne__": operator.ne,
        "__rshift__": operator.rshift,
        "__pow__": operator.pow,
    }.items():
        setattr(cls, name, cls._make_binop(op))

    return cls


def unary_operators(cls):
    for name, op in {
        "__invert__": operator.invert,
        "__neg__": operator.neg,
    }.items():
        setattr(cls, name, cls._make_unaryop(op))

    return cls


def aggregation_operators(cls):
    for name in ["max", "min", "mean", "sum", "std", "var", "argmax", "argmin", "prod"]:
        setattr(cls, name, cls._make_aggop(name))
    return cls


@unary_operators
@aggregation_operators
@natural_binary_operators
class Series:
    """
    One-dimensional arkouda array with axis labels.

    Parameters
    ----------
    index : pdarray, Strings
        an array of indices associated with the data array.
        If empty, it will default to a range of ints whose size match the size of the data.
        optional
    data : Tuple, List, groupable_element_type
        a 1D array. Must not be None.

    Raises
    ------
    TypeError
        Raised if index is not a pdarray or Strings object
        Raised if data is not a pdarray, Strings, or Categorical object
    ValueError
        Raised if the index size does not match data size

    Notes
    -----
    The Series class accepts either positional arguments or keyword arguments.
    If entering positional arguments,
        2 arguments entered:
            argument 1 - data
            argument 2 - index
        1 argument entered:
            argument 1 - data
    If entering 1 positional argument, it is assumed that this is the data argument.
    If only 'data' argument is passed in, Index will automatically be generated.
    If entering keywords,
        'data' (see Parameters)
        'index' (optional) must match size of 'data'
    """

    objType = "Series"

    @typechecked
    def __init__(
        self,
        data: Union[Tuple, List, groupable_element_type],
        index: Optional[Union[pdarray, Strings, Tuple, List, Index]] = None,
    ):
        self.registered_name: Optional[str] = None
        if isinstance(data, (tuple, list)) and len(data) == 2:
            # handles the previous `ar_tuple` case
            if not isinstance(data[0], (pdarray, Index, Strings, Categorical, list, tuple)):
                raise TypeError("indices must be a pdarray, Strings, Categorical, List, or Tuple")
            if not isinstance(data[1], (pdarray, Strings, Categorical)):
                raise TypeError("values must be a pdarray, Strings, or Categorical")
            self.values = data[1]
            self.index = Index.factory(index) if index else Index.factory(data[0])
        else:
            # When only 1 positional argument it will be treated as data and not index
            self.values = array(data) if not isinstance(data, (Strings, Categorical)) else data
            self.index = Index.factory(index) if index is not None else Index(arange(self.values.size))

        if self.index.size != self.values.size:
            raise ValueError("Index size does not match data size")
        self.size = self.index.size

    def __len__(self):
        return self.values.size

    def __repr__(self):
        """
        Return ascii-formatted version of the series.
        """

        if len(self) == 0:
            return "Series([ -- ][ 0 values : 0 B])"

        maxrows = pd.get_option("display.max_rows")
        if len(self) <= maxrows:
            prt = self.to_pandas()
            length_str = ""
        else:
            prt = pd.concat(
                [self.head(maxrows // 2 + 2).to_pandas(), self.tail(maxrows // 2).to_pandas()]
            )
            length_str = f"\nLength {len(self)}"
        return (
            prt.to_string(
                dtype=prt.dtype, min_rows=get_option("display.min_rows"), max_rows=maxrows, length=False
            )
            + length_str
        )

    def validate_key(self, key) -> Union[pdarray, Strings, all_scalars]:
        """
        Validates type requirements for keys when reading or writing the Series.
        Also converts list and tuple arguments into pdarrays.

        Parameters
        ----------
        key: pdarray, Strings, list, tuple, all_scalars
            The key or container of keys that might be used to index into the Series.

        Returns
        -------
        The validated key(s), with lists and tuples converted to pdarrays

        Raises
        ------
        TypeError
            Raised if keys are not boolean values or the type of the labels
            Raised if key is not one of the supported types
        KeyError
            Raised if container of keys has keys not present in the Series
        IndexError
            Raised if the length of a boolean key array is different 
            from the Series 
        """
        if isinstance(key, list):
            key = array(key)
        if isinstance(key, tuple):
            raise TypeError("Series does not support tuple keys")
        if isinstance(key, Series):
            # @TODO align the series indexes
            key = key.values
        
        if isinstance(key, all_scalars):
            if dtype(type(key)) != self.index.dtype:
                raise TypeError("Unexpected key type. Received {} but expected {}".format(dtype(type(key)), self.index.dtype))
        elif isinstance(key, Strings):
            if self.index.dtype != dtype(str):
                raise TypeError("Unexpected key type. Received Strings but expected {}".format(dtype(type(key)), self.index.dtype))
            if any(~in1d(key, self.index.values)):
                raise KeyError("{} not in index".format(key[~in1d(key, self.index.values)]))
        elif isinstance(key, pdarray):
            if key.dtype == self.index.dtype:
                if any(~in1d(key, self.index.values)):
                    raise KeyError("{} not in index".format(key[~in1d(key, self.index.values)]))
            elif key.dtype == bool:
                if key.size != self.index.size:
                    raise IndexError("Boolean index has wrong length: {} instead of {}".format(key.size, self.size))
            else:
                raise TypeError("Unexpected key type. Received {} but expected {}".format(dtype(type(key)), self.index.dtype))
            
        else:
            raise TypeError("Series [] only supports indexing by scalars, lists of scalars, and arrays of scalars.")
        return key
    
    def __getitem__(self, key: Union[Series, pdarray, Strings, list, all_scalars] ) -> Union[Series, all_scalars]:
        """
        Gets values from Series.

        Parameters
        ----------
        key: pdarray, Strings, Series, list, all_scalars
            The key or container of keys to get entries for.

        Returns
        -------
        Series with all entries with matching labels. If only one entry in the
        Series is accessed, returns a scalar.
        """
        key = self.validate_key(key)
        if isinstance(key, all_scalars):
            key = array([key])
        elif key.dtype == bool:
            # boolean array indexes without sorting 
            return Series(index=self.index[key], data=self.values[key])

        indices = indexof1d(key, self.index.values)
        if indices.size == 1:
            return self.values[indices[0]]
        else:
            return Series(index=self.index[indices], data=self.values[indices])
        
    
    def validate_val(self, val: Union[pdarray, Strings, all_scalars, list]) -> Union[pdarray, all_scalars]:
        """
        Validates type requirements for values being written into the Series. Also converts list and tuple arguments into pdarrays.

        Parameters
        ----------
        val: pdarray, Strings, list, all_scalars
            The value or container of values that might be assigned into the Series.

        Returns
        -------
        The validated value, with lists converted to pdarrays

        Raises
        ------
        TypeError
            Raised if val is not the same type or a container with elements of the same time as the Series
            Raised if val is a string or Strings type.
            Raised if val is not one of the supported types
        """
        if isinstance(val, list):
            val = array(val)
        if isinstance(val, all_scalars):
            if dtype(type(val)) != self.values.dtype:
                raise TypeError("Unexpected value type. Received {} but expected {}".format(dtype(type(val)), self.values.dtype))
            if isinstance(val, str):
                raise TypeError("Cannot modify string type dataframes")
        elif isinstance(val, Strings):
            raise TypeError("Cannot modify string type dataframes")
        elif isinstance(val, pdarray):
            if val.dtype != self.values.dtype:
                raise TypeError("Unexpected value type. Received {} but expected {}".format(dtype(type(val)), self.values.dtype))
        else:
            raise TypeError("cannot set with unsupported value type: {}".format(type(val)))
        return val

    def __setitem__(self, key: Union[Series, list, pdarray, Strings, all_scalars], 
                    val: Union[list, pdarray, all_scalars]):
        """
        Sets or adds entries in a Series by label. 

        Parameters
        ----------
        key: pdarray, Strings, Series, list, all_scalars
            The key or container of keys to set entries for.

        val: pdarray, list, all_scalars
            The values to set/add to the Series.

        Raises
        ------
        ValueError
            Raised when setting multiple values to a Series with repeated labels
            Raised when number of values provided does not match the number of 
            entries to set.
        """
        val = self.validate_val(val)
        key = self.validate_key(key)
        
        if isinstance(key, (pdarray,Strings)) and len(key) > 1 and self.has_repeat_labels():
            raise ValueError("Cannot set with multiple keys for Series with repeated labels.")
            
        indices = self.index == key if isinstance(key, all_scalars) else in1d(self.index.values, key)
        tf, counts = GroupBy(indices).count()
        update_count = counts[1] if len(counts) == 2 else 0
        if update_count == 0:
            #adding a new entry
            if isinstance(val, (pdarray, Strings)):
                raise ValueError("Cannot set. Too many values provided")
            new_index_values = concatenate([self.index.values, array([key])])
            self.index = Index.factory(new_index_values)
            self.values = concatenate([self.values, array([val])])
            return
        if isinstance(val, all_scalars):
            self.values[indices] = val
            return
        else:
            if val.size == 1 and isinstance(key, all_scalars):
                self.values[indices] = val[0]
                return
            if update_count != val.size:
                raise ValueError("Cannot set using a list-like indexer with a different length from the value")
            self.values[indices] = val
            return

    def has_repeat_labels(self) -> bool:
        """
        Returns whether the Series has any labels that appear more than once
        """
        tf, counts = GroupBy(self.index.values).count()
        return counts.size != self.index.size
    
    @property 
    def loc(self) -> _LocIndexer:
        """
        Accesses entries of a Series by label

        Parameters
        ----------
        key: pdarray, Strings, Series, list, all_scalars
            The key or container of keys to access entries for
        """
        return _LocIndexer(self)

    @property
    def at(self) -> _LocIndexer:
        """
        Accesses entries of a Series by label

        Parameters
        ----------
        key: pdarray, Strings, Series, list, all_scalars
            The key or container of keys to access entries for
        """
        return _LocIndexer(self)

    @property
    def iloc(self) -> _iLocIndexer:
        """
        Accesses entries of a Series by position

        Parameters
        ----------
        key: int
            The positions or container of positions to access entries for
        """
        return _iLocIndexer('iloc', self)

    @property
    def iat(self) -> _iLocIndexer:
        """
        Accesses entries of a Series by position

        Parameters
        ----------
        key: int
            The positions or container of positions to access entries for
        """
        return _iLocIndexer('iat', self) 


    dt = CachedAccessor("dt", DatetimeAccessor)
    str_acc = CachedAccessor("str", StringAccessor)

    @property
    def shape(self):
        # mimic the pandas return of series shape property
        return (self.values.size,)

    @typechecked
    def isin(self, lst: Union[pdarray, Strings, List]) -> Series:
        """Find series elements whose values are in the specified list

        Input
        -----
        Either a python list or an arkouda array.

        Returns
        -------
        Arkouda boolean which is true for elements that are in the list and false otherwise.
        """
        if isinstance(lst, list):
            lst = array(lst)

        boolean = in1d(self.values, lst)
        return Series(data=boolean, index=self.index)

    @typechecked
    def locate(self, key: Union[int, pdarray, Index, Series, List, Tuple]) -> Series:
        """Lookup values by index label

        The input can be a scalar, a list of scalers, or a list of lists (if the series has a
        MultiIndex). As a special case, if a Series is used as the key, the series labels are
        preserved with its values use as the key.

        Keys will be turned into arkouda arrays as needed.

        Returns
        -------
        A Series containing the values corresponding to the key.
        """
        if isinstance(key, Series):
            # special case, keep the index values of the Series, and lookup the values
            return Series(index=key.index, data=lookup(self.index.index, self.values, key.values))
        elif isinstance(key, MultiIndex):
            idx = self.index.lookup(key.index)
        elif isinstance(key, Index):
            idx = self.index.lookup(key.index)
        elif isinstance(key, pdarray):
            idx = self.index.lookup(key)
        elif isinstance(key, (list, tuple)):
            key0 = key[0]
            if isinstance(key0, list) or isinstance(key0, tuple):
                # nested list. check if already arkouda arrays
                if not isinstance(key0[0], pdarray):
                    # convert list of lists to list of pdarrays
                    key = [array(a) for a in np.array(key).T.copy()]

            elif not isinstance(key0, pdarray):
                # a list of scalers, convert into arkouda array
                try:
                    val = array(key)
                    if isinstance(val, pdarray):
                        key = val
                except Exception:
                    raise TypeError("'key' parameter must be convertible to pdarray")

            # else already list if arkouda array, use as is
            idx = self.index.lookup(key)
        else:
            # scalar value
            idx = self.index == key
        return Series(index=self.index[idx], data=self.values[idx])

    @classmethod
    def _make_binop(cls, operator):
        def binop(self, other):
            if isinstance(other, Series):
                if self.index._check_aligned(other.index):
                    return cls((self.index, operator(self.values, other.values)))
                else:
                    idx = self.index._merge(other.index).index
                    a = lookup(self.index.index, self.values, idx, fillvalue=0)
                    b = lookup(other.index.index, other.values, idx, fillvalue=0)
                    return cls((idx, operator(a, b)))
            else:
                return cls((self.index, operator(self.values, other)))

        return binop

    @classmethod
    def _make_unaryop(cls, operator):
        def unaryop(self):
            return cls((self.index, operator(self.values)))

        return unaryop

    @classmethod
    def _make_aggop(cls, name):
        def aggop(self):
            return getattr(self.values, name)()

        return aggop

    @typechecked
    def add(self, b: Series) -> Series:
        index = self.index.concat(b.index).index

        values = concatenate([self.values, b.values], ordered=False)

        idx, vals = GroupBy(index).sum(values)
        return Series(data=vals, index=idx)

    @typechecked
    def topn(self, n: int = 10) -> Series:
        """Return the top values of the series

        Parameters
        ----------
        n: Number of values to return

        Returns
        -------
        A new Series with the top values
        """
        k = self.index
        v = self.values

        idx = argmaxk(v, n)
        idx = idx[-1 : -n - 1 : -1]

        return Series(index=k.index[idx], data=v[idx])

    @typechecked
    def sort_index(self, ascending: bool = True) -> Series:
        """Sort the series by its index

        Returns
        -------
        A new Series sorted.
        """

        idx = self.index.argsort(ascending=ascending)
        return Series(index=self.index[idx], data=self.values[idx])

    @typechecked
    def sort_values(self, ascending: bool = True) -> Series:
        """Sort the series numerically

        Returns
        -------
        A new Series sorted smallest to largest
        """

        if not ascending:
            if isinstance(self.values, pdarray) and self.values.dtype in (int64, float64):
                # For numeric values, negation reverses sort order
                idx = argsort(-self.values)
            else:
                # For non-numeric values, need the descending arange because reverse slicing
                # is not supported
                idx = argsort(self.values)[arange(self.values.size - 1, -1, -1)]
        else:
            idx = argsort(self.values)
        return Series(index=self.index[idx], data=self.values[idx])

    @typechecked
    def tail(self, n: int = 10) -> Series:
        """Return the last n values of the series"""

        idx_series = self.index[-n:]
        return Series(index=idx_series.index, data=self.values[-n:])

    @typechecked
    def head(self, n: int = 10) -> Series:
        """Return the first n values of the series"""

        idx_series = self.index[0:n]
        return Series(index=idx_series.index, data=self.values[0:n])

    @typechecked
    def to_pandas(self) -> pd.Series:
        """Convert the series to a local PANDAS series"""

        idx = self.index.to_pandas()
        val = convert_if_categorical(self.values)
        return pd.Series(val.to_ndarray(), index=idx)

    @typechecked()
    def to_list(self) -> list:
        p = self.to_pandas()
        return p.to_list()

    @typechecked
    def value_counts(self, sort: bool = True) -> Series:
        """Return a Series containing counts of unique values.

        The resulting object will be in descending order so that the
        first element is the most frequently-occurring element.

        Parameters
        ----------
        sort : Boolean. Whether or not to sort the results.  Default is true.
        """

        dtype = get_callback(self.values)
        idx, vals = value_counts(self.values)
        s = Series(index=idx, data=vals)
        if sort:
            s = s.sort_values(ascending=False)
        s.index.set_dtype(dtype)
        return s

    @typechecked
    def diff(self) -> Series:
        """Diffs consecutive values of the series.

        Returns a new series with the same index and length.  First value is set to NaN.
        """

        values = zeros(len(self), "float64")
        if not isinstance(self.values, Categorical):
            values[1:] = akcast(self.values[1:] - self.values[:-1], "float64")
            values[0] = np.nan
        else:
            raise TypeError("Diff not supported on Series built from Categorical.")

        return Series(data=values, index=self.index)

    @typechecked
    def to_dataframe(
        self, index_labels: List[str] = None, value_label: str = None
    ) -> arkouda.dataframe.DataFrame:
        """Converts series to an arkouda data frame

        Parameters
        ----------
        index_labels:  column names(s) to label the index.
        value_label:  column name to label values.

        Returns
        -------
        An arkouda dataframe.
        """
        list_value_label = [value_label] if isinstance(value_label, str) else value_label

        return Series.concat([self], axis=1, index_labels=index_labels, value_labels=list_value_label)

    @typechecked
    def register(self, user_defined_name: str):
        """
        Register this Series object and underlying components with the Arkouda server

        Parameters
        ----------
        user_defined_name : str
            user defined name the Series is to be registered under,
            this will be the root name for underlying components

        Returns
        -------
        Series
            The same Series which is now registered with the arkouda server and has an updated name.
            This is an in-place modification, the original is returned to support
            a fluid programming style.
            Please note you cannot register two different Series with the same name.

        Raises
        ------
        TypeError
            Raised if user_defined_name is not a str
        RegistrationError
            If the server was unable to register the Series with the user_defined_name

        See also
        --------
        unregister, attach, is_registered

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.
        """
        
        if self.registered_name is not None and self.is_registered():
            raise RegistrationError(f"This object is already registered as {self.registered_name}")
        generic_msg(
            cmd="register",
            args={
                "name": user_defined_name,
                "objType": self.objType,
                "num_idxs": 1,
                "idx_names": [
                    json.dumps(
                        {
                            "codes": self.index.values.codes.name,
                            "categories": self.index.values.categories.name,
                            "NA_codes": self.index.values._akNAcode.name,
                            **(
                                {"permutation": self.index.values.permutation.name}
                                if self.index.values.permutation is not None
                                else {}
                            ),
                            **(
                                {"segments": self.index.values.segments.name}
                                if self.index.values.segments is not None
                                else {}
                            ),
                        }
                    )
                    if isinstance(self.index.values, Categorical)
                    else self.index.values.name
                ],
                "idx_types": [self.index.values.objType],
                "values": json.dumps(
                    {
                        "codes": self.values.codes.name,
                        "categories": self.values.categories.name,
                        "NA_codes": self.values._akNAcode.name,
                        **(
                            {"permutation": self.values.permutation.name}
                            if self.values.permutation is not None
                            else {}
                        ),
                        **(
                            {"segments": self.values.segments.name}
                            if self.values.segments is not None
                            else {}
                        ),
                    }
                )
                if isinstance(self.values, Categorical)
                else self.values.name,
                "val_type": self.values.objType,
            },
        )
        self.registered_name = user_defined_name
        return self

    def unregister(self):
        """
        Unregister this Series object in the arkouda server which was previously
        registered using register() and/or attached to using attach()

        Raises
        ------
        RegistrationError
            If the object is already unregistered or if there is a server error
            when attempting to unregister

        See also
        --------
        register, attach, is_registered

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.
        """
        from arkouda.util import unregister

        if not self.registered_name:
            raise RegistrationError("This object is not registered")
        unregister(self.registered_name)
        self.registered_name = None

    @staticmethod
    @typechecked
    def attach(label: str, nkeys: int = 1) -> Series:
        """
        DEPRECATED
        Retrieve a series registered with arkouda

        Parameters
        ----------
        label: name used to register the series
        nkeys: number of keys, if a multi-index was registerd
        """
        import warnings

        from arkouda.util import attach

        warnings.warn(
            "ak.Series.attach() is deprecated. Please use ak.attach() instead.",
            DeprecationWarning,
        )
        return attach(label)

    @typechecked
    def is_registered(self) -> bool:
        """
         Return True iff the object is contained in the registry or is a component of a
         registered object.

        Returns
        -------
        numpy.bool
            Indicates if the object is contained in the registry

        Raises
        ------
        RegistrationError
            Raised if there's a server-side error or a mis-match of registered components

        See Also
        --------
        register, attach, unregister

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.
        """
        from arkouda.util import is_registered

        if self.registered_name is None:
            return False
        else:
            return is_registered(self.registered_name)

    @classmethod
    @typechecked
    def from_return_msg(cls, repMsg: str) -> Series:
        """
        Return a Series instance pointing to components created by the arkouda server.
        The user should not call this function directly.

        Parameters
        ----------
        repMsg : str
            + delimited string containing the values and indexes

        Returns
        -------
        Series
            A Series representing a set of pdarray components on the server

        Raises
        ------
        RuntimeError
            Raised if a server-side error is thrown in the process of creating
            the Series instance
        """
        data = json.loads(repMsg)
        val_comps = data["value"].split("+|+")
        if val_comps[0] == Categorical.objType.upper():
            values = Categorical.from_return_msg(val_comps[1])  # type: ignore
        elif val_comps[0] == Strings.objType.upper():
            values = Strings.from_return_msg(val_comps[1])  # type: ignore
        else:
            values = create_pdarray(val_comps[1])  # type: ignore

        index = Index.from_return_msg(data["index"])

        return cls(values, index)

    @staticmethod
    @typechecked
    def _all_aligned(array: List) -> bool:
        """Is an array of Series indexed aligned?"""

        itor = iter(array)
        a1 = next(itor).index
        for a2 in itor:
            if a1._check_aligned(a2.index) is False:
                return False
        return True

    @staticmethod
    @typechecked
    def concat(
        arrays: List, axis: int = 0, index_labels: List[str] = None, value_labels: List[str] = None
    ) -> Union[arkouda.dataframe.DataFrame, Series]:
        """Concatenate in arkouda a list of arkouda Series or grouped arkouda arrays horizontally or
        vertically. If a list of grouped arkouda arrays is passed they are converted to a series. Each
        grouping is a 2-tuple with the first item being the key(s) and the second being the value.
        If horizontal, each series or grouping must have the same length and the same index. The index
        of the series is converted to a column in the dataframe.  If it is a multi-index,each level is
        converted to a column.

        Parameters
        ----------
        arrays:  The list of series/groupings to concat.
        axis  :  Whether or not to do a verticle (axis=0) or horizontal (axis=1) concatenation
        index_labels:  column names(s) to label the index.
        value_labels:  column names to label values of each series.

        Returns
        -------
        axis=0: an arkouda series.
        axis=1: an arkouda dataframe.
        """

        if len(arrays) == 0:
            raise IndexError("Array length must be non-zero")

        types = {type(x) for x in arrays}
        if len(types) != 1:
            raise TypeError(f"Items must all have same type: {types}")

        if isinstance(arrays[0], tuple):
            arrays = [Series(i) for i in arrays]

        if axis == 1:
            # Horizontal concat
            if value_labels is None:
                value_labels = [f"val_{i}" for i in range(len(arrays))]

            if Series._all_aligned(arrays):
                data = next(iter(arrays)).index.to_dict(index_labels)

                if value_labels is not None:
                    # Expect value_labels to always be not None; were doing the check for mypy
                    for col, label in zip(arrays, value_labels):
                        data[str(label)] = col.values

            else:
                aitor = iter(arrays)
                idx = next(aitor).index
                idx = idx._merge_all([i.index for i in aitor])

                data = idx.to_dict(index_labels)

                if value_labels is not None:
                    # Expect value_labels to always be not None; were doing the check for mypy
                    for col, label in zip(arrays, value_labels):
                        data[str(label)] = lookup(col.index.index, col.values, idx.index, fillvalue=0)

            return arkouda.dataframe.DataFrame(data)
        else:
            # Verticle concat
            idx = arrays[0].index
            v = arrays[0].values
            for other in arrays[1:]:
                idx = idx.concat(other.index)
                v = concatenate([v, other.values], ordered=True)
            return Series(index=idx.index, data=v)

    @staticmethod
    @typechecked
    def pdconcat(arrays: List, axis: int = 0, labels: Strings = None) -> Union[pd.Series, pd.DataFrame]:
        """Concatenate a list of arkouda Series or grouped arkouda arrays, returning a PANDAS object.

        If a list of grouped arkouda arrays is passed they are converted to a series. Each grouping
        is a 2-tuple with the first item being the key(s) and the second being the value.

        If horizontal, each series or grouping must have the same length and the same index. The index of
        the series is converted to a column in the dataframe.  If it is a multi-index,each level is
        converted to a column.

        Parameters
        ----------
        arrays:  The list of series/groupings to concat.
        axis  :  Whether or not to do a verticle (axis=0) or horizontal (axis=1) concatenation
        labels:  names to give the columns of the data frame.

        Returns
        -------
        axis=0: a local PANDAS series
        axis=1: a local PANDAS dataframe
        """
        if len(arrays) == 0:
            raise IndexError("Array length must be non-zero")

        types = {type(x) for x in arrays}
        if len(types) != 1:
            raise TypeError(f"Items must all have same type: {types}")

        if isinstance(arrays[0], tuple):
            arrays = [Series(i) for i in arrays]

        if axis == 1:
            idx = arrays[0].index.to_pandas()

            cols = []
            for col in arrays:
                cols.append(pd.Series(data=col.values.to_ndarray(), index=idx))
            retval = pd.concat(cols, axis=1)
            if labels is not None:
                retval.columns = labels
        else:
            retval = pd.concat([s.to_pandas() for s in arrays])

        return retval

class _LocIndexer():
    def __init__(self, series):
        self.series = series

    def __getitem__(self, key):
        return self.series[key]

    def __setitem__(self, key, val):
        self.series[key] = val

class _iLocIndexer():
    def __init__(self, method_name, series):
        self.name = method_name
        self.series = series

    def validate_key(self, key) -> Union[pdarray, Strings, all_scalars]:
        if isinstance(key, list):
            key = array(key)
        if isinstance(key, tuple):
            raise TypeError(".{} does not support tuple arguments".format(self.name))
        if isinstance(key, pdarray):
            if len(key) == 0:
                raise ValueError("Cannot index using 0-length iterables.")
            if key.dtype != int64 and key.dtype != bool:
                raise TypeError(".{} requires integer keys".format(self.name))
            
            if key.dtype == bool and key.size != self.series.size:
                raise IndexError("Boolean index has wrong length: {} instead of {}".format(key.size, self.series.size))
            elif any(key >= self.series.size):
                raise IndexError("{} cannot enlarge its target object.".format(self.name))
            
        elif isinstance(key, int):
            if key >= self.series.size:
                raise IndexError("{} cannot enlarge its target object.".format(self.name))
        else:
            raise TypeError(".{} requires integer keys".format(self.name))
        return key
    
    def validate_val(self, val) -> Union[pdarray, all_scalars]:
        return self.series.validate_val(val)
    
    def __getitem__(self, key) -> Series:
        key = self.validate_key(key)
        if isinstance(key, all_scalars):
            key = array([key])
        return Series(index=self.series.index[key], data=self.series.values[key])
    
    def __setitem__(self, key, val):
        key = self.validate_key(key)
        val = self.validate_val(val)

        if isinstance(val, all_scalars):
            self.series.values[key] = val
            return
        else:
            if isinstance(key, all_scalars):
                self.series.values[key] = val
                return
            if key.dtype == int64 and len(val) != len(key):
                raise ValueError("cannot set using a list-like indexer with a different length than the value")
        self.series.values[key] = val
            
        
