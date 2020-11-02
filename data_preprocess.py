#!/usr/bin/env python3

# Author: Anshuman Khaund <ansh.khaund@gmail.com>

import numpy as np


def reduce_memory_usage(df, verbose=True):
    """ Reduce numeric dtypes to the smallest required type """
    numerics = "int16 int32 int64 float32 float64".split()
    start_memory = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        if col_type := df[col].dtypes in numerics:  # only numeric columns
            mn, mx = df[col].min(), df[col].max()  # min & max of the column
            if str(col_type).startswith('int'):
                if mn >= 0:
                    for type in [np.uint8, np.uint16, np.uint32]:
                        if mx <= np.iinfo(type).max:
                            df[col] = df[col].astype(type)
                            break
                else:
                    for type in [np.int8, np.int16, np.int32]:
                        if mx <= np.iinfo(type).max:
                            df[col] = df[col].astype(type)
                            break
            else:
                for type in [np.float16, np.float32]:
                    if mx <= np.finfo(type).max:
                        df[col] = df[col].astype(type)
                        break
    end_memory = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print("Memory usage decreased from "
              f"{start_memory:.2f} Mb to {end_memory:.2f} Mb, "
              f"({1 - (end_memory / start_memory):.1%} reduction).")
    return df
