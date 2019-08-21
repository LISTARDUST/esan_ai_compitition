import pandas as pd
import numpy as np

def column_memory_usage(dt):
    for dtype in ['float', 'int', 'object']:
        selected_dtype = dt.select_dtypes(include=[dtype])
        mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
        mean_usage_mb = mean_usage_b / 1024 ** 2
    return print("Average memory usage for {} columns: {: 03.2f}MB".format(
            dtype, mean_usage_mb))

def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)

