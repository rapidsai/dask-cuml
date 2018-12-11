import dask_cudf
from cuml import Mean
import numpy as np

class Mean(object):
    
    def calculate(self, dask_cudf):

        # Calculate the mean using the GPUs
        def calc_mean(df):
            m = Mean()
            
            mus = m.calculate(df)
            headers = df.columns.tolist()
            
            return cudf.DataFrame.from_dict({"mean": mus, "col": headers})
            
        # Map each partition to GPU on local worker
        dask_cudf = dask_cudf.map_partitions(m.calculate)
        
        # Merge means together across all partitions
        return dask_cudf.group_by("mean").mean()