import dask_cudf
from cuML import Mean as cumlMean
import numpy as np
import cudf

# TODO: Let's not rely totally on dask-cudf... 
# if they only want to use our dask_cuml.Array, 
# they shouldn't need the other as a dependency.
class Mean(object):
    def calculate(self, dask_df):
      # TODO: Should data be verified float?
      def calc_mean(df): 
        m = cumlMean()
        mu = m.calculate(df)
        return cudf.DataFrame([("mean", mu), ("col", range(0, len(df.columns.tolist())))])
      mu_df = dask_df.map_partitions(calc_mean)
      
      # The following hack is only for demo purposes. Once cudf is fixed to have 
      # Groupby->apply functionality, this can all be done in the dask layer. 
      
      # Once cuml performs the harder work of calculating the mean for each 
      # partition, we only have 'npartitions' number of reductions to perform
      # on 'ncols' number of rows.
      return dask_cudf.from_cudf(mu_df.compute().groupby("col").mean(), chunksize = 500)