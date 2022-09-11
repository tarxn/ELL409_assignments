import numpy as np
import get_data as dat
import gd_functions as gdf

w_new=gdf.stochastic_gd(dat.X_subset, dat.y_subset, num_it=200000, learn_rate=0.01)
# print("w_alt=",w_new[100:110])
print(gdf.error(w_new, dat.X_subset,dat.y_subset))
print(gdf.cost_function(w_new,dat.X_subset,dat.y_subset))



