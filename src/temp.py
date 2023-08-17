import xarray as xr
import os

# Directory containing the .nc files
mydir = os.getcwd() # would be the MAIN folder
mydir_tmp = mydir + "/datasets" 

datasets = []

# Loop through all .nc files in the directory
for filename in os.listdir(mydir_tmp):
    if filename.endswith(".nc"):
        filepath = os.path.join(mydir_tmp, filename)
        ds = xr.open_dataset(filepath)
        
        # Convert 'latitude' and 'longitude' to data variables
        ds = ds.reset_coords(names=['latitude', 'longitude'])
        
        # Extract filename (without extension) for unique identification
        key_name = os.path.splitext(filename)[0]
        
        # Add a new coordinate 'polygon_id' to differentiate data from each file
        ds = ds.assign_coords(polygon_id=key_name)
        
        # Expand dataset dimensions based on this new coordinate
        ds = ds.expand_dims('polygon_id')
        
        datasets.append(ds)

# Concatenate datasets along the 'polygon_id' dimension
merged_ds = xr.concat(datasets, dim='polygon_id')

# Save the merged dataset to a single .nc file
output_filepath = os.path.join(mydir_tmp, "merged.nc")
merged_ds.to_netcdf(output_filepath)
