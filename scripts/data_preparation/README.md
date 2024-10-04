### Used for generate voxels from raw event (h5 file)

Convert raw GoPro event file (h5) to voxel file (h5):
```
python make_voxels_esim_ice.py --input_path /your/path/to/raw/event/h5file --save_path /your/path/to/save/voxel/h5file --voxel_method ice_esim
```

Convert raw REBlur event file (h5) to voxel file (h5):
```
python make_voxels_real_ice.py --input_path /your/path/to/raw/event/h5file --save_path /your/path/to/save/voxel/h5file --voxel_method ice_real_data
```

