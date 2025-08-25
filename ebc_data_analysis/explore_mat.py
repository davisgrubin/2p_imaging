import h5py

dff_file = './data/imaging/E4L7-Control/20241030_crbl_ebc_control/dff.h5'
file = h5py.File(dff_file, 'r')

print("Top-level keys:")
for key in file.keys():
    print(f"- {key}")

print("\nContent of 'name' dataset:")
print(file['name'][:])

print("\nDetailed content:")
for key in file.keys():
    item = file[key]
    print(f"- {key}: {item.name}")
    if isinstance(item, h5py.Group):
        print("  (Group)")
        for subkey in item.keys():
            subitem = item[subkey]
            print(f"    - {subkey}: {subitem.name}")
            if isinstance(subitem, h5py.Dataset):
                print(f"      Shape: {subitem.shape}, dtype: {subitem.dtype}")
                if subitem.shape and subitem.shape[0] < 10: #example to prevent printing huge datasets.
                    try:
                        print(f"      Sample data: {subitem[:]}")
                    except TypeError:
                        print(f"      Data cannot be printed.")

    elif isinstance(item, h5py.Dataset):
        print(f"  Shape: {item.shape}, dtype: {item.dtype}")
        if item.shape and item.shape[0] < 10: #example to prevent printing huge datasets.
            try:
                print(f"  Sample data: {item[:]}")
            except TypeError:
                print(f"  Data cannot be printed.")

file.close()
