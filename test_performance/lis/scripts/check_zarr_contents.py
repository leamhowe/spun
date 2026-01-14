import os
import sys

# The exact file that crashed
target_path = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/s2_training_scf/20210402_NN5270.zarr'

print(f"--- DIAGNOSTIC REPORT ---")
print(f"Target: {target_path}")

# 1. Does the directory exist?
exists = os.path.exists(target_path)
print(f"Exists: {exists}")

if exists:
    # 2. Is it actually a directory?
    is_dir = os.path.isdir(target_path)
    print(f"Is Directory: {is_dir}")
    
    # 3. What is inside? (The moment of truth)
    try:
        contents = os.listdir(target_path)
        print(f"Contents ({len(contents)} items): {contents}")
        
        if '.zgroup' in contents:
            print("✅ .zgroup found (It is a Group)")
        elif '.zarray' in contents:
            print("✅ .zarray found (It is an Array)")
        else:
            print("❌ CRITICAL: No Zarr metadata found! This is an empty/corrupt folder.")
            
    except PermissionError:
        print("❌ CRITICAL: Permission Denied! Script cannot read this folder.")
else:
    print("❌ Critical: Path not found.")