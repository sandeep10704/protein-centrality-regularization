"""
Script to download representative authentic enzyme PDB files from RCSB PDB for Track B evaluation.
"""

import os
import urllib.request

PDB_IDS = ["1TIM", "1HEW", "2ACT", "3DFR", "1UBQ", "1L2Y"]
OUTPUT_DIR = "data/raw/pdb"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for pdb_id in PDB_IDS:
    dest_path = os.path.join(OUTPUT_DIR, f"{pdb_id}.pdb")
    if not os.path.exists(dest_path):
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        print(f"Downloading {pdb_id} from {url}...")
        try:
            urllib.request.urlretrieve(url, dest_path)
            print(f"Saved {dest_path} ({os.path.getsize(dest_path)} bytes)")
        except Exception as e:
            print(f"Failed to download {pdb_id}: {e}")
    else:
        print(f"Already exists: {dest_path}")
