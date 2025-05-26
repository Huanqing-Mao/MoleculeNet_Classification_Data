from load_data import load_data
from preprocessor import Preprocessor
import os


# Source raw data
path = "./prediction_dataset"
new_directory = "./Processed_data"

# Pre-process data
raw_dta_lst, filenames = load_data(path)
print(filenames)
fp_keys = ["ecfp4", "ap", "tt", "rdkit", "maccs"]
mol_colname = "smiles"
class_colname = "Class"


'''
Note:
Class for toxcast: ATG_PPARg_TRANS_up
Class for tox21: SR-p53
Class for sider: Hepatobiliary disorders

The classes are randomly chosen.
'''

for i in range(len(raw_dta_lst)):
    raw_df = raw_dta_lst[i]
    filename = filenames[i]

    print(f"Processing {filename}...")
    # process data
    p = Preprocessor(mol_colname, class_colname, fp_keys)
    new_path = os.path.join(new_directory, f"processed_{filename}")
    results = p.get_all_fps(raw_df, new_path)
    print(f"{filename} processed and saved.")
    