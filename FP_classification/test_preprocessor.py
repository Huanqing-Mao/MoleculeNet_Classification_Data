from FP_classification.preprocessor import Preprocessor
import pandas as pd


path = r"C:\Users\huanqing\Desktop\ARIA_Documents\Dataset\HIV.csv"
raw_df = pd.read_csv(path)


#fp_keys = ["ecfp4", "ap", "tt", "rdkit", "maccs"]
fp_keys = ["ecfp4"]
mol_colname = "smiles"
p = Preprocessor(mol_colname, fp_keys)

results = p.get_all_fps(raw_df, "hiv")

print(results)