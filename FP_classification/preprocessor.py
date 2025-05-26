import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys

class Preprocessor:
    def __init__(self, mol_colname, class_colname, fp_keys):
        self.fp = {
            "ecfp4": lambda mol: AllChem.GetMorganGenerator(radius=2).GetFingerprint(mol),
            "ap": lambda mol: AllChem.GetAtomPairGenerator().GetFingerprint(mol),
            "tt": lambda mol: AllChem.GetTopologicalTorsionGenerator().GetFingerprint(mol),
            "rdkit": lambda mol: AllChem.GetRDKitFPGenerator().GetFingerprint(mol),
            "maccs": lambda mol: MACCSkeys.GenMACCSKeys(mol)
        } # add all fingerprints : function in RDKit
        self.mol_colname = mol_colname
        self.class_colname = class_colname
        self.fp_keys = fp_keys


    def smiles_to_fp(self, smiles, fp_key):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or fp_key not in self.fp:
            print("invalid mol detected")
            return None  # skip invalid smiles
        try:
            fp = self.fp[fp_key](mol)
            arr = np.zeros((fp.GetNumBits(),), dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            #print(arr.tolist())
            return arr.tolist()
        except:
            return None  # skip any fingerprinting error

    def get_df_with_fp(self, data, mol_colname, fp_key, drop_invalid=True):
        df = data.copy()
        df[fp_key] = df[mol_colname].apply(lambda x: self.smiles_to_fp(x, fp_key))
        if drop_invalid:
            df = df[df[fp_key].notnull()].reset_index(drop=True)
            df = df[df[self.class_colname].notnull()].reset_index(drop=True)
        return df
    
    def get_all_fps(self, raw_df, new_path):
        data = raw_df.copy()
        for fp in self.fp_keys:
            results = self.get_df_with_fp(data, mol_colname=self.mol_colname, fp_key=fp)
            data = results
            
        data.to_csv(new_path)
        print(data)
        print(data.columns[:])