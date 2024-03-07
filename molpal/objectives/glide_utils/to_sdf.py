import pandas as pd
from rdkit.Chem import AllChem as Chem
import os, sys


def to_sdf(name, util_dir):
    mergedSDF_OUT = Chem.SDWriter(os.path.join(util_dir, 'input2D.sdf'))
    df = pd.read_csv(name, sep=' ')
    df.columns=['SMILES','ID']
    for index, row in df.iterrows():
        try:
            mol = Chem.MolFromSmiles(row['SMILES'])
            mol.SetProp("_Name", str(row['ID']))
            mergedSDF_OUT.write(mol)
        except:
            print('Not able to read compound or get SMILES or ID')
    mergedSDF_OUT.close()

if __name__ == '__main__':
    name = sys.argv[1]
    util_dir = sys.argv[2]
    to_sdf(name, util_dir)