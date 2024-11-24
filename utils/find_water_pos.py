import numpy as np
import warnings
from Bio.PDB import PDBParser
from openbabel import openbabel as ob

def find_real_water_pos(file_path, model_index=0):
    file_extension = file_path.split('.')[-1].lower()
    
    if file_extension == 'pdb':
        warnings.simplefilter('ignore') 
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('PDB_structure', file_path)
        water_positions = []
        
        first_model = next(structure.get_models())
        for chain in first_model:
            for residue in chain:
                for atom in residue:
                    if atom.element == 'O':
                        water_positions.append(atom.coord)
        
                        
    elif file_extension == 'mol2':
        obConversion = ob.OBConversion()
        obConversion.SetInFormat("mol2")
        mol = ob.OBMol()
        obConversion.ReadFile(mol, file_path)
        water_positions = []
        for atom in ob.OBMolAtomIter(mol):
            if atom.GetType() == 'O':
                water_positions.append(np.array([atom.GetX(), atom.GetY(), atom.GetZ()]))

    else:
        raise ValueError("Unsupported file format. Please provide a PDB or MOL2 file.")
        
    return np.array(water_positions)