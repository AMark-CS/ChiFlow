#!/usr/bin/env python3
"""
Load a PDB file, detect chiral centers using RDKit's CIP rules and print R/S assignments.

Usage: python analyze_chirality_rdkit.py /path/to/sample_1.pdb

This script tries to map RDKit atom indices back to PDB atom records by spatial lookup
when possible, and prints residue/atom names with CIP labels.
"""
import sys
from collections import defaultdict
import math

def read_pdb_atoms(pdb_path):
    atoms = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith(('ATOM','HETATM')):
                # PDB columns: atom serial(7-11), name(13-16), resName(18-20), chain(22), resSeq(23-26), x(31-38), y(39-46), z(47-54)
                name = line[12:16].strip()
                resname = line[17:20].strip()
                chain = line[21].strip()
                resseq = line[22:26].strip()
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue
                atoms.append({
                    'name': name,
                    'resname': resname,
                    'chain': chain,
                    'resseq': resseq,
                    'coord': (x,y,z),
                    'line': line.rstrip('\n'),
                })
    return atoms

def dist(a,b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def main():
    if len(sys.argv) < 2:
        print('Usage: python analyze_chirality_rdkit.py /path/to/sample_1.pdb')
        sys.exit(1)
    pdb_path = sys.argv[1]

    # Lazy import RDKit and handle if missing
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except Exception as e:
        print('RDKit import failed:', e)
        print('Install rdkit (e.g. pip install rdkit-pypi)')
        sys.exit(2)

    atoms = read_pdb_atoms(pdb_path)
    if not atoms:
        print('No ATOM/HETATM records found in', pdb_path)
        sys.exit(3)

    # Read molecule using RDKit's MolFromPDBFile (PDBMolSupplier may not be available in some builds)
    mol = None
    try:
        mol = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=False)
    except Exception:
        mol = None
    if mol is None:
        # try reading as a block
        try:
            with open(pdb_path, 'r') as fh:
                pdb_block = fh.read()
            mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=False)
        except Exception:
            mol = None
    if mol is None:
        print('RDKit failed to read molecule from PDB. Consider installing a different RDKit build or convert PDB to another format.')
        sys.exit(4)

    # Try sanitization but allow partial
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        # attempt to sanitize partially
        try:
            Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        except Exception:
            pass

    # Add explicit Hs (helps RDKit determine stereochemistry). Guard with try/except
    try:
        mol = Chem.AddHs(mol, addCoords=True)
    except Exception as e:
        print('Warning: AddHs failed:', e)

    # Ensure we have 3D coordinates; embed if necessary
    conf = None
    try:
        conf = mol.GetConformer()
    except Exception:
        try:
            print('No conformer found; embedding a 3D conformation (may take a moment)')
            AllChem.EmbedMolecule(mol, randomSeed=42)
            conf = mol.GetConformer()
        except Exception as e:
            print('Warning: embedding failed:', e)

    # Assign stereochemistry (try/except since RDKit builds vary)
    try:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    except Exception as e:
        print('Warning: AssignStereochemistry failed:', e)

    # Build a simple spatial index to map RDKit atoms to PDB atom records
    pdb_coords = [a['coord'] for a in atoms]
    mapping = {}
    conf = mol.GetConformer()
    for a in mol.GetAtoms():
        idx = a.GetIdx()
        pos = conf.GetAtomPosition(idx)
        atom_coord = (pos.x, pos.y, pos.z)
        # find nearest pdb atom within 0.8 A
        best_i = None
        best_d = 1e9
        for i,c in enumerate(pdb_coords):
            d = dist(atom_coord, c)
            if d < best_d:
                best_d = d
                best_i = i
        if best_d < 0.9:
            mapping[idx] = best_i

    # Get chiral centers: (atomIdx, (neighbors...), chirality)
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)

    by_res = defaultdict(list)
    print('Found', len(centers), 'chiral center candidates (including unassigned).')
    r_count = 0
    s_count = 0
    unassigned_count = 0
    for center in centers:
        atom_idx = center[0]
        chirality = center[1]
        atom = mol.GetAtomWithIdx(atom_idx)
        symbol = atom.GetSymbol()
        # Map to PDB if possible
        pdb_i = mapping.get(atom_idx, None)
        pdb_info = None
        if pdb_i is not None:
            pdb_info = atoms[pdb_i]
            idstr = f"{pdb_info['resname']} {pdb_info['chain']}{pdb_info['resseq']}:{pdb_info['name']}"
        else:
            idstr = f"RDKit_idx_{atom_idx}_{symbol}"

        print(f"Atom: {idstr}  RDKit_idx={atom_idx}  symbol={symbol}  CIP={chirality}")
        by_res[pdb_info['resseq'] if pdb_info else 'NA'].append((idstr, atom_idx, chirality))
        if chirality == 'R':
            r_count += 1
        elif chirality == 'S':
            s_count += 1
        else:
            unassigned_count += 1

    # Sequence length (estimate from highest residue number in PDB)
    try:
        resnums = sorted({int(a['resseq']) for a in atoms if a['resseq'].isdigit()})
        seq_len = max(resnums) - min(resnums) + 1 if resnums else 0
    except Exception:
        seq_len = 0

    total_centers = r_count + s_count + unassigned_count
    print('\nChirality counts:')
    print(f'  R: {r_count}')
    print(f'  S: {s_count}')
    print(f'  unassigned/unknown: {unassigned_count}')
    print(f'  total centers: {total_centers}')
    if seq_len > 0:
        print(f'Sequence length (residues): {seq_len}')
        pct_r = 100.0 * r_count / seq_len
        pct_s = 100.0 * s_count / seq_len
        pct_un = 100.0 * unassigned_count / seq_len
        print('\nPercentages (relative to sequence length):')
        print(f'  R: {pct_r:.2f}%')
        print(f'  S: {pct_s:.2f}%')
        print(f'  unassigned: {pct_un:.2f}%')
    else:
        print('Sequence length could not be determined; cannot compute percent relative to length.')

if __name__ == '__main__':
    main()
