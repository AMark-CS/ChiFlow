import os
import numpy as np
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser, PPBuilder
from Bio.PDB.vectors import calc_dihedral
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser
from Bio.PDB.vectors import calc_dihedral
import warnings
warnings.filterwarnings("ignore")

def calculate_ramachandran_angles(pdb_file):
    """
    Calculate phi and psi angles from a PDB file manually.
    This is for PDB files that might be missing atoms like 'O'.
    Returns lists of phi and psi angles in degrees.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    phi_angles = []
    psi_angles = []
    
    for model in structure:
        for chain in model:
            residues = list(chain.get_residues())
            if len(residues) < 3:
                continue

            for i in range(1, len(residues) - 1):
                # phi: C(i-1) - N(i) - CA(i) - C(i)
                try:
                    c_prev = residues[i-1]['C'].get_vector()
                    n_curr = residues[i]['N'].get_vector()
                    ca_curr = residues[i]['CA'].get_vector()
                    c_curr = residues[i]['C'].get_vector()
                    
                    phi = calc_dihedral(c_prev, n_curr, ca_curr, c_curr)
                    phi_angles.append(np.degrees(phi))
                except KeyError:
                    # Atom not found, skip phi calculation for this residue
                    pass

                # psi: N(i) - CA(i) - C(i) - N(i+1)
                try:
                    n_curr = residues[i]['N'].get_vector()
                    ca_curr = residues[i]['CA'].get_vector()
                    c_curr = residues[i]['C'].get_vector()
                    n_next = residues[i+1]['N'].get_vector()

                    psi = calc_dihedral(n_curr, ca_curr, c_curr, n_next)
                    psi_angles.append(np.degrees(psi))
                except KeyError:
                    # Atom not found, skip psi calculation for this residue
                    pass

    return phi_angles, psi_angles

def plot_ramachandran(all_phi_angles, all_psi_angles, title, output_path):
    """
    Plot Ramachandran plot for all given phi and psi angles.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the points
    ax.scatter(all_phi_angles, all_psi_angles, alpha=0.3, s=1, color='blue')
    
    # Set limits
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Labels
    ax.set_xlabel('Phi (degrees)')
    ax.set_ylabel('Psi (degrees)')
    ax.set_title(title)
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Directory containing the protein subdirectories
    base_dir = '/root/autodl-fs/ChiFlow/eval_outputs/chiflow/default/step_3200'
    
    # Output directory for plots
    output_dir = '/root/autodl-fs/ChiFlow/ramachandran_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    all_phi_angles = []
    all_psi_angles = []
    
    pdb_files = [f for f in os.listdir(base_dir) if f.endswith('.pdb')]

    for pdb_file_name in pdb_files:
        pdb_file_path = os.path.join(base_dir, pdb_file_name)
        print(f'Processing {pdb_file_name}...')
        try:
            phi_angles, psi_angles = calculate_ramachandran_angles(pdb_file_path)
            all_phi_angles.extend(phi_angles)
            all_psi_angles.extend(psi_angles)
        except Exception as e:
            print(f"Could not process {pdb_file_name}: {e}")

    if all_phi_angles:
        output_path = os.path.join(output_dir, 'all_proteins_ramachandran.png')
        plot_ramachandran(all_phi_angles, all_psi_angles, 'Ramachandran Plot for All Proteins', output_path)
        print(f'Saved combined plot to {output_path}')
    else:
        print('No angles calculated for any protein')

if __name__ == '__main__':
    main()