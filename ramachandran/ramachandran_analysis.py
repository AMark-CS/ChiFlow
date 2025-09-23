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
    Calculate phi and psi angles from a PDB file using Bio.PDB's PPBuilder.
    Returns lists of phi and psi angles in degrees.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    phi_angles = []
    psi_angles = []

    for model in structure:
        for chain in model:
            polypeptides = PPBuilder().build_peptides(chain)
            for poly_index, poly in enumerate(polypeptides):
                phi_psi = poly.get_phi_psi_list()
                for res_index, (phi, psi) in enumerate(phi_psi):
                    if phi is not None and psi is not None:
                        phi_angles.append(np.degrees(phi))
                        psi_angles.append(np.degrees(psi))
    
    return phi_angles, psi_angles

def plot_ramachandran(all_phi_angles, all_psi_angles, title, output_path):
    """
    Plot Ramachandran plot for all given phi and psi angles using a 2D histogram
    and outlining allowed regions.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Use a 2D histogram for density
    h = ax.hist2d(all_phi_angles, all_psi_angles, bins=180, cmap='viridis', cmin=1)
    fig.colorbar(h[3], ax=ax, label='Density')

    # Outline of allowed regions (approximate)
    # Beta-sheet region
    ax.add_patch(plt.Rectangle((-180, 110), 90, 70, fill=False, edgecolor='red', linewidth=1, linestyle='--'))
    ax.add_patch(plt.Rectangle((-150, -180), 100, 90, fill=False, edgecolor='red', linewidth=1, linestyle='--'))
    # Alpha-helix region (right-handed)
    ax.add_patch(plt.Rectangle((-150, -70), 100, 50, fill=False, edgecolor='red', linewidth=1, linestyle='--'))
    # Left-handed helix region
    ax.add_patch(plt.Rectangle((40, 40), 50, 50, fill=False, edgecolor='red', linewidth=1, linestyle='--'))

    # Set limits and ticks
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-180, 181, 60))
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Labels and title
    ax.set_xlabel('Phi (φ) degrees')
    ax.set_ylabel('Psi (ψ) degrees')
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

import argparse
from datetime import datetime

# ... (existing code for calculate_ramachandran_angles and plot_ramachandran) ...

def main():
    parser = argparse.ArgumentParser(description="Generate a combined Ramachandran plot from PDB files in a directory.")
    parser.add_argument(
        "--input_dir", "-i",
        type=str,
        required=True,
        help="Directory containing subdirectories with PDB files."
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default='/root/autodl-fs/ChiFlow/ramachandran_plots',
        help="Output directory for the plot."
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    all_phi_angles = []
    all_psi_angles = []
    pdb_files_found = 0

    print(f"🔍 Searching for PDB files in: {input_dir}")
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".pdb"):
                pdb_file_path = os.path.join(root, file)
                pdb_files_found += 1
                print(f"  Processing {pdb_file_path}...")
                try:
                    phi, psi = calculate_ramachandran_angles(pdb_file_path)
                    if phi:
                        all_phi_angles.extend(phi)
                        all_psi_angles.extend(psi)
                    else:
                        print(f"    ⚠️ No angles found in {file}")
                except Exception as e:
                    print(f"    ❌ Could not process {pdb_file_path}: {e}")

    if all_phi_angles:
        print(f"\n✅ Processed {pdb_files_found} PDB files.")
        dir_name = os.path.basename(os.path.normpath(input_dir))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f'combined_ramachandran_{dir_name}_{timestamp}.png'
        output_path = os.path.join(output_dir, output_filename)
        
        title = f'Combined Ramachandran Plot for {pdb_files_found} Proteins\n(from {dir_name})'
        plot_ramachandran(all_phi_angles, all_psi_angles, title, output_path)
        print(f"🎉 Combined plot saved to: {output_path}")
    else:
        print("No angles were calculated from any PDB files. No plot generated.")

if __name__ == '__main__':
    main()
