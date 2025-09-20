import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

def calc_dihedral(p0, p1, p2, p3):
    """
    Calculate the dihedral angle given four points.
    The angle is calculated in degrees.
    """
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b1 /= np.linalg.norm(b1)

    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)

    return np.degrees(np.arctan2(y, x))

def calculate_ramachandran_from_pkl(pkl_file):
    """
    Calculate phi and psi angles from a .pkl file containing atomic coordinates.
    This function assumes the pkl file contains a dictionary with a 'coords' key,
    which holds another dictionary of atomic coordinates like {'N': array, 'CA': array, 'C': array}.
    """
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    # --- This part may need adjustment based on the actual pkl structure ---
    # Assuming the coordinates are stored in a dictionary under the key 'coords'
    # and each atom type has its own array of coordinates.
    if 'coords' not in data:
        # Try another common key, e.g., 'atom_positions'
        if 'atom_positions' in data:
             coords = data['atom_positions']
        # If you know the exact key, replace it here.
        # For now, let's assume the data itself is the coordinate dictionary.
        else:
             coords = data
    else:
        coords = data['coords']

    n_coords = coords['N']
    ca_coords = coords['CA']
    c_coords = coords['C']
    # --- End of adjustable part ---

    num_residues = len(n_coords)
    if num_residues < 2:
        return [], []

    phi_angles = []
    psi_angles = []

    for i in range(1, num_residues):
        # Phi angle: C(i-1) - N(i) - CA(i) - C(i)
        if i > 0:
            p0 = c_coords[i-1]
            p1 = n_coords[i]
            p2 = ca_coords[i]
            p3 = c_coords[i]
            phi = calc_dihedral(p0, p1, p2, p3)
            phi_angles.append(phi)

    for i in range(num_residues - 1):
        # Psi angle: N(i) - CA(i) - C(i) - N(i+1)
        if i < num_residues - 1:
            p0 = n_coords[i]
            p1 = ca_coords[i]
            p2 = c_coords[i]
            p3 = n_coords[i+1]
            psi = calc_dihedral(p0, p1, p2, p3)
            psi_angles.append(psi)
            
    # The first residue has no phi, and the last has no psi.
    # To make them the same length for plotting, we only return pairs.
    # So we calculate phi for residues 1 to n-1, and psi for 0 to n-2.
    # This gives us n-1 phi and n-1 psi values.
    return phi_angles, psi_angles


def plot_ramachandran(all_phi_angles, all_psi_angles, title, output_path):
    """
    Plot Ramachandran plot for all given phi and psi angles.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(all_phi_angles, all_psi_angles, alpha=0.3, s=1, color='green')
    
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.grid(True, alpha=0.3)
    
    ax.set_xlabel('Phi (degrees)')
    ax.set_ylabel('Psi (degrees)')
    ax.set_title(title)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Ramachandran plot saved to {output_path}")
    plt.close()

def main():
    pkl_dir = '/root/autodl-fs/ChiFlow/data/processed_pdb'
    output_dir = '/root/autodl-fs/ChiFlow/ramachandran_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    all_phi_angles = []
    all_psi_angles = []
    
    pkl_file_paths = []
    for root, dirs, files in os.walk(pkl_dir):
        for file in files:
            if file.endswith('.pkl'):
                pkl_file_paths.append(os.path.join(root, file))

    # Limit to the first 100 files found
    files_to_process = pkl_file_paths[:100]
    print(f"Found {len(pkl_file_paths)} .pkl files. Processing the first {len(files_to_process)}.")

    for pkl_file_path in files_to_process:
        print(f'Processing {os.path.basename(pkl_file_path)}...')
        try:
            phi_angles, psi_angles = calculate_ramachandran_from_pkl(pkl_file_path)
            all_phi_angles.extend(phi_angles)
            all_psi_angles.extend(psi_angles)
        except Exception as e:
            print(f"Could not process {os.path.basename(pkl_file_path)}: {e}")

    if all_phi_angles:
        output_path = os.path.join(output_dir, 'ramachandran_plot_from_pkl.png')
        plot_ramachandran(all_phi_angles, all_psi_angles, 'Ramachandran Plot from .pkl files', output_path)
    else:
        print('No angles were calculated. No .pkl files found or processed correctly.')

if __name__ == '__main__':
    main()
