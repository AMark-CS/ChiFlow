#!/usr/bin/env python3
"""
Command-line interface for ChiFlow inference using high-dimensional torus flow + NERF backbone generation.
"""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from foldflow.models.chiflow import ChiFlowModel
from foldflow.models.flows.common.nerf import nerf_build_batch
import foldflow.data.utils as du
from tools.analysis import utils as au


class ChiFlowInference:
    """ChiFlow inference interface for command-line usage."""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(f"cuda:{config.gpu_id}" if torch.cuda.is_available() and config.gpu_id >= 0 else "cpu")

        # Set random seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        print(f"Using device: {self.device}")
        print(f"Random seed: {config.seed}")

    def load_model(self):
        """Load ChiFlow model from checkpoint."""
        weights_path = self.config.weights_path
        if not weights_path or not os.path.exists(weights_path):
            print("⚠ No checkpoint provided or path is invalid, using random initialization.")
            # Fallback to default config for random initialization
            cfg = self._create_default_config()
            self.model = ChiFlowModel(cfg).to(self.device).eval()
            return

        print(f"Loading model from: {weights_path}")
        
        # Ensure path is absolute
        weights_path = os.path.abspath(weights_path)
        print(f"Attempting to load weights from absolute path: {weights_path}")

        # Load checkpoint
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)

        # Load config from checkpoint if available
        cfg = self._create_default_config()
        stochastic_paths = False  # Default value
        
        if 'conf' in checkpoint:
            print("✓ Model configuration loaded from checkpoint.")
            loaded_cfg = checkpoint['conf']
            # Merge loaded config into default config
            for key, value in loaded_cfg.items():
                setattr(cfg, key, value)
            
            # Safely get stochastic_paths if it exists
            if 'stochastic_paths' in loaded_cfg:
                stochastic_paths = loaded_cfg.stochastic_paths
        elif 'cfg' in checkpoint:
            print("✓ Model configuration loaded from checkpoint.")
            loaded_cfg = checkpoint['cfg']
            # Merge loaded config into default config
            for key, value in loaded_cfg.items():
                setattr(cfg, key, value)

            # Safely get stochastic_paths if it exists
            if 'stochastic_paths' in loaded_cfg:
                stochastic_paths = loaded_cfg.stochastic_paths
        else:
            print("⚠ No configuration found in checkpoint, using default inference config.")
            # cfg is already the default config

        # Create model with the loaded or default configuration
        self.model = ChiFlowModel(cfg, stochastic_paths=stochastic_paths)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load model state dict
        if 'model' in checkpoint:
            try:
                self.model.load_state_dict(checkpoint['model'], strict=False)
                print("✓ Model weights loaded successfully")
            except RuntimeError as e:
                print(f"❌ Error loading state_dict: {e}")
                print("Retrying with strict=False might help if some layers are missing.")
        else:
            print("⚠ No model weights ('model' key) found in checkpoint, using random initialization.")

    def _create_default_config(self):
        """Creates a default model configuration for inference."""
        class ModelConfig:
            def __init__(self):
                self.n_context_dims = 256
                self.residue_feat_dim = 256
                self.pair_feat_dim = 128
                self.num_aa_types = 21
                self.time_embed_dim = 64
                self.flow = type('FlowConfig', (), {
                    'num_hidden_dims': 512,
                    'num_hidden_layers': 4,
                    'num_mixtures': 3
                })()
                self.residue_encoder = type('ResidueEncoder', (), {
                    'num_aa_types': 21,
                    'feat_dim': 256,
                    'max_num_atoms': 37
                })()
                self.pair_encoder = type('PairEncoder', (), {
                    'feat_dim': 128,
                    'max_num_residues': 512
                })()
                self.mirror_constraint_weight = 0.1
                self.use_mirror_constraint = True
                self.ot_plan = False
                self.ot_fn = 'exact'
                self.reg = 0.05
                self.stochastic_paths = False
        return ModelConfig()

    def create_batch(self, sequence_length):
        """Create input batch for inference."""
        batch_size = 1  # Single sample for command line

        batch = {
            'dihedrals': torch.randn(batch_size, sequence_length, 3, device=self.device),  # φ, ψ, ω
            'aatype': torch.randint(0, 21, (batch_size, sequence_length), device=self.device),
            'atom_positions': torch.randn(batch_size, sequence_length, 37, 3, device=self.device),  # atom37
            'atom_mask': torch.ones(batch_size, sequence_length, 37, device=self.device),
            'res_mask': torch.ones(batch_size, sequence_length, device=self.device),
        }

        return batch

    def save_pdb(self, backbone_coords, output_path, sequence_name="chiflow_sample"):
        """Save backbone coordinates to PDB format."""
        # Convert to numpy
        coords_np = backbone_coords[0].cpu().numpy()  # (L, 3, 3) -> N, CA, C atoms

        # Create atom records
        pdb_lines = []
        atom_idx = 1

        for residue_idx in range(len(coords_np)):
            # N atom
            pdb_lines.append(self._create_pdb_atom_line(
                atom_idx, "N", "", "ALA", residue_idx + 1,
                coords_np[residue_idx, 0, 0], coords_np[residue_idx, 0, 1], coords_np[residue_idx, 0, 2]
            ))
            atom_idx += 1

            # CA atom
            pdb_lines.append(self._create_pdb_atom_line(
                atom_idx, "CA", "", "ALA", residue_idx + 1,
                coords_np[residue_idx, 1, 0], coords_np[residue_idx, 1, 1], coords_np[residue_idx, 1, 2]
            ))
            atom_idx += 1

            # C atom
            pdb_lines.append(self._create_pdb_atom_line(
                atom_idx, "C", "", "ALA", residue_idx + 1,
                coords_np[residue_idx, 2, 0], coords_np[residue_idx, 2, 1], coords_np[residue_idx, 2, 2]
            ))
            atom_idx += 1

        pdb_lines.append("END")

        # Write to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(pdb_lines))

        print(f"✓ PDB saved to: {output_path}")

    def _create_pdb_atom_line(self, atom_idx, atom_name, alt_loc, res_name, res_idx, x, y, z):
        """Create a PDB ATOM record line."""
        return ("ATOM  %5d %4s%1s%3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f           %2s%2s" %
                (atom_idx, atom_name, alt_loc, res_name, "A", res_idx, "",
                 x, y, z, 1.0, 0.0, atom_name[0], ""))

    def run_inference(self):
        """Run ChiFlow inference for multiple samples."""
        print("🚀 Starting ChiFlow inference...")
        print(f"Sequence length: {self.config.length}")
        print(f"Sampling steps: {self.config.num_steps}")
        print(f"Number of samples: {self.config.num_samples}")

        # Load model once
        self.load_model()

        # Create timestamped base output directory
        base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_run_dir = os.path.join(self.config.output_dir, f"{self.config.name}_batch_{base_timestamp}")
        os.makedirs(base_run_dir, exist_ok=True)
        print(f"📁 Created base output directory: {base_run_dir}")

        # Generate multiple samples
        for sample_idx in range(self.config.num_samples):
            print(f"\n🔄 Generating sample {sample_idx + 1}/{self.config.num_samples}...")

            # Set different seed for each sample to ensure diversity
            sample_seed = self.config.seed + sample_idx
            torch.manual_seed(sample_seed)
            np.random.seed(sample_seed)

            # Create input batch
            batch = self.create_batch(self.config.length)

            # Run sampling
            with torch.no_grad():
                sample_output = self.model.sample(batch, num_steps=self.config.num_steps)

            backbone_coords = sample_output['backbone_coords']
            dihedrals = sample_output['dihedrals']

            print(f"✓ Sample {sample_idx + 1} completed!")
            print(f"Generated backbone shape: {backbone_coords.shape}")
            print(f"Generated dihedrals shape: {dihedrals.shape}")

            # Create sample-specific output directory
            sample_name = f"{self.config.name}_{sample_idx + 1:03d}"
            sample_dir = os.path.join(base_run_dir, sample_name)
            os.makedirs(sample_dir, exist_ok=True)

            # Save PDB
            pdb_path = os.path.join(sample_dir, f"{sample_name}.pdb")
            self.save_pdb(backbone_coords, pdb_path)

            # Save dihedrals
            dihedrals_path = os.path.join(sample_dir, f"{sample_name}_dihedrals.npy")
            np.save(dihedrals_path, dihedrals[0].cpu().numpy())
            print(f"✓ Sample {sample_idx + 1} saved to: {sample_dir}")

        # Save batch configuration
        config_path = os.path.join(base_run_dir, f"{self.config.name}_batch_config.txt")
        with open(config_path, 'w') as f:
            f.write(f"ChiFlow Batch Inference Configuration\n")
            f.write(f"===================================\n")
            f.write(f"Sequence length: {self.config.length}\n")
            f.write(f"Sampling steps: {self.config.num_steps}\n")
            f.write(f"Number of samples: {self.config.num_samples}\n")
            f.write(f"Base random seed: {self.config.seed}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Base output directory: {base_run_dir}\n")
            f.write(f"Timestamp: {base_timestamp}\n")
        print(f"✓ Batch configuration saved to: {config_path}")

        print("🎉 ChiFlow batch inference completed successfully!")
        print(f"📁 All results saved in: {base_run_dir}")
        print(f"📊 Generated {self.config.num_samples} protein samples")


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description="ChiFlow: High-dimensional Torus Flow + NERF Backbone Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a single 100-residue protein backbone
  python chiflow_inference.py --length 100 --output_dir ./results

  # Generate 50 protein samples with custom sampling steps
  python chiflow_inference.py --length 150 --num_steps 100 --num_samples 50 --name batch_protein

  # Use specific GPU and weights
  python chiflow_inference.py --length 200 --gpu_id 1 --seed 42 --weights_path ./ckpt/model.pth
        """
    )

    parser.add_argument(
        "--length", "-l",
        type=int,
        default=100,
        help="Sequence length to generate (default: 100)"
    )

    parser.add_argument(
        "--num_steps", "-n",
        type=int,
        default=50,
        help="Number of sampling steps for torus flow (default: 50)"
    )

    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="./chiflow_results",
        help="Output directory for results (default: ./chiflow_results)"
    )

    parser.add_argument(
        "--name", "-N",
        type=str,
        default="chiflow_sample",
        help="Name prefix for output files (default: chiflow_sample)"
    )

    parser.add_argument(
        "--gpu_id", "-g",
        type=int,
        default=0,
        help="GPU device ID (-1 for CPU, default: 0)"
    )

    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    parser.add_argument(
        "--weights_path", "-w",
        type=str,
        default=None,
        help="Path to model checkpoint (optional)"
    )

    parser.add_argument(
        "--num_samples", "-ns",
        type=int,
        default=1,
        help="Number of protein samples to generate (default: 1)"
    )

    args = parser.parse_args()

    # Resolve weights_path to an absolute path to be independent of CWD
    if args.weights_path:
        args.weights_path = os.path.abspath(args.weights_path)

    # Validate arguments
    if args.length <= 0:
        parser.error("Sequence length must be positive")
    if args.num_steps <= 0:
        parser.error("Number of steps must be positive")
    if args.num_samples <= 0:
        parser.error("Number of samples must be positive")

    # Create config object
    config = args

    # Run inference
    try:
        inference = ChiFlowInference(config)
        inference.run_inference()
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()