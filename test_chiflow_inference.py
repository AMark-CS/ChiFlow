#!/usr/bin/env python3
"""
Test script for ChiFlow model with migrated components.
"""

import torch
import sys
import os

# Add the project root to Python path
sys.path.append('/root/autodl-fs/ChiFlow')

from foldflow.models.chiflow import ChiFlowModel

def test_chiflow():
    """Test ChiFlow model instantiation and basic functionality."""

    # Create a simple config
    class Config:
        def __init__(self):
            self.n_context_dims = 256
            self.residue_feat_dim = 128
            self.pair_feat_dim = 64
            self.num_aa_types = 21
            self.time_embed_dim = 64
            self.flow = type('FlowConfig', (), {
                'num_hidden_dims': 512,
                'num_hidden_layers': 4,
                'num_mixtures': 3
            })()

    cfg = Config()

    # Create model
    model = ChiFlowModel(cfg)
    print("✓ ChiFlow model created successfully")

    # Create test batch
    batch_size = 2
    seq_length = 10

    batch = {
        'dihedrals': torch.randn(batch_size, seq_length, 3),  # φ, ψ, ω
        'aatype': torch.randint(0, 21, (batch_size, seq_length)),
        'atom_positions': torch.randn(batch_size, seq_length, 37, 3),  # atom37
        'atom_mask': torch.ones(batch_size, seq_length, 37),
        'res_mask': torch.ones(batch_size, seq_length),
    }

    # Test forward pass
    output = model(batch)
    print(f"✓ Forward pass successful, output shapes: { {k: v.shape for k, v in output.items()} }")

    # Test sampling
    sampled = model.sample(batch, num_steps=10)
    print(f"✓ Sampling successful, sampled shapes: { {k: v.shape for k, v in sampled.items()} }")

    print("All tests passed! ChiFlow is working correctly.")

if __name__ == "__main__":
    test_chiflow()