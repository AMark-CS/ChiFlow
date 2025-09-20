#!/usr/bin/env python3
"""
Test script for ChiFlow OT and SFM functionality.
"""

import torch
import numpy as np
from omegaconf import OmegaConf
from foldflow.models.chiflow import ChiFlowModel, ChiFlowMatcher

def test_chiflow_ot_sfm():
    """Test ChiFlow with OT and SFM enabled."""

    # Create config with OT and SFM enabled
    config = {
        'model_name': 'chiflow',
        'n_context_dims': 256,
        'residue_feat_dim': 128,
        'pair_feat_dim': 64,
        'num_aa_types': 21,
        'time_embed_dim': 64,
        'ot_plan': True,
        'ot_fn': 'exact',
        'reg': 0.05,
        'stochastic_paths': True,
        'flow': {
            'num_hidden_dims': 512,
            'num_hidden_layers': 4,
            'num_mixtures': 3,
            'stochastic_paths': True,
            'g': 0.1,
            'min_sigma': 0.01
        }
    }

    # Convert to object
    class Config:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                if isinstance(v, dict):
                    setattr(self, k, Config(**v))
                else:
                    setattr(self, k, v)

    cfg = Config(**config)

    print("Testing ChiFlow with OT and SFM...")
    print(f"OT enabled: {cfg.ot_plan}")
    print(f"SFM enabled: {cfg.stochastic_paths}")

    # Create model
    try:
        model = ChiFlowModel(cfg)
        matcher = ChiFlowMatcher(cfg)
        print("✓ ChiFlow model and matcher created successfully")
    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        return False

    # Test basic functionality
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    matcher = matcher.to(device)

    # Create test batch
    batch_size, seq_len = 2, 10
    batch = {
        'dihedrals': torch.randn(batch_size, seq_len, 3, device=device),
        'aatype': torch.randint(0, 21, (batch_size, seq_len), device=device),
        'atom_positions': torch.randn(batch_size, seq_len, 37, 3, device=device),
        'atom_mask': torch.ones(batch_size, seq_len, 37, device=device),
        'res_mask': torch.ones(batch_size, seq_len, device=device),
    }

    try:
        # Test forward pass
        with torch.no_grad():
            output = model(batch)
            print("✓ Model forward pass successful")
            print(f"  Output keys: {list(output.keys())}")

        # Test matcher forward pass
        with torch.no_grad():
            matcher_output = matcher(batch)
            print("✓ Matcher forward pass successful")
            print(f"  Output keys: {list(matcher_output.keys())}")

        # Test sampling
        with torch.no_grad():
            sample_output = model.sample(batch, num_steps=10)
            print("✓ Sampling successful")
            print(f"  Sample shape: {sample_output['dihedrals'].shape}")

        print("✓ All tests passed!")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_chiflow_ot_sfm()
    if success:
        print("\n🎉 ChiFlow OT and SFM functionality test completed successfully!")
    else:
        print("\n❌ ChiFlow OT and SFM functionality test failed!")