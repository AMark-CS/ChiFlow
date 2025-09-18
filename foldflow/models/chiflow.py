"""
ChiFlow model: Pure high-dimensional toroidal flow for protein backbone generation using dihedral angles.
Based on the ChiFlow paper using torsional asymmetry flow matching.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from .flows.common.layers import LeakyMLP
from .flows.encoders.residue import PerResidueEncoder
from .flows.encoders.pair import ResiduePairEncoder


class HighDimTorusFlow(nn.Module):
    """
    High-dimensional toroidal flow for dihedral angles (φ, ψ, ω).
    Adapted from ppflow's TorusFlow with enhancements for protein generation.
    """
    def __init__(self, n_dims=3, n_context_dims=256, cfg_flow=None):
        super().__init__()
        self.n_dims = n_dims

        # Default config if not provided
        if cfg_flow is None:
            cfg_flow = type('Config', (), {
                'num_hidden_dims': 512,
                'num_hidden_layers': 4,
                'num_mixtures': 3
            })()

        # Vector field network for each dihedral angle
        self.vector_field_nets = nn.ModuleList([
            LeakyMLP(
                dim_start=n_context_dims,
                dim_hidden=cfg_flow.num_hidden_dims,
                dim_end=3 * cfg_flow.num_mixtures,  # For mixture of von Mises
                num_layer=cfg_flow.num_hidden_layers,
            ) for _ in range(n_dims)
        ])

        self.num_mixtures = cfg_flow.num_mixtures

    def forward(self, dihedrals, context, t=None, mask=None):
        """
        Forward pass for high-dim toroidal flow.
        dihedrals: (B, L, 3) - φ, ψ, ω angles
        context: (B, L, C) - contextual features
        t: (B,) or (B, L) time
        mask: (B, L) mask for valid positions
        """
        B, L, D = dihedrals.shape
        assert D == self.n_dims, f"Expected {self.n_dims} dims, got {D}"

        # Flatten batch and sequence dimensions
        dihedrals_flat = dihedrals.view(B*L, D)  # (B*L, 3)
        context_flat = context.view(B*L, -1)     # (B*L, C)

        # Process each dihedral angle separately
        flows = []
        xts = []

        for d in range(self.n_dims):
            # Get vector field for this angle
            vf_params = self.vector_field_nets[d](context_flat)  # (B*L, 3*num_mixtures)
            vx_t = vf_params.view(B*L, 3, self.num_mixtures)     # (B*L, 3, num_mixtures)

            # Sample flow for this angle using simple Euler integration
            angle_dihedrals = dihedrals_flat[:, d]  # (B*L,)
            if t is None:
                t_tensor = torch.rand(B*L, device=dihedrals.device)
            else:
                t_tensor = t.view(B*L) if t.dim() > 1 else t.repeat(B*L)

            # Simple Euler step for torus flow
            dt = 0.01  # Small time step
            noise = torch.randn_like(angle_dihedrals) * 0.1
            ut = vx_t.mean(dim=-1)[:, 0]  # Take mean across mixtures, first component
            xt = angle_dihedrals + ut * dt + noise

            # Project to torus [-pi, pi]
            xt = torch.remainder(xt + torch.pi, 2 * torch.pi) - torch.pi

            flows.append(ut.unsqueeze(-1))
            xts.append(xt.unsqueeze(-1))

        # Stack flows and xts
        dihedral_flow = torch.cat(flows, dim=-1)  # (B*L, 3)
        dihedral_xt = torch.cat(xts, dim=-1)      # (B*L, 3)

        # Reshape back to (B, L, 3)
        dihedral_flow = dihedral_flow.view(B, L, D)
        dihedral_xt = dihedral_xt.view(B, L, D)

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(-1).expand_as(dihedral_flow)
            dihedral_flow = torch.where(mask, dihedral_flow, torch.zeros_like(dihedral_flow))
            dihedral_xt = torch.where(mask, dihedral_xt, dihedrals)

        return dihedral_flow, dihedral_xt


class ChiFlowMatcher(nn.Module):
    """
    ChiFlow matcher: Pure toroidal flow matching for protein dihedral angles.
    Implements the torsional asymmetry flow matching from the ChiFlow paper.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # High-dimensional toroidal flow
        self.torus_flow = HighDimTorusFlow(
            n_dims=3,  # φ, ψ, ω
            n_context_dims=cfg.n_context_dims,
            cfg_flow=cfg.flow
        )

        # Simple residue encoder
        self.aa_embedding = nn.Embedding(cfg.num_aa_types, cfg.residue_feat_dim)
        self.pos_encoder = nn.Linear(4, cfg.residue_feat_dim)  # position + mask
        self.res_mlp = nn.Sequential(
            nn.Linear(cfg.residue_feat_dim * 2, cfg.residue_feat_dim),
            nn.ReLU(),
            nn.Linear(cfg.residue_feat_dim, cfg.residue_feat_dim)
        )

        # Simple pair encoder
        self.dist_encoder = nn.Sequential(
            nn.Linear(1, cfg.pair_feat_dim),
            nn.ReLU(),
            nn.Linear(cfg.pair_feat_dim, cfg.pair_feat_dim)
        )

        # Context network
        self.context_net = nn.Sequential(
            nn.Linear(cfg.residue_feat_dim + cfg.pair_feat_dim, cfg.n_context_dims),
            nn.ReLU(),
            nn.LayerNorm(cfg.n_context_dims)
        )

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, cfg.time_embed_dim),
            nn.ReLU(),
            nn.Linear(cfg.time_embed_dim, cfg.n_context_dims)  # Match context dimension
        )

    def encode_context(self, batch):
        """
        Encode protein context for flow conditioning.
        """
        # Get residue features from amino acid types and atom positions
        res_feat = self.encode_residue_features(batch)

        # Get pair features from atom positions
        pair_feat = self.encode_pair_features(batch)

        # Combine residue and pair features
        pair_mean = pair_feat.mean(dim=2)  # (B, L, pair_feat_dim)
        combined = torch.cat([res_feat, pair_mean], dim=-1)  # (B, L, res_feat_dim + pair_feat_dim)

        context = self.context_net(combined)  # (B, L, n_context_dims)
        return context

    def encode_residue_features(self, batch):
        """
        Encode per-residue features from amino acid types and atom positions.
        """
        aatype = batch['aatype']  # (B, L)
        atom_positions = batch['atom_positions']  # (B, L, 37, 3)
        atom_mask = batch['atom_mask']  # (B, L, 37)

        # Simple amino acid embedding
        aa_embed = self.aa_embedding(aatype)  # (B, L, feat_dim)

        # Use CA position as residue position feature
        ca_positions = atom_positions[:, :, 1, :]  # CA is index 1 in atom37
        ca_mask = atom_mask[:, :, 1]

        # Simple position encoding (can be improved)
        pos_feat = torch.cat([ca_positions, ca_mask.unsqueeze(-1)], dim=-1)
        pos_encoded = self.pos_encoder(pos_feat)

        # Combine features
        res_feat = torch.cat([aa_embed, pos_encoded], dim=-1)
        res_feat = self.res_mlp(res_feat)

        return res_feat

    def encode_pair_features(self, batch):
        """
        Encode pairwise features between residues.
        """
        atom_positions = batch['atom_positions']  # (B, L, 37, 3)
        atom_mask = batch['atom_mask']  # (B, L, 37)

        # Use CA-CA distances as pair features
        ca_positions = atom_positions[:, :, 1, :]  # (B, L, 3)
        ca_mask = atom_mask[:, :, 1]  # (B, L)

        # Compute pairwise distances
        ca_dist = torch.cdist(ca_positions, ca_positions)  # (B, L, L)
        ca_mask_pair = ca_mask.unsqueeze(-1) * ca_mask.unsqueeze(-2)  # (B, L, L)

        # Simple distance embedding
        dist_feat = self.dist_encoder(ca_dist.unsqueeze(-1))  # (B, L, L, feat_dim)

        return dist_feat

    def forward(self, batch, t=None):
        """
        Forward pass of ChiFlow matcher.
        """
        dihedrals = batch['dihedrals']  # (B, L, 3)
        mask = batch.get('mask', None)  # (B, L)

        # Encode context
        context = self.encode_context(batch)

        # Add time embedding if provided
        if t is not None:
            if t.dim() == 1:
                t = t.unsqueeze(-1).unsqueeze(-1).expand(-1, dihedrals.shape[1], -1)  # (B, L, 1)
            time_feat = self.time_embed(t)
            context = context + time_feat

        # Get toroidal flow
        dihedral_flow, dihedral_xt = self.torus_flow(dihedrals, context, t, mask)

        return {
            'dihedral_flow': dihedral_flow,
            'dihedral_xt': dihedral_xt,
            'context': context
        }

    def sample(self, batch, num_steps=100):
        """
        Sample dihedral angles using the flow.
        """
        device = batch['dihedrals'].device
        B, L, D = batch['dihedrals'].shape

        # Start from random angles
        xt = torch.rand(B, L, D, device=device) * 2 * torch.pi - torch.pi

        # Reverse time steps
        ts = torch.linspace(1.0, 0.0, num_steps, device=device)

        for t in ts:
            t_batch = t.expand(B)
            flow_output = self.forward({**batch, 'dihedrals': xt}, t=t_batch)
            vx_t = flow_output['dihedral_flow']

            # Euler step
            dt = -1.0 / num_steps
            xt = xt + vx_t * dt

            # Project to torus
            xt = torch.remainder(xt + torch.pi, 2 * torch.pi) - torch.pi

        return xt


class ChiFlowModel(nn.Module):
    """
    Complete ChiFlow model for chiral-aware protein backbone generation.
    """
    def __init__(self, cfg):
        super().__init__()
        self.flow_matcher = ChiFlowMatcher(cfg)
        self.cfg = cfg

    def forward(self, batch, t=None):
        return self.flow_matcher.forward(batch, t)

    def sample(self, batch, num_steps=100):
        """
        Generate protein backbone using high-dimensional torus flow combined with NERF iterative generation.
        This method implements the complete inference pipeline:
        1. Sample dihedral angles using torus flow matching
        2. Iteratively reconstruct 3D coordinates using NERF
        """
        # Sample dihedrals using high-dimensional torus flow
        sampled_dihedrals = self.flow_matcher.sample(batch, num_steps)

        # Iteratively reconstruct backbone coordinates using NERF
        backbone_coords = self.reconstruct_backbone(sampled_dihedrals, batch)

        return {
            'dihedrals': sampled_dihedrals,
            'backbone_coords': backbone_coords
        }

    def reconstruct_backbone(self, dihedrals, batch):
        """
        Reconstruct 3D coordinates from dihedral angles using NERF iterative generation.
        Uses the project's NERF implementation for accurate backbone coordinate reconstruction.
        """
        from .flows.common.nerf import nerf_build_batch

        B, L, D = dihedrals.shape
        assert D == 3, "Expected 3 dihedral angles (phi, psi, omega)"

        # Split dihedrals into phi, psi, omega
        phi = dihedrals[:, :, 0]   # (B, L)
        psi = dihedrals[:, :, 1]   # (B, L)
        omega = dihedrals[:, :, 2] # (B, L)

        # Use NERF to build backbone coordinates
        # Returns (B, L*3, 3) where each residue has N, CA, C coordinates
        backbone_coords = nerf_build_batch(phi, psi, omega)

        # Reshape to (B, L, 3, 3) for N, CA, C atoms per residue
        backbone_coords = backbone_coords.view(B, L, 3, 3)

        return backbone_coords