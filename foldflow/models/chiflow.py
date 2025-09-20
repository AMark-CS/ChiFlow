"""
ChiFlow model: Pure high-dimensional toroidal flow for protein backbone generation using dihedral angles.
Based on the ChiFlow paper using torsional asymmetry flow matching.
"""

import torch
import torch.nn as nn
import math
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

        # Add stochastic paths support
        self.stochastic_paths = getattr(cfg_flow, 'stochastic_paths', False)
        self.g = getattr(cfg_flow, 'g', 0.1)
        self.min_sigma = getattr(cfg_flow, 'min_sigma', 0.01)

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

    def compute_sigma_t(self, t):
        """Compute noise scale for stochastic paths."""
        if isinstance(t, float):
            t = torch.tensor(t)
        return torch.sqrt(self.g**2 * t * (1 - t) + self.min_sigma**2)

    def forward(self, dihedrals, context, t=None, mask=None):
        """Pure vector-field prediction on torus (no internal Euler step).

        Returns predicted instantaneous velocity (vector field) in angle space.
        dihedrals: (B, L, 3)
        context: (B, L, C)
        t: optional time (B,) or (B,1,1) used only for potential conditioning in the future.
        mask: (B, L)
        """
        B, L, D = dihedrals.shape
        assert D == self.n_dims
        context_flat = context.reshape(B * L, -1)
        flows = []
        for d in range(self.n_dims):
            vf_params = self.vector_field_nets[d](context_flat)  # (B*L, 3*num_mixtures)
            vf_params = vf_params.view(B * L, 3, self.num_mixtures)
            # Use mean over mixture components first channel as velocity (simple reduction)
            ut = vf_params.mean(dim=-1)[:, 0]  # (B*L,)
            flows.append(ut.view(B, L, 1))
        dihedral_flow = torch.cat(flows, dim=-1)  # (B,L,3)
        if mask is not None:
            dihedral_flow = dihedral_flow * mask.unsqueeze(-1)
        # For backward compatibility we return dihedral_xt as the current dihedrals (identity)
        return dihedral_flow, dihedrals


# --------- Angle utility functions for path-based supervision ---------- #
def angle_wrap(x: torch.Tensor) -> torch.Tensor:
    return (x + math.pi) % (2 * math.pi) - math.pi


def angle_diff(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    return angle_wrap(target - source)


def linear_torus_path(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Analytic linear interpolation on torus per angle with wrapping.

    x0,x1: (...,3) starting and ending dihedral angles in [-pi,pi]
    t: broadcastable scalar or (...,1,1) in [0,1]
    Returns x_t with wrap to [-pi,pi].
    """
    delta = angle_diff(x1, x0)
    x_t = angle_wrap(x0 + delta * t)
    return x_t


class ChiFlowMatcher(nn.Module):
    """
    ChiFlow matcher: Pure toroidal flow matching for protein dihedral angles.
    Implements the torsional asymmetry flow matching from the ChiFlow paper.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Add OT and SFM support
        self.ot_plan = getattr(cfg, 'ot_plan', False)
        self.ot_fn = getattr(cfg, 'ot_fn', 'exact')
        self.reg = getattr(cfg, 'reg', 0.05)
        self.stochastic_paths = getattr(cfg, 'stochastic_paths', False)

        # High-dimensional toroidal flow
        self.torus_flow = HighDimTorusFlow(
            n_dims=3,  # φ, ψ, ω
            n_context_dims=cfg.n_context_dims,
            cfg_flow=cfg.flow
        )

        # Simple residue encoder
        self.aa_embedding = nn.Embedding(cfg.residue_encoder.num_aa_types, cfg.residue_feat_dim)
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

        # Log configuration
        if self.ot_plan:
            print(f"Using OT plan with {self.ot_fn} computation.")
        if self.stochastic_paths:
            print("Using stochastic paths.")

        # Mirror operator for chirality constraints
        self.mirror_constraint_weight = getattr(cfg, 'mirror_constraint_weight', 0.1)
        self.use_mirror_constraint = getattr(cfg, 'use_mirror_constraint', True)

    def dihedral_forward_marginal(self, dihedrals_0, t, flow_mask=None, dihedrals_1=None):
        """
        Forward marginal for ChiFlow with dihedral angles.
        dihedrals_0: (B, L, 3) ground truth dihedral angles
        t: time step (scalar)
        flow_mask: (B, L) mask for flowed positions
        dihedrals_1: (B, L, 3) noise dihedral angles (optional)
        """
        # Create batch with dihedral angles
        batch = {
            'dihedrals': dihedrals_0,
            'aatype': torch.zeros(dihedrals_0.shape[0], dihedrals_0.shape[1], dtype=torch.long, device=dihedrals_0.device),
            'atom_positions': torch.zeros(dihedrals_0.shape[0], dihedrals_0.shape[1], 37, 3, device=dihedrals_0.device),
            'atom_mask': torch.ones(dihedrals_0.shape[0], dihedrals_0.shape[1], 37, device=dihedrals_0.device),
            'res_mask': torch.ones(dihedrals_0.shape[0], dihedrals_0.shape[1], device=dihedrals_0.device),
        }

        # Add time to batch
        t_tensor = torch.tensor([t], device=dihedrals_0.device).expand(dihedrals_0.shape[0])

        # Forward pass (disable mirror loss to avoid recursion)
        flow_output = self.forward(batch, t=t_tensor, compute_mirror_loss=False)

        return {
            'dihedrals_t': flow_output['dihedral_xt'],
            'dihedral_vectorfield': flow_output['dihedral_flow']
        }

    def forward_marginal(self, rigids_0, t, flow_mask=None, rigids_1=None):
        """
        Forward marginal for ChiFlow with dihedral angles.
        This method handles both rigid-based (for compatibility) and dihedral-based processing.
        """
        # Check if we have dihedral data in the input
        if hasattr(rigids_0, 'dihedrals') or (isinstance(rigids_0, dict) and 'dihedrals' in rigids_0):
            # Direct dihedral processing
            if isinstance(rigids_0, dict):
                dihedrals_0 = rigids_0['dihedrals']
            else:
                dihedrals_0 = rigids_0.dihedrals

            return self.dihedral_forward_marginal(dihedrals_0, t, flow_mask, rigids_1)
        else:
            # For compatibility with existing OT code that uses rigids
            # We'll need to convert rigids to dihedrals or implement a different approach
            # For now, raise a more informative error
            raise NotImplementedError(
                "ChiFlow forward_marginal with rigids requires dihedral angle data. "
                "Please ensure dihedral angles are available in the input data, "
                "or implement rigid-to-dihedral conversion."
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

        # Ensure aatype is on the same device as the embedding weights
        aatype = aatype.to(self.aa_embedding.weight.device)

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

    def _compute_mirror_constraint_loss(self, dihedral_flow, dihedrals, t, batch, mask=None):
        """
        Compute mirror operator antisymmetric constraint loss for chirality.

        The mirror operator M should satisfy:
        M * φ = -φ (antisymmetric for phi)
        M * ψ = -ψ (antisymmetric for psi)
        M * ω = ω  (symmetric for omega)

        This enforces that the flow respects chirality.
        """
        # Apply mirror operator: flip the sign of phi and psi, keep omega unchanged
        mirror_dihedrals = torch.zeros_like(dihedrals)
        mirror_dihedrals[:, :, 0] = -dihedrals[:, :, 0]  # M * φ = -φ
        mirror_dihedrals[:, :, 1] = -dihedrals[:, :, 1]  # M * ψ = -ψ
        mirror_dihedrals[:, :, 2] = dihedrals[:, :, 2]   # M * ω = ω

        # Reuse original batch features (aatype / atom positions) to keep context consistent
        mirror_batch = {**batch, 'dihedrals': mirror_dihedrals}

        # Prevent recursive mirror loss computation by disabling in inner forward
        mirror_output = self.forward(mirror_batch, t=t, compute_mirror_loss=False)
        mirror_flow = mirror_output['dihedral_flow']

        # Mirror constraint: M * v(θ) should equal -v(M * θ) for phi and psi
        # For omega: M * v(θ) should equal v(M * θ)
        expected_mirror_flow = torch.zeros_like(dihedral_flow)
        expected_mirror_flow[:, :, 0] = -dihedral_flow[:, :, 0]  # -v_φ
        expected_mirror_flow[:, :, 1] = -dihedral_flow[:, :, 1]  # -v_ψ
        expected_mirror_flow[:, :, 2] = dihedral_flow[:, :, 2]   # v_ω

        # Compute constraint loss
        if mask is not None:
            mask = mask.unsqueeze(-1).expand_as(dihedral_flow)

        mirror_constraint_loss = (mirror_flow - expected_mirror_flow) ** 2
        if mask is not None:
            mirror_constraint_loss = mirror_constraint_loss * mask

        return mirror_constraint_loss.mean()

    def forward(self, batch, t=None, compute_mirror_loss: bool = True, x_t: Optional[torch.Tensor] = None):
        """Forward pass of ChiFlow matcher.

        Args:
            batch: dict with 'dihedrals', 'mask', plus context feature inputs.
            t: (optional) time tensor (currently unused but reserved).
            compute_mirror_loss: whether to compute chirality mirror penalty.
            x_t: optional interpolated angles replacing raw dihedrals for path-based supervision.
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

        # Decide which input angles (original or interpolated) to condition on
        input_angles = dihedrals if x_t is None else x_t
        dihedral_flow, _ = self.torus_flow(input_angles, context, t, mask)
        dihedral_xt = input_angles
        # Apply mirror operator antisymmetric constraints for chirality
        if self.use_mirror_constraint and self.training and compute_mirror_loss:
            mirror_loss = self._compute_mirror_constraint_loss(
                dihedral_flow, dihedrals, t, batch, mask
            )
            # Store mirror loss for external access
            self.mirror_loss = mirror_loss
        else:
            # Ensure attribute exists to avoid hasattr checks every step
            if not hasattr(self, 'mirror_loss'):
                self.mirror_loss = torch.tensor(0.0, device=dihedrals.device)

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
        # Keep interface unchanged for external callers
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