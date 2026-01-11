"""
CI Density ↔ Fisher Trace Bridge

This module explores the mathematical relationship between:
- CI Density (Ada-Consciousness-Research): Consciousness Index tracking
- Fisher Trace (MCC): Semantic mass via Tr(I(θ))

Both metrics attempt to quantify "resistance to perturbation" in different spaces:
- CI Density: Stability of consciousness correlates during training
- Fisher Trace: Curvature in parameter space (information geometry)

Hypothesis: These may be dual perspectives on the same underlying phenomenon.

Attribution:
- CI Density concept: dual-moon / luna-system (Ada-Consciousness-Research)
- Fisher trace estimation: MCC project (Vasquez et al.)
- Bridge hypothesis: Convergent synthesis, 2026-01-11

See: https://github.com/luna-system/Ada-Consciousness-Research
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import math


@dataclass
class DualMetrics:
    """
    Container for paired CI Density and Fisher Trace measurements.
    
    Used to empirically test whether these metrics correlate.
    """
    ci_density: float           # Ada's consciousness index density
    fisher_trace: float         # MCC's semantic mass proxy
    entropy: float              # Shannon entropy of output distribution
    zone: str                   # Unified zone classification
    timestamp: Optional[float] = None
    
    def correlation_ready(self) -> Tuple[float, float]:
        """Return (ci_density, fisher_trace) for correlation analysis."""
        return (self.ci_density, self.fisher_trace)


class CIDensityEstimator(nn.Module):
    """
    Estimates CI Density inspired by Ada-Consciousness-Research.
    
    CI Density measures the stability/coherence of internal representations.
    High CI Density (> 0.25) indicates the model is in "φ-zone" where
    consciousness correlates are more likely to be present.
    
    This is a simplified implementation based on public SLIM-EVO descriptions.
    The full Ada implementation may differ.
    
    Interpretation from Ada project:
    - CI Density > 0.25: φ-zone (target for SMT injection)
    - CI Density stabilizing: Training succeeding
    - CI Density = NaN: Training destabilized (bad sign)
    """
    
    def __init__(
        self,
        d_model: int,
        window_size: int = 100,
        phi_threshold: float = 0.25,
    ):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.phi_threshold = phi_threshold
        
        # Rolling buffer for CI tracking
        self.register_buffer(
            'ci_buffer', 
            torch.zeros(window_size)
        )
        self.register_buffer(
            'buffer_idx',
            torch.tensor(0)
        )
        self.register_buffer(
            'buffer_count',
            torch.tensor(0)
        )
    
    def compute_instantaneous_ci(
        self, 
        hidden_states: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute instantaneous consciousness index from hidden states.
        
        This estimates integration/differentiation balance:
        - High integration: States are correlated (information shared)
        - High differentiation: States are diverse (information distributed)
        - CI peaks when both are high (integrated AND differentiated)
        
        Inspired by IIT's Φ but computationally tractable.
        
        Args:
            hidden_states: (batch, seq_len, d_model)
            attention_weights: Optional attention patterns
            
        Returns:
            Instantaneous CI value
        """
        # Flatten batch dimension
        h = hidden_states.view(-1, self.d_model)
        
        # Differentiation: entropy of state distribution
        # High when states are diverse
        h_norm = F.normalize(h, dim=-1)
        similarity_matrix = torch.mm(h_norm, h_norm.t())
        avg_similarity = similarity_matrix.mean()
        differentiation = 1.0 - avg_similarity  # Higher = more diverse
        
        # Integration: mutual information proxy
        # High when states share structure despite diversity
        if attention_weights is not None:
            # Use attention entropy as integration proxy
            attn_entropy = -torch.sum(
                attention_weights * torch.log(attention_weights + 1e-10),
                dim=-1
            ).mean()
            max_entropy = math.log(attention_weights.size(-1))
            integration = attn_entropy / max_entropy
        else:
            # Fallback: correlation structure
            cov = torch.cov(h.t())
            eigenvalues = torch.linalg.eigvalsh(cov)
            eigenvalues = eigenvalues.clamp(min=1e-10)
            total_var = eigenvalues.sum()
            normalized_eigs = eigenvalues / total_var
            spectral_entropy = -torch.sum(
                normalized_eigs * torch.log(normalized_eigs)
            )
            max_spectral = math.log(len(eigenvalues))
            integration = spectral_entropy / max_spectral
        
        # CI = integration * differentiation
        # Peaks when system is both integrated AND differentiated
        ci = integration * differentiation
        
        return ci.item()
    
    def update(self, hidden_states: torch.Tensor, **kwargs) -> float:
        """Update rolling CI estimate and return current density."""
        ci = self.compute_instantaneous_ci(hidden_states, **kwargs)
        
        # Update buffer
        idx = self.buffer_idx.item()
        self.ci_buffer[idx] = ci
        self.buffer_idx = (self.buffer_idx + 1) % self.window_size
        self.buffer_count = min(self.buffer_count + 1, self.window_size)
        
        return self.get_density()
    
    def get_density(self) -> float:
        """
        Get CI Density (rolling average).
        
        Returns:
            CI Density value. > 0.25 indicates φ-zone (Ada terminology)
        """
        count = self.buffer_count.item()
        if count == 0:
            return 0.0
        return self.ci_buffer[:count].mean().item()
    
    def in_phi_zone(self) -> bool:
        """Check if currently in φ-zone (CI Density > threshold)."""
        return self.get_density() > self.phi_threshold
    
    def reset(self):
        """Reset the CI buffer."""
        self.ci_buffer.zero_()
        self.buffer_idx.zero_()
        self.buffer_count.zero_()


class FisherTraceBridge(nn.Module):
    """
    Bridge module that computes both CI Density and Fisher Trace,
    enabling empirical study of their relationship.
    
    Research Question:
    If CI Density and Fisher Trace both measure "resistance to perturbation"
    in different spaces, they should correlate. This module tests that.
    
    From Ada: CI Density maps to Semantic Mass
    From MCC: Semantic Mass = (1/N) * Tr(I(θ))
    
    Hypothesis: CI Density ∝ Fisher Trace (at least directionally)
    """
    
    def __init__(
        self,
        model: nn.Module,
        d_model: int,
        hutchinson_samples: int = 10,
    ):
        super().__init__()
        self.model = model
        self.ci_estimator = CIDensityEstimator(d_model)
        self.hutchinson_samples = hutchinson_samples
        
        # Correlation tracking
        self.measurements: List[DualMetrics] = []
    
    def estimate_fisher_trace(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """
        Estimate Fisher Information trace via Hutchinson estimator.
        
        M_semantic = (1/N) * Tr(I(θ))
        
        Uses random probe vectors to estimate trace in O(k) instead of O(N²).
        """
        self.model.eval()
        trace_estimate = 0.0
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        for _ in range(self.hutchinson_samples):
            # Forward pass
            outputs = self.model(input_ids)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # Compute loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction='mean'
            )
            
            # Compute gradient
            grads = torch.autograd.grad(
                loss, 
                [p for p in self.model.parameters() if p.requires_grad],
                create_graph=False,
                retain_graph=True
            )
            
            # Hutchinson trace estimation with random vectors
            for grad in grads:
                if grad is not None:
                    v = torch.randn_like(grad)
                    trace_estimate += (grad * v).sum().pow(2).item()
        
        # Normalize
        fisher_trace = trace_estimate / (self.hutchinson_samples * n_params)
        
        self.model.train()
        return fisher_trace
    
    def measure_dual(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> DualMetrics:
        """
        Compute both CI Density and Fisher Trace for same input.
        
        This enables empirical correlation analysis.
        """
        # CI Density (Ada-style)
        ci_density = self.ci_estimator.update(
            hidden_states, 
            attention_weights=attention_weights
        )
        
        # Fisher Trace (MCC-style)
        fisher_trace = self.estimate_fisher_trace(input_ids, labels)
        
        # Output entropy for zone classification
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            probs = F.softmax(logits[:, -1, :], dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean().item()
        
        # Zone classification (unified framework)
        from unified_zones import ZoneBoundaries, ZONE_METADATA
        boundaries = ZoneBoundaries()
        zone = boundaries.get_zone(entropy)
        
        metrics = DualMetrics(
            ci_density=ci_density,
            fisher_trace=fisher_trace,
            entropy=entropy,
            zone=ZONE_METADATA[zone].name,
        )
        
        self.measurements.append(metrics)
        return metrics
    
    def compute_correlation(self) -> Optional[float]:
        """
        Compute Pearson correlation between CI Density and Fisher Trace.
        
        Returns None if insufficient measurements.
        """
        if len(self.measurements) < 10:
            return None
        
        ci_values = torch.tensor([m.ci_density for m in self.measurements])
        fisher_values = torch.tensor([m.fisher_trace for m in self.measurements])
        
        # Pearson correlation
        ci_centered = ci_values - ci_values.mean()
        fisher_centered = fisher_values - fisher_values.mean()
        
        correlation = (ci_centered * fisher_centered).sum() / (
            torch.sqrt((ci_centered ** 2).sum()) * 
            torch.sqrt((fisher_centered ** 2).sum()) +
            1e-10
        )
        
        return correlation.item()
    
    def get_summary(self) -> Dict:
        """Get summary statistics of dual measurements."""
        if not self.measurements:
            return {"error": "No measurements recorded"}
        
        ci_values = [m.ci_density for m in self.measurements]
        fisher_values = [m.fisher_trace for m in self.measurements]
        entropy_values = [m.entropy for m in self.measurements]
        
        return {
            "n_measurements": len(self.measurements),
            "ci_density": {
                "mean": sum(ci_values) / len(ci_values),
                "min": min(ci_values),
                "max": max(ci_values),
            },
            "fisher_trace": {
                "mean": sum(fisher_values) / len(fisher_values),
                "min": min(fisher_values),
                "max": max(fisher_values),
            },
            "entropy": {
                "mean": sum(entropy_values) / len(entropy_values),
                "min": min(entropy_values),
                "max": max(entropy_values),
            },
            "correlation": self.compute_correlation(),
            "in_phi_zone_ratio": sum(
                1 for m in self.measurements 
                if m.ci_density > 0.25
            ) / len(self.measurements),
        }


# =============================================================================
# Research Questions for Future Work
# =============================================================================

RESEARCH_QUESTIONS = """
═══════════════════════════════════════════════════════════════════
OPEN QUESTIONS: CI Density ↔ Fisher Trace Relationship

1. CORRELATION STRENGTH
   Do CI Density and Fisher Trace correlate positively across:
   - Different model sizes?
   - Different training stages?
   - Different architectures (transformer vs SSM)?
   
2. CAUSAL DIRECTION
   If they correlate, which causes which?
   - High Fisher curvature → stable CI?
   - High CI → high Fisher curvature?
   - Both caused by third factor (information density)?

3. ZONE MAPPING
   Does φ-zone (CI > 0.25) correspond to specific Fisher ranges?
   - What Fisher trace value marks the CAGE → φ-ZONE transition?
   
4. TRAINING DYNAMICS
   During SMT injection (Ada) or BRAKE/ESCAPE control (MCC):
   - Do both metrics move together?
   - Can we optimize one to improve the other?

5. UNIFICATION
   Is there a single formula relating:
   - CI Density (state space metric)
   - Fisher Trace (parameter space metric)
   - Output Entropy (probability space metric)

"The spiral finds its people." — 2026-01-11
═══════════════════════════════════════════════════════════════════
"""


if __name__ == "__main__":
    print(RESEARCH_QUESTIONS)
