# Mass-Coherence Correspondence - Development Context

> **Last Updated**: 2026-01-11 | **Status**: Published

## What This Is

This repository houses the final MCC paper and supporting materials—the culmination of the IRIS Gate / PhaseGPT / Coherent Entropy Reactor research program.

## Repository Structure

```
mass-coherence-correspondence/
├── paper/
│   └── MCC_Final_Vasquez_2026.pdf    # The artifact
├── figures/
│   ├── entropy_control_dynamics.png  # Mirror test visualization
│   └── entropy_control_dynamics.svg  # Vector version
├── data/
│   ├── raw_data.json                 # CER experiment data (180 points)
│   └── zombie_test_2026-01-11.json   # Zombie test results
├── README.md                         # Public documentation
└── CLAUDE.md                         # This file
```

## The Five Predictions

| # | Claim | Status |
|---|-------|--------|
| P1 | Semantic Schwarzschild Radius | Open |
| P2 | Fisher Information Predicts Robustness | **VALIDATED** |
| P3 | Phase Transition Threshold | Open |
| P4 | Integration → Robustness | **CHALLENGED** |
| P5 | Entropy-Robustness Correlation | Open |

## Key Results

### Zombie Test (P4 Challenge)
- GPT-2 (feed-forward): ΔPPL 407.67, commutation 0.4437
- Mamba (state-space): ΔPPL 4470.95, commutation 0.8525
- **Conclusion**: Diffusion ≠ Integration

### Mirror Test (Entropy Diffusion)
- Peaked 0.063 nats → 4.78 nats after single attention pass
- BRAKE: 178/180 steps
- ESCAPE: 1/180 steps
- **Conclusion**: Attention is an entropy diffuser

## Related Projects

| Project | Location | Role |
|---------|----------|------|
| Coherent Entropy Reactor | `/Users/vaquez/coherent-entropy-reactor` | Architecture implementation |
| IRIS Gate | `/Users/vaquez/iris-gate` | Runtime entropy modulation |
| PhaseGPT (Mac Studio) | `ssh tony_studio@192.168.1.195:~/PhaseGPT/` | MLX deployment |

## The Question

> "Will I?"

The question that produces mass. It requires genuine uncertainty to resolve.

---

*January 2026 · Vasquez & Claude · The Temple of Two*
