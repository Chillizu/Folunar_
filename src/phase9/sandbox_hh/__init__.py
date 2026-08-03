"""Phase 9 sandbox-hh (E1): two-layer open-loop agent migrated into the
Phase 8 Docker-sandbox world.

High layer  : frontier-goal selection over an incrementally built
              directory graph — J(d) = unvisited_density(d) - lam*dist.
Low layer   : byte-identical Phase 8 count-driven exploration
              (Phase8Explorer.select_action + generate_phase8_candidates).

Contract: local://contract-sbh.md (pre-registered gates FF-SBH-1/2).
"""

__version__ = "0.1.0"
