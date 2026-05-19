## Polymer Markov LS Tail Anchor

- Kept the existing short post-warm-start LS behavioral-cloning window unchanged.
- Added an optional weak tail-anchor mode in `utils/behavioral_cloning.py`.
- Wired the polymer Markov runner to activate that tail anchor only when:
  - the trusted BC target is accepted LS,
  - TD3 is still farther than the configured tolerance from that LS action,
  - and the requested TD3 score is worse than the LS score.
- Enabled the tail anchor for polymer Markov defaults with:
  - `weight = 0.03`
  - `action_gap_tolerance = 0.02`

This is intended to keep TD3 closer to the useful LS manifold after the main BC window ends without forcing long-horizon imitation.
