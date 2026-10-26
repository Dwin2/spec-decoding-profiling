# Candidate primitives

Each primitive is asymmetric: only the draft is throttled, never the target.

**P1: CTA admission cap.** Draft CTAs may run on any SM, but only in residency the target does not use, so the draft fills spare slots without the fragmentation of a static SM split.

**P2: HBM credit gate.** Bound the draft's outstanding L2-miss traffic (or transactions per window) so draft requests cannot inflate the target's memory queueing delay.

**P3: Cache pollution guard.** Keep draft lines from evicting the target's hot working set, e.g. by persisting the target's weights/KV in L2 and giving draft loads evict-first hints.
