# Video recording guide (one video per part)

Seven short videos: an overview, then one per Section 10 group. Each script lists what to show on screen, what to run live, and what to say. The numbers quoted are the ones the code prints, so they will match on your screen. Everything is seeded.

**Setup before recording**
* Terminal at the repo root, font ≥ 16 pt, a wide window (tables are about 120 characters).
* Editor open on the file named in each step.
* Do a warm-up run of `python run_all.py --fast` once so imports are cached.
* Record screen + voice (OBS Studio or the Windows Game Bar, Win+Alt+R). Upload as *Unlisted* if you prefer, then paste the links into the README's **Videos** table.
* Suggested length: 8–12 minutes each. Show each command *executing*. The assignment requires showing execution.

---

## Part 0 — Overview: the problem area and the harness (≈8 min)

1. **Problem (README top).** "I picked home battery dispatch. Each day is 24 decisions; the controller is a small program the improvers rewrite."
2. **`gridwise/env.py`**: show `_prices()` (five tariffs), `simulate()`, and `oracle()`, the DP that gives the cheapest possible day. "Because of the oracle, every day has a score: the fraction of the best possible savings captured."
3. **`gridwise/policy.py`**: `Policy.source()` vs `Policy.act()`. Run `python -m gridwise.cli show-expert`.
4. **`gridwise/evaluate.py`**: `Evaluator.ledger` (every simulated day is billed), `BoundaryViolation` (held-out is locked).
5. **Run** `python run_all.py --fast` and show 17/17 ok. Then show `results/logs/`.
6. **The three methods** (README table): Budget RSI, SO-RSI, SwarmRSI. Say that the other papers appear as smaller labs.

## Part 1 — Group 00: claim vs evidence, audits (10.01–10.02) (≈10 min)

1. Open `reference/course_demos/so_rsi_demo.py` at lines 203 and 249 (`< 0.3` vs `< 0.5`).
2. **Run** `python labs/g00_reading_frontier_research/lab10_01_claim_vs_evidence/run.py`.
   * "With diagnostics OFF the gap is still 0.204, so the improvement is hard-coded."
   * "Recurrence fires on the first-ever failure: each trace is compared with itself."
   * "In the Budget RSI demo, held-out is generated *from* the steering score, correlation 0.92."
3. Open `claim_evidence_analysis.md` and walk through the three claims.
4. **Run** lab 10.02 and show the 9 PASS rows. Stop on check 6: "My own oracle can be beaten by about one 0.25 kWh step, so the 'ideal' is oracle-level, not a proof."
5. Show one paper audit table (`paper_audit_so_rsi.md`).

## Part 2 — Group 01: Budget RSI (10.03–10.05) (≈12 min)

1. **Code walk** of `gridwise/methods/budget_rsi.py`: `calibrate()` (96 edits on 16 fit days, one ridge fit, frozen), `search()` (archive of 4, score every proposal, expand only the top one, return the earliest max-solve node), and the HGM branch (Thompson sampling on Beta posteriors).
2. `gridwise/edits.py` → `diff_features()`: "eight coordinates from the diff text, without running anything."
3. **Run lab 10.03.** Pause on the printed diff and its 8 coordinates (predicted 0.879, actual 0.537). Then the 12-run table: readout 0.817 > HGM 0.787 > parent-rate 0.770 > root 0.712, the paper's ordering. Point out calibration-best 0.887 and the cost table (calibration 6.6× the search).
4. **Run lab 10.04:** coefficients (constants_changed +, conds_delta −, parent_rate ≈ 0). The ablation shows diff-only = full.
5. **Run lab 10.05:** domain features 0.847; the frozen readout transfers; matched-cost table, where HGM on the same total budget gets 0.863. Conclusion: "reproduced at the paper's operating point; the compute claim depends on amortising calibration."

## Part 3 — Group 02: SO-RSI (10.06–10.08) (≈12 min)

1. **Code walk** of `gridwise/methods/so_rsi.py`: `FirstOrderImprover.run()`, `detect()` (each attempt judged against *earlier* attempts only), and the three probes: `sweep_probe`/`lever_probe`, `split_probe`, `attribution_probe`. Then `InquiryMemory`: bans, guards, remap. "This is the improver editing itself."
2. **Run lab 10.06.** Scroll to attempt [12]: the rejected night-charge patch → attribution: "grid charging filled the battery before free solar arrived" → guard `surplus_left_fc <= 4` → ACCEPT +0.069. Show the final program, then the 5-tariff table: +0.122 TOU, +0.083 critical-peak, losses on wholesale and solar_glut, and recurrences 60 → 6.
3. **Run lab 10.07:** 78% of findings are negative; mismatch → guard gives nearly all the gain; probes are 64% of the budget.
4. **Run lab 10.08:** the failing test, five patches with zero progress, anomalies flagged, the rule firing −1 hour on all 37 spike days, lag +1 confirmed. Fix the pipeline and the original policy passes. "Test setup wrong, not function."

## Part 4 — Group 03: AReaL-RSI and SwarmRSI (10.09–10.11) (≈12 min)

1. **Code walk** of `swarm_rsi.py`: `Agent.sample_templates()` (the feedback board), the complementarity formula, `gate()` (context-gated recombination), and hypothesis restarts that keep the implementation.
2. **Run lab 10.11.** Show the recombination events and the diversity curve. In the ablation table (TOU): removing recombination costs 0.022, removing the board 0.026. In the budget curve: +0.029 at small budget. In the experience table: fresh 0.907 → 0.929 with experience, but the control gets 0.933, "so most of it is program transfer, except on solar_glut."
3. **Code walk** of `areal_rsi.py`: the leader's ranking, `ConditionalMemory`, the review.
4. **Run lab 10.10:** the campaign log and memory table. Sharing beats independent on 3/3 tariffs.
5. **Run lab 10.09:** the comparison table and the "when to choose" table.

## Part 5 — Group 04: RSIFlow and Self-HPO (10.12–10.14) (≈10 min)

1. **Code walk** of `rsiflow.py`: `MetaAgent.route()` (UCB per bottleneck) and `evolve()` (prune / grow `extend+tune`).
2. **Run lab 10.13:** the graph edit at step 23, the credit table, and adaptive ≈ static. "The bottleneck was wasted_solar in 56 of 58 steps; nothing to adapt to." That's a legitimate negative result.
3. **Code walk** of `self_hpo.py`: `Guard`, `guard_candidates()`, `h_step()` (validation days, hard-task curriculum), `p_step()`.
4. **Run lab 10.14:** the learned guard `no_grid_charge_if_sunny>=6.0`, and the synergy table (+0.032 on TOU only). Explain why additive synergy double-counts on VPP.
5. **Run lab 10.12:** the side-by-side table and "when to choose".

## Part 6 — Group 05: RSIGym, RSI-Index, metrics (10.15–10.17) (≈12 min)

1. **Code walk** of `gridwise/eaas.py`: `parse_policy()` (a whitelist grammar, not `exec`), `MeteredEvaluator`, and the roles.
2. **Run lab 10.16:** four rejected submissions, agent DENIED held-out, `BudgetExceeded` after 16 evaluations, the three-track table, and the CLI run with exit code 3 on held-out.
3. **Run lab 10.15:** show the course formula sign error, the 5 × 3 table, and RSI-Index SwarmRSI 0.732 / SO-RSI 0.607 / Budget RSI 0.524. Note SO-RSI used 57% of its budget.
4. **Run lab 10.17:** steering proxy Spearman 0.94 with winner's curse +0.037; the learnability table (random days 0.913 beat learnable 0.794); diversity ≠ complementarity.
5. **Close:** "Three methods rebuilt on one task under one protocol. Some claims reproduced (Budget RSI's ordering, SO-RSI's anti-thrashing, swarm experience), some didn't, and every result is in `results/logs/`."
