# Group 00 · Read a Frontier Claim (labs 10.01–10.02)

**Question:** which claim does the available evidence actually support?

Before building anything, we treat both the papers and the course's own laptop demos as claims to check.

| Lab | What it does | Headline finding |
|-----|--------------|------------------|
| [10.01 · Claim vs evidence](lab10_01_claim_vs_evidence/README.md) | Runs the course's SO-RSI and Budget RSI demos as experiments, with ablations | Both demo headlines are produced by their data generators, not by the mechanisms |
| [10.02 · Audit a paper](lab10_02_audit_paper/README.md) | Audit templates for the three chosen papers, plus 9 automated checks of this repo's protocol | All 9 checks pass. One real limitation found: the oracle is optimal only to 0.25 kWh |

**What we carry forward:** a harness whose outcomes come from a simulator, not a random-number generator; a frozen held-out boundary enforced in code; matched budgets measured by a ledger; and the habit of asking "what would this look like if the mechanism did nothing?"

Next: [Group 01 · Budget RSI](../g01_budget_rsi/README.md)
