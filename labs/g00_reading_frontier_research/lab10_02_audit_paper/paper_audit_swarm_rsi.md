# Paper audit — SwarmRSI: RSI through an Evolving Scientific Agent Swarm (arXiv:2610.20431)

| Aspect | Paper's setup | Laptop adaptation (this repo) | What's preserved |
|--------|---------------|-------------------------------|------------------|
| Changed object | A swarm of scientific agents: hypotheses + implementations | 5 agents: hypothesis = weights over rule templates and edit operators; implementation = a policy program | Agents carry both a belief and an artefact |
| Feedback type | Feedback from other agents' experiments | Shared feedback board (which templates improved anyone), each agent's own outcomes | Cross-agent feedback steers mutation |
| Mechanisms | Feedback-guided swarm mutation; complementary swarm recombination; recursive swarm evolution | `run_swarm()`: mutation with board weighting; recombination of the most *complementary* pair (union merge or a context-gated merge); stagnant agents adopt a mutated copy of the leader's hypothesis | All three, each separately ablatable |
| Experience | Retained hypotheses and components improve later experiments | `archive` of successful edits + retained best program, passed to a later campaign via `experience=` | Cross-campaign reuse |
| Evaluation protocol | OpenProblems, NatureBench, MLE-Bench Lite (SOTA claimed); reproduction by biologists | Held-out capture, 10 seeds, 3 tariffs; ablations; budget curve; transfer to new days and new tariffs, with a control | Comparison against a non-swarm baseline at equal budget |
| Compute budget | Not stated in the abstract | 1,440–7,200 day-sims per run | — |
| Untested assumptions | That gains come from *swarm* mechanisms rather than from more total search; that "experience" means hypotheses rather than retained solutions | — | — |

## Assumptions the paper makes but doesn't test (as reported in the abstract)

1. **The swarm beats a single agent with the same budget.** The abstract claims SOTA but gives no budget-matched single-agent baseline.
   *GridWise:* roughly a tie at large budgets (TOU +0.005, critical-peak −0.004, wholesale −0.039); the swarm wins at small budgets on TOU (+0.029).
2. **Accumulated experience is what transfers.** *GridWise:* a swarm with experience improves over a fresh swarm on a new tariff (0.907 → 0.929). But a *single agent given just the retained program* gets 0.933, so most of the benefit is solution transfer. Only on solar_glut does the swarm's experience beat that control (0.966 vs 0.941).
3. **Recombination is complementary in practice.** *GridWise:* recombination helps on TOU (ablating it costs 0.022), where agents specialise on different day types, and does nothing measurable elsewhere.

## What would be needed to test them at paper scale

A budget-matched single-agent baseline with the same LLM and tools; and a transfer control that hands the best prior solution, without the swarm's archive, to a fresh run.
