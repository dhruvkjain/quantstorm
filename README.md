## 58th / 2,118 participants in Quantstorm 2026

**A fixed quote tilt of 2.2, with the opponent-quote de-bias off.**

Everything above is in the code
my approach: the single posterior on the opponent's revealed sum, parity-aligned quotes, opportunity-priced auction bidding with state-conditional power values, closed-form negotiation with "never settle except at a pin", the edge-comparison swap rule, and the directional tilt.

### The configuration

| Parameter | Value | Rationale |
|---|---|---|
| Quote tilt | 2.2 | The exploit, at the largest value keeping the gate margin safely positive |
| Opponent-quote de-bias | 0.00 | Take their midpoint at face value; leaning into the read outearns the bias in this field |
| Auction shade | 0.62 | First-price; the basin is flat but the high end is fragile against aggressive bidders |
| Energy price by round | 0.135 to 0.080 | Opportunity cost, declining to salvage in the final round |
| Reserve per unplayed round | 2 | Keeps budget available for later rounds |
| Tie-break bonus | 1 | Converts a coin-flip win into a certainty for a trivial cost |
| Early-settle threshold | 8.0 | Effectively never settle before the final turn |
| Counter width | widest legal | Width is free in a counter and defends against adverse selection |
| Pin-settle width | 1 | The one boundary case where never-settle stops being free |
| Adverse-selection weight | 0 | The model was derived, tested, and refuted |
| Swap margin | 0.35 coins | Fire when the other hand is the more decisive one |

### The outcome

**Final result: `avg_pnl_per_deal = 1.584006` - positive against the live entrant field.**
