# Name: dhruvkjain
# College: Dhirubhai Ambani University (formerly DAIICT)
# Roll Number: 202301272

"""
logic_4e.py: Divided Oracle entry
================================

Maintains a posterior on ``T``, the opponent's revealed coin sum, and derives
every decision from it as a closed-form expected value:

    E[S]   = k_mine + mu_T
    Var[S] = v_T + 2 * (N_PRIVATE - REVEAL_PER_ROUND * r)

``T`` is the only inferable unknown: the coins neither side has revealed are
mean-zero. Three sources feed the posterior, the prior, a FORESIGHT leak
(exact), and the opponent's opening quote midpoints (noisy, de-biased).

Behaviour by phase:

  auction       value each offered power in ticks, convert to TE at an
                opportunity price, shade, cap at the opponent's budget
  quote         centre on the posterior at parity, width by obligation EV
  negotiation   counter every turn; settle only on the final turn, or earlier
                if the range is too narrow to negotiate
  transform     fire when the opponent's hand is the more decisive one

Design rationale, measurements and rejected alternatives are in PLAN.md.
"""

import math

#  TUNABLES
#: Fraction of fair value to bid in the first-price auction.
SHADE = 0.62

#: Opportunity cost of one TE, in ticks, by round. Above TE_SALVAGE while
#: powers remain to buy, equal to it in round 5.
TE_PRICE = {1: 0.135, 2: 0.125, 3: 0.115, 4: 0.100, 5: 0.080}

#: TE withheld per round not yet played.
RESERVE_PER_REMAINING_ROUND = 2

#: Extra TE added to a non-zero bid, so an equal bid does not go to a coin flip.
TIE_BREAK_BONUS = 1

#: Ticks of edge required to settle before the final negotiation turn.
#: Set high: the bot counters every turn and settles on the last one.
ACCEPT_EARLY_THRESHOLD = 8.0

#: Width to counter at. -1 = widest legal, -2 = the round's floor, otherwise an
#: absolute width. Counters are bounded on both sides,
#: `final_cap <= width <= max_width`, so a target below the round's floor is
#: re-centred up to it: in rounds 1-2 (floor 4) a target of 3 is the floor.
COUNTER_WIDTH_TARGET = -1

#: Settle instead of countering once the range is this narrow.
PIN_SETTLE_WIDTH = 1

#: Also settle when the range is already at the round's floor width. At that
#: point max_width equals the floor, so the only legal counter of that width
#: inside the range is the range itself: countering cannot move the price and
#: merely passes the choice of side to the next mover.
PIN_SETTLE_AT_FLOOR = False

#: Ticks of opening width above the obligation-optimal choice.
OPEN_WIDTH_BONUS = 0

#: Extra weight on mu_T in the QUOTE CENTRE only, not in accept decisions.
#:
#: A symmetric shift of the centre is zero-EV against a reader: moving it by d
#: moves the trade price by d but not which side they take, so it gains d on a
#: buy and loses d on a sell. It only pays if the side is predictable, and it is
#: partly predictable -- their value is T + rho*centre, so a high mu_T makes a
#: buy more likely and a buy is the side I want priced high. Tilting in the
#: direction of mu_T is therefore directional rather than symmetric, and the
#: obligation cost of a small tilt is second-order while the gain is first.
QUOTE_TILT = 2.2

#: Noise variance when reading T off an opponent's opening quote midpoint.
QUOTE_READ_VAR = 0.5
#: Same for a counter midpoint, which is censored by the clamping rules.
COUNTER_READ_VAR = 9.0
USE_COUNTER_READS = True

#: Fraction of my own last quote centre to strip out of their quote midpoint
#: before reading it as a read on T. 1.0 treats them as a full quote-reader,
#: 0.0 as pricing off their own coins alone.
OPP_QUOTE_DEBIAS = 0.0

#: Margin by which the other hand must look more decisive before firing
#: TRANSFORM, in coins.
TRANSFORM_MARGIN = 0.35
#: Value of buying TRANSFORM to deny it, as a multiple of the swap's own value.
DENIAL_WEIGHT = 0.35

#: Tick value per power per round, used as a prior and then scaled by state.
POWER_BASE = {
    "FORESIGHT":    {1: 0.76, 2: 1.16, 3: 1.48, 4: 1.97, 5: 2.02},
    "TRICK_ROOM":   {1: 1.14, 2: 0.85, 3: 0.75, 4: 0.60, 5: 0.52},
    "SUBSTITUTE":   {1: 1.46, 2: 1.15, 3: 0.95, 4: 0.57, 5: 0.29},
    "STEALTH_ROCK": {1: 1.51, 2: 0.75, 3: 0.75, 4: 0.75, 5: 0.00},
    "TRANSFORM":    {1: 1.58, 2: 1.24, 3: 1.31, 4: 0.00, 5: 0.00},
}

#: Multiplier on POWER_BASE for the two shift powers.
SHIFT_VALUE_MULT = 1.0

#: Powers that shift a forced midpoint fill, and by how much.
SHIFT_POWERS = {"TRICK_ROOM": 3, "STEALTH_ROCK": 2}

#: Cap bids at `te_theirs + 1`, which already wins outright.
CAP_BID_AT_THEIR_BUDGET = True

_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)



#  NORMAL-DISTRIBUTION HELPERS
def _phi(x):
    """Standard normal density."""
    if x > 40.0 or x < -40.0:
        return 0.0
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


def _Phi(x):
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _abs_mean(mu, var):
    """E|X| for X ~ N(mu, var)."""
    if var <= 1e-12:
        return abs(mu)
    sd = math.sqrt(var)
    z = mu / sd
    return mu * (2.0 * _Phi(z) - 1.0) + 2.0 * sd * _phi(z)


def _mean_capped(mu, var, cap):
    """E[max(Y, -cap)] for Y ~ N(mu, var) — a payoff with SUBSTITUTE applied."""
    if var <= 1e-12:
        return max(mu, -cap)
    sd = math.sqrt(var)
    a = (mu + cap) / sd
    return -cap + (mu + cap) * _Phi(a) + sd * _phi(a)


class Bot:
    name = "Oracle_4e"

    # lifecycle
    def _reset(self, seat, config, seed):
        """Establish all per-deal state."""
        self.seat = seat
        self.config = config
        self.rev_per_round = config.REVEAL_PER_ROUND
        self.n_private = config.N_PRIVATE
        self.n_rounds = config.N_ROUNDS

        # Reads on T, each (round_seen, mean, noise_var).
        self._quote_reads = []
        self._hard_reads = []
        # Opponent opening-quote midpoints already latched, by round.
        self._latched = {}
        # My own opening quote centre, by round.
        self._my_centre = {}
        # Previous value of obs.my_revealed, for swap detection.
        self._prev_revealed = None

    # bookkeeping, run at the top of every public call
    def _observe(self, obs):
        """Detect a fired TRANSFORM and record T exactly.

        A swap re-slices `revealed` from the swapped hands, so my own revealed
        prefix changes. After it, the opponent's revealed sum equals my previous
        one. Earlier quote reads describe a hand that is now mine, so drop them.
        """
        cur = tuple(obs.my_revealed)
        prev = self._prev_revealed
        if prev is not None and len(prev) <= len(cur) and cur[:len(prev)] != prev:
            known_round = len(prev) // self.rev_per_round
            if known_round >= 1:
                self._hard_reads = [(known_round, float(sum(prev)), 0.0)]
                self._quote_reads = []
                self._latched = {}
        self._prev_revealed = cur

    # the posterior
    def _their_quote_bias(self, obs, j):
        """Their estimate of MY revealed sum at the time they quoted round j.

        Subtracted from their midpoint before reading it as a read on T, since
        a maker centres on its own revealed sum plus its estimate of mine. The
        two possible sources are the last quote centre I showed them and a
        FORESIGHT leak of my coins, blended evenly when they have both.
        """
        theirs = 1 - self.seat
        n_rev = self.rev_per_round * j

        quote_part = None
        earlier = [r for r in self._my_centre if r < j]
        if earlier and OPP_QUOTE_DEBIAS > 0.0:
            quote_part = OPP_QUOTE_DEBIAS * self._my_centre[max(earlier)]

        leak_part = None
        held = any(
            e.get("round") == j and e.get("seat") == theirs
            and e.get("power") == "FORESIGHT"
            for e in obs.auction_log
        )
        if held and n_rev > 0 and len(obs.my_revealed) >= n_rev:
            leak = min(self.config.POWERS["FORESIGHT"]["magnitude"], n_rev)
            leak_part = (leak / n_rev) * float(sum(obs.my_revealed[:n_rev]))

        if quote_part is not None and leak_part is not None:
            return 0.5 * quote_part + 0.5 * leak_part
        if quote_part is not None:
            return quote_part
        if leak_part is not None:
            return leak_part
        return 0.0

    def _harvest_quote_reads(self, obs):
        """Latch the opponent's opening quotes from the contract record.

        Only the opening quote is a clean read; later ranges are contaminated
        by both sides. Latched once per round.
        """
        for c in obs.contracts:
            if c.maker_seat == self.seat or c.round in self._latched:
                continue
            mid = 0.5 * (c.open_bid + c.open_ask)
            self._latched[c.round] = mid
            corrected = mid - self._their_quote_bias(obs, c.round)
            self._quote_reads.append((c.round, corrected, QUOTE_READ_VAR))

    def note_opening(self, obs, quote):
        """Latch the opening quote of the round in progress."""
        r = obs.round
        if obs.is_maker or quote is None or r in self._latched:
            return
        mid = 0.5 * (quote[0] + quote[1])
        self._latched[r] = mid
        corrected = mid - self._their_quote_bias(obs, r)
        self._quote_reads.append((r, corrected, QUOTE_READ_VAR))

    def _posterior_t(self, obs):
        """Posterior mean and variance of the opponent's revealed sum."""
        r = obs.round
        n_rev = self.rev_per_round * r

        # Base: FORESIGHT is an exact conditional, otherwise the prior.
        leak = obs.foresight
        if leak:
            mu, var = float(sum(leak)), float(n_rev - len(leak))
        else:
            mu, var = 0.0, float(n_rev)

        # Use only the single lowest-variance read. Reads are nested prefixes of
        # one random walk, so combining several as independent overstates
        # precision. An older read carries the coins revealed since as variance.
        best = None
        for j, mean, noise in self._hard_reads + self._quote_reads:
            total = noise + self.rev_per_round * max(0, r - j)
            if best is None or total < best[1]:
                best = (mean, total)

        if best is not None:
            mean, total = best
            if total <= 1e-12:
                mu, var = mean, 0.0
            elif var > 1e-12:
                p0, p1 = 1.0 / var, 1.0 / total
                mu = (mu * p0 + mean * p1) / (p0 + p1)
                var = 1.0 / (p0 + p1)

        # T cannot exceed its own coin count.
        if mu > n_rev:
            mu = float(n_rev)
        elif mu < -n_rev:
            mu = float(-n_rev)
        return mu, max(0.0, min(var, float(n_rev)))

    def _posterior_s(self, obs):
        """Posterior mean and variance of the hidden score S."""
        mu_t, var_t = self._posterior_t(obs)
        unrevealed_each = self.n_private - self.rev_per_round * obs.round
        return obs.k_mine + mu_t, var_t + 2.0 * max(0, unrevealed_each)

    def _effective_unseen(self, var_s):
        """Variance as a count of fair coins, for `config.straddle_prob`.

        Forced even: the lattice calculation skips values of the wrong parity,
        so an odd count would model an odd residual and return a wrong figure.
        """
        n = int(round(var_s))
        if n < 0:
            n = 0
        return n - (n % 2)

    def _shift(self, powers):
        """Total fill-shift magnitude a set of powers controls.

        Reimplements `engine.shift_sources`; `engine` is not importable.
        """
        return sum(mag for name, mag in SHIFT_POWERS.items() if name in powers)

    #  AUCTION
    def _forced_rate(self, obs):
        """Forced-fill rate observed this deal, shrunk toward a 0.45 prior."""
        n = len(obs.contracts)
        forced = sum(1 for c in obs.contracts if c.forced)
        return (forced + 0.45 * 2.0) / (n + 2.0)

    def _transform_value(self, obs, mu_t, var_t):
        """Tick value of winning TRANSFORM: swapping trades |k_mine| for E|T|.

        Positive gain is priced as the swap; a decisive hand prices the power as
        a veto instead, since it is consumed whether or not it fires.
        """
        base = POWER_BASE["TRANSFORM"].get(obs.round, 0.0)
        if base <= 0.0:
            return 0.0
        theirs = _abs_mean(mu_t, var_t)
        mine = abs(obs.k_mine)
        gain = theirs - mine
        if gain > TRANSFORM_MARGIN:
            return base * min(1.3, max(0.3, gain / 2.0))
        if DENIAL_WEIGHT > 0.0 and mine - theirs > 1.0:
            return base * DENIAL_WEIGHT
        return 0.0

    def _power_value(self, obs, name):
        """Tick value of `name` in this state: the base row, scaled."""
        r = obs.round
        base = POWER_BASE.get(name, {}).get(r, 0.5)
        if base <= 0.0:
            return 0.0

        mu_t, var_t = self._posterior_t(obs)

        if name == "TRANSFORM":
            return self._transform_value(obs, mu_t, var_t)

        if name == "FORESIGHT":
            # Scale by how much uncertainty about T is left to remove.
            prior_var = float(self.rev_per_round * r)
            if prior_var <= 0.0:
                return base
            return base * min(1.15, max(0.15, var_t / prior_var))

        if name == "SUBSTITUTE":
            # Scale by the width of the loss distribution being capped.
            _, var_s = self._posterior_s(obs)
            ref = self.config.residual_sd(r)
            if ref <= 0.0:
                return base
            return base * min(1.4, max(0.6, math.sqrt(var_s) / ref))

        if name in SHIFT_POWERS:
            # Only pays on a forced fill, so scale by the observed rate.
            return base * SHIFT_VALUE_MULT \
                * min(1.6, max(0.5, self._forced_rate(obs) / 0.45))

        return base

    def _bid(self, obs, offered):
        self._observe(obs)
        self._harvest_quote_reads(obs)

        te = obs.te_mine
        if not offered or te <= 0:
            return {}

        rounds_left = max(0, self.n_rounds - obs.round)
        headroom = max(0, te - RESERVE_PER_REMAINING_ROUND * rounds_left)
        if headroom <= 0:
            headroom = min(te, 1) if obs.round >= self.n_rounds else 0
        price = TE_PRICE.get(obs.round, self.config.TE_SALVAGE)

        out = {}
        spent = 0
        for name in offered:
            value = self._power_value(obs, name)
            if value <= 0.0:
                continue
            amount = int(SHADE * value / price)
            if amount > 0:
                amount += TIE_BREAK_BONUS

            # Their bid cannot exceed obs.te_theirs, so te_theirs + 1 wins
            # outright and anything above it is wasted TE.
            if CAP_BID_AT_THEIR_BUDGET:
                need = max(1, obs.te_theirs + 1)
                if amount > need:
                    amount = need

            amount = min(amount, headroom - spent, te - spent)
            if amount <= 0:
                continue
            out[name] = amount
            spent += amount

        # A vector totalling over te_mine is zeroed by the engine, not scaled.
        if spent > te:
            return {}
        return out

    #  MAKER
    def _choose_width(self, obs, var_s):
        """Opening width maximising the obligation EV, plus OPEN_WIDTH_BONUS.

            MAKER_OBLIGATION * (p_actual - p_baseline) - WIDTH_PREMIUM * excess

        The obligation scores against the default unseen count whatever width is
        quoted, so the first term is the edge held by a better-informed maker.
        Ties go to the tighter quote.
        """
        cfg = self.config
        r, floor, cap = obs.round, obs.final_cap, obs.spread_cap
        if cap <= floor:
            return floor

        m_eff = self._effective_unseen(var_s)
        lam, prem = cfg.MAKER_OBLIGATION, cfg.WIDTH_PREMIUM

        best_w, best_ev = floor, None
        for w in range(floor, cap + 1):
            edge = lam * (cfg.straddle_prob(r, w, unseen=m_eff)
                          - cfg.straddle_prob(r, w))
            ev = edge - prem * (w - floor)
            if best_ev is None or ev > best_ev + 1e-12:
                best_w, best_ev = w, ev
        return min(cap, best_w + OPEN_WIDTH_BONUS)

    def _centre(self, mu, k_mine):
        """Round `mu` to an integer matching the parity of S.

        S - k_mine sums an even number of coins, so S == k_mine (mod 2). The
        wrong parity misaligns the quote window against the lattice.
        """
        c = int(round(mu))
        if (c - k_mine) % 2:
            c += 1 if mu >= c else -1
        return c

    def _quote(self, obs):
        self._observe(obs)
        self._harvest_quote_reads(obs)

        mu_s, var_s = self._posterior_s(obs)
        w = self._choose_width(obs, var_s)
        if QUOTE_TILT:
            # mu_s is k_mine + mu_T, so the read is recoverable without a
            # second posterior evaluation.
            mu_s = mu_s + QUOTE_TILT * (mu_s - obs.k_mine)
        c = self._centre(mu_s, obs.k_mine)

        # Keep the quote inside the range S can take.
        limit = self.config.N_COINS
        lo = max(-limit, min(limit - w, c - w // 2))
        self._my_centre[obs.round] = lo + 0.5 * w
        return (lo, lo + w)

    #  RESPONDER
    def _note_counter(self, obs, quote):
        """Record a low-confidence read on T from an opponent's counter.

        Their counter sits near their own value, `T` plus their read of my
        revealed sum; subtracting my centre leaves a read on T. Censored by the
        clamping rules, and skipped when it sits exactly on my own centre.
        """
        if not USE_COUNTER_READS or not obs.is_maker:
            return
        centre = self._my_centre.get(obs.round)
        if centre is None:
            return
        mid = 0.5 * (quote[0] + quote[1])
        if abs(mid - centre) < 1e-9:
            return
        self._quote_reads.append((obs.round, mid - centre, COUNTER_READ_VAR))

    def _counter_width(self, obs, quote):
        """Counter width: the reduction rule's maximum, narrowed to target.

            max_width = min(ask-bid, max(final_cap, (ask-bid) - MIN_REDUCTION))
        """
        bid, ask = quote
        cur = ask - bid
        w = min(cur, max(obs.final_cap, cur - self.config.MIN_REDUCTION))
        if COUNTER_WIDTH_TARGET == -2:
            w = min(w, obs.final_cap)
        elif COUNTER_WIDTH_TARGET >= 0:
            w = min(w, COUNTER_WIDTH_TARGET)
        return max(0, w)

    def _counter_range(self, obs, quote, mu_s):
        """A legal counter centred on the posterior, clamped inside the range."""
        bid, ask = quote
        w = self._counter_width(obs, quote)
        down = w // 2
        c = self._centre(mu_s, obs.k_mine)
        c = max(bid + down, min(ask - (w - down), c))
        nb = c - down
        return nb, nb + w

    def _forcing_range(self, obs, quote):
        """The counter that forces the midpoint fill on the best terms.

        Counters are bounded on BOTH sides: `final_cap <= width <= max_width`.
        Anything narrower is re-centred up to the floor, so a zero-width pin is
        not available and asking for one gives back a range straddling the
        anchor instead of sitting at it.

        The last quoter is the short seat, so the best legal counter is the
        narrowest allowed width pushed to the top of the range: that maximises
        the midpoint being sold into. Width is the floor, which is always legal
        because every range on the table is at least that wide.
        """
        bid, ask = quote
        w = obs.final_cap
        nb = max(bid, ask - w)
        return nb, min(ask, nb + w)

    def _respond(self, obs, quote, turn):
        self._observe(obs)
        self._harvest_quote_reads(obs)
        self.note_opening(obs, quote)

        bid, ask = quote
        if turn > 2:
            self._note_counter(obs, quote)

        mu_s, var_s = self._posterior_s(obs)
        capped = "SUBSTITUTE" in obs.powers_mine
        cap = float(self.config.POWERS["SUBSTITUTE"]["magnitude"]) if capped else 0.0

        def payoff(mean):
            return _mean_capped(mean, var_s, cap) if capped else mean

        ev_buy = payoff(mu_s - ask)
        ev_sell = payoff(bid - mu_s)

        # Countering on the final turn makes me the last quoter, hence short,
        # and charges the forcing fee. Priced off the range I would submit.
        shift = self._shift(obs.powers_mine) - self._shift(obs.powers_theirs)
        fb, fa = self._forcing_range(obs, quote)
        forced_price = (fb + fa) // 2 + shift
        ev_force = payoff(forced_price - mu_s) - self.config.FORCED_FILL_FEE

        if turn >= obs.n_turns:
            if ev_buy >= ev_sell and ev_buy >= ev_force:
                return "ACCEPT_BUY"
            if ev_sell >= ev_force:
                return "ACCEPT_SELL"
            return ("COUNTER", fb, fa)

        # Too narrow to negotiate: countering would only pass the choice of side
        # to the next mover. Settle unless riding to a forced fill still wins.
        settle_w = PIN_SETTLE_WIDTH
        if PIN_SETTLE_AT_FLOOR and obs.final_cap > settle_w:
            settle_w = obs.final_cap
        if ask - bid <= settle_w and ev_force <= max(ev_buy, ev_sell):
            return "ACCEPT_BUY" if ev_buy >= ev_sell else "ACCEPT_SELL"

        # Otherwise counter, and settle early only on an overwhelming edge.
        nb, na = self._counter_range(obs, quote, mu_s)
        if ACCEPT_EARLY_THRESHOLD >= ev_buy and ACCEPT_EARLY_THRESHOLD >= ev_sell:
            return ("COUNTER", nb, na)
        if ev_buy >= ev_sell:
            return "ACCEPT_BUY"
        return "ACCEPT_SELL"

    #  TRANSFORM
    def _use_transform(self, obs):
        """Fire when the opponent's hand is the more decisive one.

        Compares E|T| against |k_mine|; declining spends the power to keep my
        own coins, since it is consumed either way.
        """
        self._observe(obs)
        self._harvest_quote_reads(obs)
        mu_t, var_t = self._posterior_t(obs)
        return _abs_mean(mu_t, var_t) > abs(obs.k_mine) + TRANSFORM_MARGIN


    #  PUBLIC INTERFACE - every call is failure-isolated
    #
    # The engine turns a raise into that method's fallback, but a worker killed
    # mid-call is marked out for the rest of the deal, and every remaining call
    # then counts a fresh time violation; six forfeit the match. So nothing
    # below propagates: each method answers with a cheap legal value instead.

    def reset(self, seat, config, seed):
        """Per-deal setup, with minimal state on failure so later calls work."""
        try:
            self._reset(seat, config, seed)
        except Exception:
            self.seat = seat
            self.config = config
            self.rev_per_round = 4
            self.n_private = 20
            self.n_rounds = 5
            self._quote_reads = []
            self._hard_reads = []
            self._latched = {}
            self._my_centre = {}
            self._prev_revealed = None

    def bid(self, obs, offered):
        """TE bids; contests nothing on failure."""
        try:
            return self._bid(obs, offered)
        except Exception:
            return {}

    def quote(self, obs):
        """Opening quote; k_mine at the round's floor width on failure."""
        try:
            return self._quote(obs)
        except Exception:
            try:
                w = obs.final_cap
                lo = obs.k_mine - w // 2
                return (lo, lo + w)
            except Exception:
                return (-2, 2)

    def respond(self, obs, quote, turn):
        """Negotiation move; the better side of the quote on failure."""
        try:
            return self._respond(obs, quote, turn)
        except Exception:
            try:
                bid, ask = quote[0], quote[1]
                v = obs.k_mine
                return "ACCEPT_BUY" if (v - ask) >= (bid - v) else "ACCEPT_SELL"
            except Exception:
                return "ACCEPT_BUY"

    def use_transform(self, obs):
        """Swap decision; declines on failure."""
        try:
            return bool(self._use_transform(obs))
        except Exception:
            return False
