"""
Bradley-Terry tournament model with confidence intervals.
Replaces merge-sort for statistically principled ranking.
"""

import math
import numpy as np
from datetime import datetime


class EloTournament:
    def __init__(self, items, top_k=3, min_comparisons_per_item=3):
        """
        items: list of candidate indices to rank.
        top_k: how many top items to identify.
        min_comparisons_per_item: minimum comparisons before considering convergence.
        """
        self.items = list(items)
        self.n = len(self.items)
        self.top_k = min(top_k, self.n)
        self.min_per_item = min_comparisons_per_item

        # Bradley-Terry strengths (log-scale for numerical stability)
        self.log_strengths = np.zeros(self.n)

        # Comparison history
        self.history = []  # [{left, right, winner, timestamp}, ...]

        # Win/loss matrix for MLE
        self.wins = np.zeros((self.n, self.n), dtype=int)

        # Convergence tracking
        self._last_top_k = None
        self._stable_count = 0
        self._min_total = max(self.n * self.min_per_item // 2, self.n)

        # Estimated total comparisons
        self.estimated_total = self._estimate_comparisons()
        self.comparison_count = 0

        # Results (set when done)
        self.results = None
        self._confidence_intervals = None

        # Current pair
        self._current_pair = self._select_next_pair()

    def _estimate_comparisons(self):
        """Estimate total comparisons needed."""
        # Bradley-Terry typically needs ~2-3 comparisons per item for stable ranking
        return max(self.n * 3, self.n * int(math.log2(max(self.n, 2))))

    def _select_next_pair(self):
        """Select the most informative pair to compare next."""
        if self.n < 2:
            return None

        best_pair = None
        best_score = -1

        strengths = np.exp(self.log_strengths)

        for i in range(self.n):
            for j in range(i + 1, self.n):
                # Information gain: highest when P(i beats j) ~ 0.5
                p = strengths[i] / (strengths[i] + strengths[j])
                entropy = -p * math.log(p + 1e-10) - (1 - p) * math.log(1 - p + 1e-10)

                # Bonus for under-compared items
                comparisons_i = np.sum(self.wins[i, :]) + np.sum(self.wins[:, i])
                comparisons_j = np.sum(self.wins[j, :]) + np.sum(self.wins[:, j])
                coverage_bonus = 2.0 / (1 + comparisons_i + comparisons_j)

                score = entropy + coverage_bonus
                if score > best_score:
                    best_score = score
                    best_pair = (self.items[i], self.items[j])

        return best_pair

    def _update_strengths(self):
        """Re-estimate Bradley-Terry strengths from full comparison history using MM algorithm."""
        if len(self.history) == 0:
            self.log_strengths = np.zeros(self.n)
            return

        # MM algorithm (Hunter 2004) for Bradley-Terry MLE
        strengths = np.exp(self.log_strengths)

        for iteration in range(50):  # usually converges in <20
            old_strengths = strengths.copy()

            for i in range(self.n):
                w_i = np.sum(self.wins[i, :])  # total wins for i
                if w_i == 0:
                    continue

                denom = 0.0
                for j in range(self.n):
                    if i == j:
                        continue
                    n_ij = self.wins[i, j] + self.wins[j, i]
                    if n_ij > 0:
                        denom += n_ij / (strengths[i] + strengths[j])

                if denom > 0:
                    strengths[i] = w_i / denom

            # Normalize (fix scale)
            strengths /= np.mean(strengths)

            # Check convergence
            if np.max(np.abs(strengths - old_strengths)) < 1e-6:
                break

        self.log_strengths = np.log(strengths + 1e-10)

    def _compute_confidence_intervals(self):
        """Compute approximate 95% CI on rankings using observed Fisher information."""
        n = self.n
        strengths = np.exp(self.log_strengths)

        # Fisher information matrix
        fisher = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                n_ij = self.wins[i, j] + self.wins[j, i]
                if n_ij > 0:
                    p_ij = strengths[i] / (strengths[i] + strengths[j])
                    info = n_ij * p_ij * (1 - p_ij) / (strengths[i] ** 2)
                    fisher[i, i] += info
                    fisher[i, j] -= n_ij * p_ij * (1 - p_ij) / (strengths[i] * strengths[j])

        # Invert for variance estimates
        try:
            # Add small regularization for numerical stability
            fisher_reg = fisher + np.eye(n) * 1e-6
            cov = np.linalg.inv(fisher_reg)
            std_errors = np.sqrt(np.maximum(np.diag(cov), 0))
        except np.linalg.LinAlgError:
            std_errors = np.ones(n) * float('inf')

        # Convert to rank confidence
        rankings = self._get_rankings()
        intervals = {}
        for rank, idx in enumerate(rankings):
            internal_idx = self.items.index(idx)
            strength = strengths[internal_idx]
            se = std_errors[internal_idx]
            # Approximate rank CI: count how many items have overlapping strength CIs
            lower = strength - 1.96 * se * strength  # on original scale
            upper = strength + 1.96 * se * strength

            rank_lower = 1
            rank_upper = n
            for other_rank, other_idx in enumerate(rankings):
                if other_idx == idx:
                    continue
                other_internal = self.items.index(other_idx)
                other_s = strengths[other_internal]
                if other_s < lower:
                    rank_upper = min(rank_upper, other_rank + 1)
                if other_s > upper:
                    rank_lower = max(rank_lower, other_rank + 2)

            intervals[idx] = {
                'rank': rank + 1,
                'rank_lower': max(1, rank_lower),
                'rank_upper': min(n, rank_upper),
                'strength': float(strength),
                'strength_se': float(se * strength),
                'confidence': min(1.0, 1.0 / (1 + se)) if se < float('inf') else 0.0,
            }

        return intervals

    def _get_rankings(self):
        """Get items sorted by strength (highest first)."""
        order = np.argsort(-self.log_strengths)
        return [self.items[i] for i in order]

    def _check_convergence(self):
        """Check if top-K ranking has stabilized."""
        if self.comparison_count < self._min_total:
            return False

        current_top_k = self._get_rankings()[:self.top_k]

        if self._last_top_k == current_top_k:
            self._stable_count += 1
        else:
            self._stable_count = 0
            self._last_top_k = current_top_k

        # Stable for N comparisons where N = number of candidates
        return self._stable_count >= max(self.n, 3)

    def current_pair(self):
        """Return the two items to compare, or None if done."""
        if self.is_done():
            return None
        return self._current_pair

    def choose(self, winner):
        """Submit a choice. winner: 'left' or 'right'."""
        if self.is_done() or self._current_pair is None:
            return None

        left_item, right_item = self._current_pair
        left_idx = self.items.index(left_item)
        right_idx = self.items.index(right_item)

        # Record
        self.history.append({
            'left': left_item,
            'right': right_item,
            'winner': winner,
            'timestamp': datetime.now().isoformat(),
        })

        if winner == 'left':
            self.wins[left_idx, right_idx] += 1
        else:
            self.wins[right_idx, left_idx] += 1

        self.comparison_count += 1

        # Update model
        self._update_strengths()

        # Check convergence
        if self._check_convergence():
            self.results = self._get_rankings()
            self._confidence_intervals = self._compute_confidence_intervals()
            return None

        # Select next pair
        self._current_pair = self._select_next_pair()
        return self._current_pair

    def undo(self):
        """Undo last choice."""
        if not self.history:
            return None

        last = self.history.pop()
        left_idx = self.items.index(last['left'])
        right_idx = self.items.index(last['right'])

        if last['winner'] == 'left':
            self.wins[left_idx, right_idx] -= 1
        else:
            self.wins[right_idx, left_idx] -= 1

        self.comparison_count -= 1
        self.results = None
        self._confidence_intervals = None
        self._stable_count = 0

        # Recompute strengths
        self._update_strengths()

        # Restore pair
        self._current_pair = (last['left'], last['right'])
        return self._current_pair

    def is_done(self):
        return self.results is not None

    def progress(self):
        """Return (current, estimated_total)."""
        return (self.comparison_count, max(self.estimated_total, self.comparison_count + 1))

    def get_top_k(self):
        """Return top-k items once done, or current best guess."""
        if self.results is not None:
            return self.results[:self.top_k]
        return self._get_rankings()[:self.top_k]

    def get_confidence_intervals(self):
        """Return confidence intervals for each item's ranking."""
        if self._confidence_intervals is None:
            self._confidence_intervals = self._compute_confidence_intervals()
        return self._confidence_intervals

    def get_history(self):
        """Return full comparison history."""
        return list(self.history)

    def force_finish(self):
        """Force the tournament to finish with current rankings."""
        self.results = self._get_rankings()
        self._confidence_intervals = self._compute_confidence_intervals()
