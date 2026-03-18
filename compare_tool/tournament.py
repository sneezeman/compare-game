"""
Partial merge-sort tournament to identify top-3 epochs.
Supports undo via an explicit state stack.
"""

import math
from datetime import datetime


class Tournament:
    def __init__(self, items):
        """items: list of epoch indices to rank."""
        self.items = list(items)
        self.n = len(self.items)
        # Choice history for logging
        self.history = []
        # sorted sublists that get merged pairwise
        self.sublists = [[x] for x in self.items]
        self.top_k = min(1, self.n)
        # estimate total comparisons
        self.estimated_total = self._estimate_comparisons()
        self.comparison_count = 0
        self.results = None  # set when done

        # Current merge operation
        self._left = []
        self._right = []
        self._merged = []
        self._li = 0
        self._ri = 0

        # Merge state
        self._merge_queue = []  # pairs of sublists to merge
        self._new_sublists = []
        self._leftover = None
        self._prepare_merge_round()

        # Undo stack: each entry stores the full state before a choice
        self._undo_stack = []

        # Advance to first comparison
        self._advance_to_next_comparison()

    def _estimate_comparisons(self):
        """Estimate comparisons for partial merge sort.
        With top_k=1, the last merge of two n/2-sized sorted lists
        needs only top_k comparisons instead of ~n. Earlier rounds
        are full merges: each round merges pairs fully."""
        n = self.n
        if n <= self.top_k:
            return max(0, n - 1)
        rounds = int(math.ceil(math.log2(max(n, 2))))
        # Each round except the last does ~n/2 comparisons on average
        # Last round: only top_k comparisons (short-circuit)
        comparisons = (rounds - 1) * n // 2 + self.top_k
        return max(n - 1, comparisons)

    def _prepare_merge_round(self):
        """Pair up sublists for a merge round."""
        self._merge_queue = []
        i = 0
        while i + 1 < len(self.sublists):
            self._merge_queue.append((self.sublists[i], self.sublists[i + 1]))
            i += 2
        # odd one out stays as-is
        self._leftover = self.sublists[i] if i < len(self.sublists) else None
        self._new_sublists = []

        if self._merge_queue:
            self._start_next_merge()

    def _start_next_merge(self):
        """Start merging the next pair from the queue."""
        if not self._merge_queue:
            return False
        left, right = self._merge_queue.pop(0)
        self._left = left
        self._right = right
        self._merged = []
        self._li = 0
        self._ri = 0
        return True

    def _advance_to_next_comparison(self):
        """Advance state until we need user input or are done."""
        while True:
            # Check if current merge is done or can be short-circuited
            if self._li >= len(self._left):
                self._merged.extend(self._right[self._ri:])
                self._finish_current_merge()
                if not self._try_next():
                    return
                continue
            if self._ri >= len(self._right):
                self._merged.extend(self._left[self._li:])
                self._finish_current_merge()
                if not self._try_next():
                    return
                continue

            # We already have enough top elements in this merge
            if len(self._merged) >= self.top_k and len(self._merge_queue) == 0 and len(self._new_sublists) == 0:
                # Only short-circuit on the very last merge
                remaining = (self._left[self._li:] + self._right[self._ri:])
                self._merged.extend(remaining)
                self._finish_current_merge()
                if not self._try_next():
                    return
                continue

            # Need user comparison
            return

    def _finish_current_merge(self):
        """Mark current merge as complete."""
        self._new_sublists.append(self._merged)

    def _try_next(self):
        """Try to start the next merge or round. Returns True if more work, False if done."""
        if self._merge_queue:
            self._start_next_merge()
            return True

        # Round complete
        if self._leftover:
            self._new_sublists.append(self._leftover)
            self._leftover = None

        self.sublists = self._new_sublists
        self._new_sublists = []

        if len(self.sublists) <= 1:
            # Done
            self.results = self.sublists[0] if self.sublists else []
            return False

        self._prepare_merge_round()
        return True

    def is_done(self):
        return self.results is not None

    def current_pair(self):
        """Return the two items to compare, or None if done."""
        if self.is_done():
            return None
        if self._li < len(self._left) and self._ri < len(self._right):
            return (self._left[self._li], self._right[self._ri])
        return None

    def get_state(self):
        """Snapshot full state for undo."""
        import copy
        return {
            'sublists': copy.deepcopy(self.sublists),
            'merge_queue': copy.deepcopy(self._merge_queue),
            'leftover': copy.deepcopy(self._leftover) if self._leftover else None,
            'new_sublists': copy.deepcopy(self._new_sublists),
            'left': list(self._left),
            'right': list(self._right),
            'merged': list(self._merged),
            'li': self._li,
            'ri': self._ri,
            'comparison_count': self.comparison_count,
            'results': copy.deepcopy(self.results) if self.results else None,
        }

    def restore_state(self, state):
        """Restore from a snapshot."""
        self.sublists = state['sublists']
        self._merge_queue = state['merge_queue']
        self._leftover = state['leftover']
        self._new_sublists = state['new_sublists']
        self._left = state['left']
        self._right = state['right']
        self._merged = state['merged']
        self._li = state['li']
        self._ri = state['ri']
        self.comparison_count = state['comparison_count']
        self.results = state['results']

    def choose(self, winner):
        """
        winner: 'left' or 'right' — which item wins this comparison.
        Returns the next pair or None if done.
        """
        if self.is_done():
            return None

        # Save state for undo
        self._undo_stack.append(self.get_state())

        # Log the choice
        pair = self.current_pair()
        if pair:
            self.history.append({
                'left': pair[0],
                'right': pair[1],
                'winner': winner,
                'timestamp': datetime.now().isoformat(),
            })

        self.comparison_count += 1

        if winner == 'left':
            self._merged.append(self._left[self._li])
            self._li += 1
        else:
            self._merged.append(self._right[self._ri])
            self._ri += 1

        self._advance_to_next_comparison()
        return self.current_pair()

    def undo(self):
        """Undo last choice. Returns the restored pair or None if nothing to undo."""
        if not self._undo_stack:
            return None
        state = self._undo_stack.pop()
        self.restore_state(state)
        if self.history:
            self.history.pop()
        return self.current_pair()

    def progress(self):
        """Return (current, estimated_total)."""
        return (self.comparison_count, max(self.estimated_total, self.comparison_count + 1))

    def get_top_k(self):
        """Return top-k results once done."""
        if self.results is None:
            return None
        return self.results[:self.top_k]

    def get_history(self):
        """Return full comparison history."""
        return list(self.history)

    def get_confidence_intervals(self):
        """Not available for merge-sort tournament."""
        return {}

    def force_finish(self):
        """Force tournament to finish with current best ordering."""
        if self.results is not None:
            return
        # Merge remaining sublists as-is (interleave without comparisons)
        flat = []
        for sl in self.sublists:
            flat.extend(sl)
        # Add any in-progress merge
        remaining = self._merged + self._left[self._li:] + self._right[self._ri:]
        if remaining and remaining != flat:
            flat = remaining
            for sl in self._new_sublists:
                flat.extend(sl)
            if self._leftover:
                flat.extend(self._leftover)
        self.results = flat
