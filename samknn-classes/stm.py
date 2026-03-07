from __future__ import annotations
import operator
import math
from typing import List, Tuple, Callable
from collections import defaultdict, Counter
from typing import Any, Hashable, Dict
from memory_base import SAMkNNMemory
import random
from helper import mixed_distance_dict, _to_numeric

class SAMkNNShortTermMemory(SAMkNNMemory):
    def __init__(self, n_neighbors: int, dist_func: Callable, min_stm_size: int, weighted: bool,
                 recalculate_stm_error: bool, sensitive_key: str | None = None,
                 balance_sensitive_neighbors: bool = False, use_synthetic: bool = False, smote_k_neighbors: int = 5,
                 smote_random_state: int = 0, categorical_features: set[str] | None = None):
        super().__init__(
            n_neighbors=n_neighbors,
            dist_func=dist_func,
            sensitive_key=sensitive_key,
            balance_sensitive_neighbors=balance_sensitive_neighbors,
            categorical_features=categorical_features,
        )

        self.min_stm_size = min_stm_size
        self.weighted = weighted
        self.recalculate_stm_error = recalculate_stm_error
        self.prediction_histories: dict[int, list[bool]] = {}

        self.use_synthetic = use_synthetic
        self.smote_k_neighbors = smote_k_neighbors
        self.smote_random_state = smote_random_state
        self.synthetic_items: List[Tuple[Dict[str, float], Hashable]] = []
        self._num_minmax = {}

    def pop_n(self, n: int):
        # Invalidate cache and prediction histories as items are changed
        self.last_search_item = None
        self.prediction_histories = {}

        for _ in range(n):
            yield self.items.pop(0)

    def __iter__(self):
        yield from self.items

    def get_clean_distance(self, item: Tuple[Dict[str, float], Hashable]) -> float:
        # As item itself is included in window, search for self.n_neighbors+1 neighbors
        nearest = self.search(item, n_neighbors=self.n_neighbors + 1)

        def _same_label(a: Hashable, b: Hashable) -> bool:
            if a == b:
                return True
            try:
                return math.isnan(a) and math.isnan(b)
            except TypeError:
                return False

        same_label_distances = [
            item_dist[1] for item_dist in nearest if _same_label(item_dist[0][1], item[1])
        ]
        if not same_label_distances:
            return 0.0

        return max(same_label_distances)

    def partial_interleaved_test_train_error(self, size: int) -> float:
        start_idx = len(self.items) - size

        # Fall A: Usecase: optimial_size() wird nochmal aufgerufen, mit derselben size = 3, Prediction history für start_idx = 2 existiert schon
        if start_idx in self.prediction_histories.keys():
            # Make new prediction and append to prediction history
            item = self.items[-1]
            items_distances = ((p, self.dist_func(item, p)) for p in self.items[start_idx:-1])
            nearest = sorted(items_distances, key=operator.itemgetter(1))[: self.n_neighbors]

            if nearest[0][1] <= 1e-12:
                prediction = nearest[0][0][1]
            else:
                probas = defaultdict(float)
                for item, dist in nearest:
                    probas[item[1]] += 1 / dist if self.weighted else 1
                prediction = max(probas, key=probas.__getitem__)
            self.prediction_histories[start_idx].append(prediction == item[1])

        # Fall B: Shift-Heuristik, Size ist gleich
        elif start_idx - 1 in self.prediction_histories.keys() and not self.recalculate_stm_error:
            # Use prediction history with start shifted by 1
            self.prediction_histories[start_idx] = self.prediction_histories[start_idx - 1]
            del self.prediction_histories[start_idx - 1]
            self.prediction_histories[start_idx].pop(0)

            # Make new prediction and append to prediction history
            item = self.items[-1]
            items_distances = ((p, self.dist_func(item, p)) for p in self.items[start_idx:-1])
            nearest = sorted(items_distances, key=operator.itemgetter(1))[: self.n_neighbors]

            if nearest[0][1] == 0:
                prediction = nearest[0][0][1]
            else:
                probas = defaultdict(lambda: 0)
                for item, dist in nearest:
                    probas[item[1]] += 1 / dist if self.weighted else 1
                prediction = max(probas, key=probas.__getitem__)
            self.prediction_histories[start_idx].append(prediction == item[1])

        # Fall C: Alles neu berechnen (from scratch), Es gibt noch keine passende History (oder man will exakt rechnen).
        else:
            # Generate new Prediction history from scratch
            self.prediction_histories[start_idx] = []
            for cur_idx in range(start_idx + 1, len(self.items)):
                item = self.items[cur_idx]
                items_distances = (
                    (p, self.dist_func(item, p)) for p in self.items[start_idx:cur_idx]
                )
                nearest = sorted(items_distances, key=operator.itemgetter(1))[: self.n_neighbors]
                if nearest[0][1] == 0:
                    prediction = nearest[0][0][1]
                else:
                    probas = defaultdict(lambda: 0)
                    for item, dist in nearest:
                        probas[item[1]] += 1 / dist if self.weighted else 1

                    prediction = max(probas, key=probas.__getitem__)
                self.prediction_histories[start_idx].append(prediction == item[1])

        # Return interleaved-test-then-train accuracy
        return sum(self.prediction_histories[start_idx]) / len(self.prediction_histories[start_idx])

    def optimial_size(self)-> int:
        # Generate candidate sizes using repeated halving
        candidate_sizes = []
        cur_candidate_size = len(self.items)
        while cur_candidate_size > self.min_stm_size:
            candidate_sizes.append(cur_candidate_size)
            cur_candidate_size //= 2

        # If no alternative candidate sizes exist, return the current size
        if len(candidate_sizes) <= 1:
            return self.size()

        # Score all candidate sizes
        candidate_sizes_scores = {
            size: self.partial_interleaved_test_train_error(size) for size in candidate_sizes
        }

        # Delete unused prediction histories if necessary ?????
        if self.recalculate_stm_error:
            for start_idx in list(self.prediction_histories.keys()):
                if len(self.items) - start_idx not in candidate_sizes:
                    del self.prediction_histories[start_idx]

        best_size = max(candidate_sizes_scores, key=candidate_sizes_scores.get)
        return best_size
    def search(self, item, n_neighbors: int | None = None):
        # temporarily swap self.items to include synthetic points
        if not self.use_synthetic or not self.synthetic_items:
            return super().search(item, n_neighbors=n_neighbors)

        original = self.items
        try:
            self.items = original + self.synthetic_items
            # invalidate cache because search space changed
            self.last_search_item = None
            return super().search(item, n_neighbors=n_neighbors)
        finally:
            self.items = original

    def _generate_synthetic_from_bucket(
            self,
            bucket,
            n_new: int,
            k: int,
            force_sensitive_value: float,
    ):
        new_points = []

        if len(bucket) < 2:
            return new_points
        # Make results reproducible if desired
        if self.smote_random_state is not None:
            random.seed(self.smote_random_state)

        for _ in range(n_new):
            base_idx = random.randrange(len(bucket))
            x_base, y = bucket[base_idx]

            k_eff = min(int(k), len(bucket) - 1)
            if k_eff < 1:
                continue

            # --- compute distances to all others in bucket (normalized + min/max tracking) ---
            dists: list[tuple[float, int]] = []
            for j, (xj, _yj) in enumerate(bucket):
                if j == base_idx:
                    continue
                d = mixed_distance_dict(
                    x_base, xj,
                    num_minmax=self._num_minmax,  # UPDATED in-place
                    sensitive_key=self.sensitive_key,
                    drop_sensitive=True,  # distance ignores sensitive attribute
                    categorical_features=self.categorical_features,
                )
                dists.append((d, j))

            dists.sort(key=lambda t: t[0])
            nn_indices = [j for _d, j in dists[:k_eff]]
            if not nn_indices:
                continue

            # choose one neighbor for interpolation
            nei_idx = random.choice(nn_indices)
            x_nei, _ = bucket[nei_idx]

            # pool for categorical mode: base + ALL k neighbors
            neighbor_xs = [bucket[j][0] for j in nn_indices]
            categorical_pool= [x_base] + neighbor_xs

            x_new: Dict[str, Any] = {}
            keys = set(x_base.keys()) | set(x_nei.keys())

            for key in keys:
                if key == self.sensitive_key:
                    continue

                v_base = x_base.get(key, None)
                v_nei = x_nei.get(key, None)

                # missing handling
                if v_base is None and v_nei is None:
                    continue
                if v_base is None:
                    x_new[key] = v_nei
                    continue
                if v_nei is None:
                    x_new[key] = v_base
                    continue

                # numeric/date/datetime interpolation
                a_num = _to_numeric(v_base)
                b_num = _to_numeric(v_nei)

                # categorical/other -> MODE over (base + k neighbors)
                def _mode_with_random_tie(values: list[Any]) -> Any:
                    c = Counter(values)
                    top = max(c.values())
                    # Gib all values with max frequency and break ties randomly.
                    tied = [v for v, cnt in c.items() if cnt == top]
                    return random.choice(tied)

                if key in self.categorical_features:
                    vals = []
                    for xd in categorical_pool:
                        vv = xd.get(key, None)
                        if vv is not None:
                            vals.append(vv)
                    if vals:
                        x_new[key] = _mode_with_random_tie(vals)
                    continue

                if a_num is not None and b_num is not None:
                    gap = random.random()
                    x_new[key] = float(a_num + gap * (b_num - a_num))
                    continue

                vals = []
                for xd in categorical_pool:
                    vv = xd.get(key, None)
                    if vv is not None:
                        vals.append(vv)
                if vals:
                    x_new[key] = _mode_with_random_tie(vals)

            x_new[self.sensitive_key] = force_sensitive_value

            new_points.append((x_new, y))

        return new_points

    def refresh_synthetic_balanced_sy(
            self,
            *,
            max_synth_ratio: float = 0.5,
            minority_threshold: float = 0.245,  # <= 0.25 for 4 groups
            require_all_four_groups: bool = False,  # if True: only balance if all 4 exist in real data
    ) -> None:
        """
        Global intersectional balancing for (y,s) groups:
          target: each (y,s) group roughly 25% of total (real+synthetic)
          stop when min_group_share >= minority_threshold
        """
        self.synthetic_items = []

        if not getattr(self, "use_synthetic", False):
            return
        if self.sensitive_key is None:
            raise ValueError("sensitive_key must be set for (y,s) synthesis.")
        if len(self.items) < 2:
            return

        s_key = self.sensitive_key
        max_total_syn = int(max_synth_ratio * len(self.items))

        # --- Buckets of REAL samples by (y,s) ---
        buckets: dict[tuple[Hashable, int], list[tuple[Dict[str, Any], Hashable]]] = defaultdict(list)
        for x, y in self.items:
            if s_key not in x:
                continue
            s = int(x[s_key])
            if s not in (0, 1):
                raise ValueError(f"Sensitive feature must be 0/1, got {x[s_key]}")
            buckets[(y, s)].append((x, y))

        # Decide which groups to balance
        present_groups = [g for g, pts in buckets.items() if len(pts) > 0]

        if require_all_four_groups:
            # only proceed if all 4 combos exist in real data
            all_four = {(0, 0), (0, 1), (1, 0), (1, 1)}
            if not all_four.issubset(set(present_groups)):
                self.last_search_item = None
                return
            groups_to_balance = list(all_four)
        else:
            # balance only groups that exist in real STM (prevents fabricating missing groups)
            if len(present_groups) < 2:
                self.last_search_item = None
                return
            groups_to_balance = present_groups

        # Synthetic counts per group
        syn_counts: dict[tuple[Hashable, int], int] = defaultdict(int)

        def min_group_share() -> float:
            eff_counts = []
            total = 0
            for g in groups_to_balance:
                c = len(buckets[g]) + syn_counts[g]
                eff_counts.append(c)
                total += c
            if total <= 0:
                return 0.0
            return min(eff_counts) / total

        def eff_count(g: tuple[Hashable, int]) -> int:
            return len(buckets[g]) + syn_counts[g]

        # --- Generate one-by-one ---
        while len(self.synthetic_items) < max_total_syn and min_group_share() < minority_threshold:
            g_min = min(groups_to_balance, key=eff_count)
            y_min, s_min = g_min
            bucket_min = buckets[g_min]

            # need >=2 REAL points in that group for SMOTE interpolation
            if len(bucket_min) < 2:
                break

            k = min(int(self.smote_k_neighbors), len(bucket_min) - 1)
            if k < 1:
                break

            new_pts = self._generate_synthetic_from_bucket(
                bucket_min,
                n_new=1,
                k=k,
                force_sensitive_value=float(s_min),
            )
            if not new_pts:
                break

            self.synthetic_items.extend(new_pts)
            syn_counts[g_min] += len(new_pts)  # ~1

        self.last_search_item = None
