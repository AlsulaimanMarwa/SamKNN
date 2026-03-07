from __future__ import annotations
from typing import Tuple, Callable
from collections import defaultdict, Counter
import random
import datetime
from helper import mixed_distance_dict, _to_numeric
from typing import Any, Hashable, Dict
from memory_base import SAMkNNMemory
class SAMkNNLongTermMemory(SAMkNNMemory):
    def __init__(self, n_neighbors: int, dist_func: Callable, sensitive_key: str | None = None,
                 balance_sensitive_neighbors: bool = False, categorical_features: set[str] | None = None,):
        # Initialize parent class FIRST
        super().__init__(
            n_neighbors=n_neighbors,
            dist_func=dist_func,
            sensitive_key=sensitive_key,
            balance_sensitive_neighbors=balance_sensitive_neighbors,
            categorical_features=categorical_features
        )

    def clean(self, item: Tuple[Dict[str, float], Hashable], clean_dist: float):
        if clean_dist == 0:
            return

        # Clean items
        self.items = [
            cur_item
            for cur_item in self.items
            if cur_item[1] != item[1] or self.dist_func(cur_item, item) > clean_dist
        ]

        self.last_search_item = None

    def compress(self, *, n_iters: int = 3) -> None:
        """Prototype compression per (y,s) if sensitive_key exists, else per y."""
        self.last_search_item = None
        if len(self.items) <= 1:
            return

        s_key = self.sensitive_key
        # group by (y,s) if possible else by y only
        groups: dict[tuple[Any, Any], list[tuple[dict, Any]]] = defaultdict(list)
        for x, y in self.items:
            if s_key is None:
                g = (y, None)
            else:
                if s_key not in x:
                    continue
                s = int(x[s_key])
                if s not in (0, 1):
                    continue
                g = (y, s)
            groups[g].append((x, y))

        new_items: list[tuple[dict, Any]] = []

        for (y, s), pts in groups.items():
            n = len(pts)
            # if a (y,s)-Group only has one point -> no compression
            if n <= 1:
                for x, yy in pts:
                    xx = x.copy()
                    if s_key is not None:
                        xx[s_key] = float(s)
                    new_items.append((xx, yy))
                continue

            n_clusters = max(1, n // 2)
            centers = [pts[i][0].copy() for i in random.sample(range(n), k=n_clusters)]

            for _ in range(max(1, int(n_iters))):
                clusters = [[] for _ in range(n_clusters)]

                for x, _yy in pts:
                    best_j, best_d = 0, float("inf")
                    for j, c in enumerate(centers):
                        d = mixed_distance_dict(x, c, sensitive_key=s_key, drop_sensitive=True, categorical_features=self.categorical_features)
                        if d < best_d:
                            best_d, best_j = d, j
                    clusters[best_j].append(x)
                # take a random real point as a center if the receptive field is empty
                centers = [
                    (pts[random.randrange(n)][0].copy() if not cl else self._prototype_median_mode(cl))
                    for cl in clusters
                ]
                if s_key is not None:
                    for c in centers:
                        c[s_key] = float(s)

            for c in centers:
                if s_key is not None:
                    c[s_key] = float(s)
                new_items.append((c, y))

        self.items = new_items


    def _prototype_median_mode(self, xs: list[dict]) -> dict:
        if not xs:
            return {}

        s_key = self.sensitive_key
        keys = set().union(*(x.keys() for x in xs))
        if s_key is not None and s_key in keys:
            keys.remove(s_key)

        proto: dict = {}

        for k in keys:
            vals = [x[k] for x in xs if k in x]
            if not vals:
                continue

            nums = []
            has_dt = any(isinstance(v, datetime.datetime) for v in vals)
            has_date = any(isinstance(v, datetime.date) and not isinstance(v, datetime.datetime) for v in vals)

            for v in vals:
                nv = _to_numeric(v)
                if nv is None:
                    nums = None
                    break
                nums.append(nv)

            if nums is not None:
                nums.sort()
                if len(nums) % 2 == 1:
                    m = nums[len(nums) // 2]
                else:
                    m = 0.5 * (nums[len(nums)//2 - 1] + nums[len(nums)//2])

                if has_dt:
                    proto[k] = datetime.datetime.fromtimestamp(m)
                elif has_date:
                    proto[k] = datetime.datetime.fromtimestamp(m).date()
                else:
                    proto[k] = float(m)
                    
            if k in self.categorical_features:
                c = Counter(vals)
                proto[k] = c.most_common(1)[0][0]
                continue

        return proto

