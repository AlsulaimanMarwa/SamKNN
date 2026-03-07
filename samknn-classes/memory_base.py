from __future__ import annotations
import heapq
import operator
from typing import List, Tuple, Callable, Hashable, Dict

class SAMkNNMemory:
    def __init__(self, n_neighbors: int,
                 dist_func: Callable,
                 sensitive_key: str | None = None,
                 balance_sensitive_neighbors: bool = False,
                 categorical_features: set[str] | None = None,
                 ):
        self.n_neighbors = n_neighbors
        self.dist_func = dist_func

        self.items: List[Tuple[Dict[str, float], Hashable]] = []
        self.last_search_item: Tuple[Dict[str, float], Hashable] | None = None
        self.last_search_result: List[Tuple[Tuple[Dict[str, float], Hashable], float]] | None = None

        self.sensitive_key = sensitive_key
        self.balance_sensitive_neighbors = balance_sensitive_neighbors
        self.categorical_features = set(categorical_features or [])

    def append(self, item):
        if isinstance(item, list):
            self.items += item
        else:
            self.items.append(item)
        self.last_search_item = None
        self.last_search_result = None

    def size(self) -> int:
        return len(self.items)

    def search(self, item: Tuple[Dict[str, float], Hashable], n_neighbors: int | None = None):
        # If search result is cached, return it
        if self.last_search_item is not None and self.last_search_item == item:
            return self.last_search_result or []

        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        # If no balancing requested (or sensitive_key unknown) -> normal behavior
        if (not self.balance_sensitive_neighbors) or (self.sensitive_key is None):
            result=heapq.nsmallest(n_neighbors, ((p, self.dist_func(item, p)) for p in self.items), key=lambda x: x[1],)
            self.last_search_item = item
            self.last_search_result = result
            return result

        # Balanced selection by sensitive attribute S (binary 0/1)
        sensitive_key = self.sensitive_key
        k0 = n_neighbors // 2
        k1 = n_neighbors - k0

        group0: list = []
        group1: list = []
        other: list = []

        for p in self.items:
            d = self.dist_func(item, p)
            feat = p[0]
            s = feat.get(sensitive_key)

            if s == 0:
                group0.append((p, d))
            elif s == 1:
                group1.append((p, d))
            else:
                other.append((p, d))  # missing S -> fallback pool

        picked: list = []
        group0.sort(key=operator.itemgetter(1))
        group1.sort(key=operator.itemgetter(1))
        picked.extend(group0[:k0])
        picked.extend(group1[:k1])

        # If one group has too few points, fill from the other group, then from 'other'
        if len(picked) < n_neighbors:
            remaining = n_neighbors - len(picked)
            # candidates not already used
            rest = group0[k0:] + group1[k1:] + other
            rest.sort(key=operator.itemgetter(1))
            picked.extend(rest[:remaining])

        picked.sort(key=operator.itemgetter(1))

        self.last_search_item = item
        self.last_search_result = picked
        return picked
