from __future__ import annotations
import operator
from typing import Callable
from helper import softmax, mixed_distance_dict
from typing import Hashable, Dict
from ltm import SAMkNNLongTermMemory
from stm import SAMkNNShortTermMemory
class SAMkNNClassifier:
    """Self Adjusting Memory k-Nearest Neighbors (SAMkNN) [^1] for classification.

    SAM-kNN is a neighbors based online classifier designed to handle
    heterogeneous concept drift. To do so, it splits up its memory into Short
    Term Memory (STM) and Long Term Memory (LTM). The STM tracks the currently
    active concept and is continually resized to best represent it. Observations
    discarded from the STM are transferred to LTM. To limit the memory size
    without the need to discard observations, the LTM is regularly compressed.

    Parameters
    ----------
    n_neighbors.
        Number of neighbors to use for the underlying k nearest neighbor
        classifier.
    max_mem_size.
        Maximum size of the Short and Long Term Memory combined.
    max_ltm_size.
        Maximum size of the Long Term Memory. If LTM reaches this size, it is
        compressed.
    min_stm_size.
        Minimum size of the Short Term Memory. Smaller sizes will not be
        considered while calculating optimal STM size.
    weighted.
        Use distance weighted kNN. If turned off majority voting is used.
    softmax.
        Apply softmax on the output probabilities.
    dist_func.
        Distance function to use for the k nearest neighbor classifier.
    recalculate_stm_error.
        Disables a heuristic that incrementally computes the interleaved-test-
        then-train accuracy for the optimal STM size estimation. Activating this
        increases runtime but may result in slightly better model performance.
    sensitive_keys.
        The keys of features that are sensitives.
    balance_sensitive_neighbors.
        When classifying make sure that the neaerest neighbors are balanced regarding the keys

    Notes
    -----
    As the LTM compression mechanism uses kmeans, SAM-kNN only works with
    nummerical features and all datapoints are required to have the same features.
    """

    def __init__(
            self,
            n_neighbors: int = 20,
            max_mem_size: int = 100,
            max_ltm_size: int = 50,
            min_stm_size: int = 10,
            dist_func: Callable[[dict, dict], float] | None = None,
            weighted: bool = True,
            cleanup_every: int = 0,
            softmax_norm: bool = False,
            recalculate_stm_error: bool = False,
            sensitive_key: str | None = None,
            balance_sensitive_neighbors: bool = False,
            use_synthetic_stm: bool = False,
            smote_update_every: int = 5,
            smote_k_neighbors: int = 5,
            smote_random_state: int = 0,
    ):
        self.n_neighbors = n_neighbors
        self.max_mem_size = max_mem_size
        self.max_ltm_size = max_ltm_size
        self.min_stm_size = min_stm_size
        self.weighted = weighted
        self.classes: set[Hashable] = set()
        self.softmax_norm = softmax_norm
        self._cleanup_counter = cleanup_every
        self.recalculate_stm_error = recalculate_stm_error #was macht das???
        self.weights: dict[str, int] = {"stm": 0, "ltm": 0, "cm": 0}
        self.sensitive_key = sensitive_key  # which feature name is S (if None -> infer from first sample)
        self.balance_sensitive_neighbors = balance_sensitive_neighbors
        self.use_synthetic_stm = use_synthetic_stm
        self._learn_counter = 0
        self.smote_update_every = smote_update_every
        self.smote_k_neighbors = smote_k_neighbors
        self.smote_random_state = smote_random_state

        def drop_sensitive(features: Dict[str, float]) -> Dict[str, float]:
            if self.sensitive_key is None:
                return features
            return {k: v for k, v in features.items() if k != self.sensitive_key}

        if dist_func is None:
            self.dist_func = lambda a, b: mixed_distance_dict(drop_sensitive(a[0]), drop_sensitive(b[0]))
        else:
            self.dist_func = lambda a, b: dist_func(drop_sensitive(a[0]), drop_sensitive(b[0]))

        self.stm = SAMkNNShortTermMemory(
            n_neighbors=self.n_neighbors,
            dist_func=self.dist_func,
            min_stm_size=self.min_stm_size,
            weighted=self.weighted,
            recalculate_stm_error=self.recalculate_stm_error,
            sensitive_key=self.sensitive_key,
            balance_sensitive_neighbors=self.balance_sensitive_neighbors,
            use_synthetic=self.use_synthetic_stm,
            smote_k_neighbors=self.smote_k_neighbors,
            smote_random_state=self.smote_random_state,
        )
        self.ltm = SAMkNNLongTermMemory(
            n_neighbors=self.n_neighbors,
            dist_func=self.dist_func,
            sensitive_key=self.sensitive_key,
            balance_sensitive_neighbors=self.balance_sensitive_neighbors,
        )
    def learn_one(self, x: Dict[str, float], y: Hashable):
        self._learn_counter += 1
        self.classes.add(y)

        # Update memory weights
        for memory in self.weights.keys():
            self.weights[memory] += self.predict_one(x, memory=memory) == y

        self.stm.append((x, y))

        # Check if max memory size is exceeded
        if self.stm.size() + self.ltm.size() > self.max_mem_size:
            # Transfer items from STM to LTM and compress LTM
            n_items_to_transfer = self.max_ltm_size - self.ltm.size()
            for item in self.stm.pop_n(n_items_to_transfer):
                self.ltm.append(item)
            self.ltm.compress()

        # Clean LTM with (x, y)
        clean_dist = self.stm.get_clean_distance((x, y))
        self.ltm.clean((x, y), clean_dist)

        # Determine optimal STM size
        optimal_stm_size = self.stm.optimial_size()
        stm_size_changed = False
        if optimal_stm_size != self.stm.size():
            stm_size_changed = True
            # Transfer items to LTM to achieve optimal STM size
            n_items_to_transfer = self.stm.size() - optimal_stm_size
            new_ltm_items = []
            for item in self.stm.pop_n(n_items_to_transfer):
                new_ltm_items.append(item)

            # Clean new LTM samples before appending
            cleaned_new_ltm_items = [
                new_ltm_item
                for new_ltm_item in new_ltm_items
                if all(
                    [
                        clean_dist == 0
                        or new_ltm_item[1] != stm_item[1]
                        or self.dist_func(new_ltm_item, stm_item) > clean_dist
                        for stm_item, clean_dist in zip(
                        self.stm, map(self.stm.get_clean_distance, self.stm)
                    )
                    ]
                )
            ]

            self.ltm.append(cleaned_new_ltm_items)
        #refresh the syn-memory if the sizes has changed or the counter is done
        if self.use_synthetic_stm and (
                stm_size_changed or self._learn_counter % self.smote_update_every == 0
        ):
            self.stm.refresh_synthetic_balanced_sy()


    def predict_proba_one(self, x: Dict[str, float], memory: str | None = None) -> Dict[Hashable, float]:
        # Select memory by weight, if none is specified
        if memory is None:
            memory = max(self.weights, key=self.weights.get)

        # Make predictions using the selected memory
        if memory == "stm":
            nearest = self.stm.search((x, None))
        elif memory == "ltm":
            nearest = self.ltm.search((x, None))
        else:
            nearest_stm = self.stm.search((x, None))
            nearest_ltm = self.ltm.search((x, None))
            nearest = sorted(nearest_stm + nearest_ltm, key=operator.itemgetter(1))[
                : self.n_neighbors
            ]

        probas = {c: 0.0 for c in self.classes}

        # If no neighbors are found, return a uniform distribution
        if not nearest:
            return {cls: 1 / len(self.classes) for cls in self.classes}

        # If closest neighbor is exact match, assign it a probability of 1
        if nearest[0][1] == 0:
            return {cls: 1 if cls == nearest[0][0][1] else 0 for cls in self.classes}

        # Add up unnormalized probas
        for item, dist in nearest:
            probas[item[1]] += 1 / dist if self.weighted else 1

        if self.softmax_norm:
            return softmax(probas)

        return {cls: proba / sum(probas.values()) for cls, proba in probas.items()}


    def predict_one(self, x: Dict[str, float], memory: str | None = None) -> Hashable | None:
        proba = self.predict_proba_one(x, memory=memory)
        if not proba:
            return None
        return max(proba, key=proba.get)