from __future__ import annotations
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from classifier import SAMkNNClassifier
from river import base


class RiverSAMKNNClassifier(base.Classifier):
    def __init__(
        self,
        *,
        n_neighbors=10,
        max_mem_size=100,
        max_ltm_size=50,
        min_stm_size=10,
        dist_func=None,
        weighted=True,
        softmax_norm=False,
        recalculate_stm_error=False,
        sensitive_key=None,
        balance_sensitive_neighbors=False,
        use_synthetic_stm=False,
        smote_update_every=5,
        smote_k_neighbors=None,
        smote_random_state=0,
        categorical_features: set[str] | None = None,

    ):
        self.core = SAMkNNClassifier(n_neighbors=n_neighbors, max_mem_size=max_mem_size, max_ltm_size=max_ltm_size,
                                     min_stm_size=min_stm_size, dist_func=dist_func, weighted=weighted,
                                     softmax_norm=softmax_norm, recalculate_stm_error=recalculate_stm_error,
                                     sensitive_key=sensitive_key,
                                     balance_sensitive_neighbors=balance_sensitive_neighbors,
                                     use_synthetic_stm=use_synthetic_stm, smote_update_every=smote_update_every,
                                     smote_k_neighbors=smote_k_neighbors, smote_random_state=smote_random_state,
                                     categorical_features=categorical_features)

    def learn_one(self, x, y):
        self.core.learn_one(x, y)
        return self

    def predict_one(self, x):
        return self.core.predict_one(x)

    def predict_proba_one(self, x):
        return self.core.predict_proba_one(x)


"""
def evaluate_stream_with_drift_report(
    model,
    stream: Iterator[Tuple[Dict[str, float], Hashable, int]],
    *,
    warmup: int = 300,
    report_every: int = 5000,
):
    
    #Prequential evaluation: predict -> compare -> learn
    #Also reports accuracy by gender (S=0 vs S=1) to see fairness impact under drift.
   
    correct = 0
    seen = 0
    #{0. gender: Anzahl, 1.gender: Anzahl}
    correct_s = {0: 0, 1: 0}
    seen_s = {0: 0, 1: 0}

    cur_year = None
    year_counts = 0

    for x, y, year in stream:
        s = int(x["gender"])

        if cur_year is None:
            cur_year = year
        if year != cur_year:
            # year boundary -> drift checkpoint
            eval_seen = seen - warmup
            acc = correct / eval_seen if eval_seen > 0 else 0.0
            acc0 = (correct_s[0] / seen_s[0]) if seen_s[0] else 0.0
            acc1 = (correct_s[1] / seen_s[1]) if seen_s[1] else 0.0
            print(f"\n=== Year changed: {cur_year} -> {year} | overall_acc={acc:.4f} | acc_S0={acc0:.4f} | acc_S1={acc1:.4f} ===")
            cur_year = year
            year_counts = 0

        # predict (skip early warmup)
        if seen >= warmup:
            y_hat = model.predict_one(x)  # uses your memory selection
            if y_hat == y:
                correct += 1
                correct_s[s] += 1
            seen_s[s] += 1

        # learn
        model.learn_one(x, y)
        seen += 1
        year_counts += 1

        if seen % report_every == 0:
            eval_seen = seen - warmup
            acc = correct / eval_seen if eval_seen > 0 else 0.0
            acc0 = (correct_s[0] / seen_s[0]) if seen_s[0] else 0.0
            acc1 = (correct_s[1] / seen_s[1]) if seen_s[1] else 0.0
            print(
                f"progress@{seen:,} | acc={acc:.4f} | acc_S0={acc0:.4f} | acc_S1={acc1:.4f} | "
                f"STM={model.stm.size():,} LTM={model.ltm.size():,} weights={model.weights}"
            )

    eval_seen = seen - warmup
    acc = correct / eval_seen if eval_seen > 0 else 0.0
    acc0 = (correct_s[0] / seen_s[0]) if seen_s[0] else 0.0
    acc1 = (correct_s[1] / seen_s[1]) if seen_s[1] else 0.0
    print(f"\nFINAL | seen={seen:,} | acc={acc:.4f} | acc_S0={acc0:.4f} | acc_S1={acc1:.4f}")
    return acc


def synthetic_gender_drift_stream(
        n_total: int,
        *,
        p_gender_1: float = 0.5,  # proportion of gender=1 in the population
        noise: float = 0.6,  # gaussian noise level
        bias_phase2: float = 0.0,  # additional bias against gender=1 in phase 2 (0.0 = none)
) -> Iterator[Tuple[Dict[str, float], int, int]]:

    n0 = n_total // 3
    n1 = n_total // 3
    n2 = n_total - n0 - n1

    def sample_point():
        g = 1 if random.random() < p_gender_1 else 0
        x1 = random.gauss(0.0, noise)
        x2 = random.gauss(0.0, noise)
        return g, x1, x2

    # --- Phase 0 (2018): y depends on x1 ---
    for _ in range(n0):
        g, x1, x2 = sample_point()
        score = x1  # concept rule
        y = 1 if score > 0 else 0
        yield {"gender": float(g), "x1": x1, "x2": x2}, y, 2018

    # --- Phase 1 (2019): concept drift: y depends on x2 ---
    for _ in range(n1):
        g, x1, x2 = sample_point()
        score = x2  # drifted rule
        y = 1 if score > 0 else 0
        yield {"gender": float(g), "x1": x1, "x2": x2}, y, 2019

    # --- Phase 2 (2020): concept back + optional fairness drift ---
    for _ in range(n2):
        g, x1, x2 = sample_point()
        score = x1
        # fairness drift: bias changes only in this phase
        # if bias_phase2 > 0, then gender=1 is pushed towards y=0 more often
        if g == 1:
            score -= bias_phase2
        y = 1 if score > 0 else 0
        yield {"gender": float(g), "x1": x1, "x2": x2}, y, 2020

"""
