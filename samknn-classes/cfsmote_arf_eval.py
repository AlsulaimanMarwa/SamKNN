from itertools import islice
import pandas as pd
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
import datetime
import random
import river.metrics
from samknn_classifier import RiverSAMKNNClassifier
from river import stream
from typing import Hashable


from river_fairness_metrics.metrics import Metrics as FMetrics
from river_fairness_metrics.equalized_odds import Equalized_FPR
from river_fairness_metrics.equal_opportunity import Equal_Opportunity
from river_fairness_metrics.demographic_parity import Demographic_Parity
from river_fairness_metrics.disparate_impact import Disparate_Impact
from river_fairness_metrics.fairness_unawareness import Fairness_Unawareness

BASE_DIR = Path(__file__).resolve().parent.parent
ADULT_PATH = BASE_DIR / "kathrin" / "data_iconip_marwa" / "adult" / "Marwa"
STUDENT_PATH = BASE_DIR / "kathrin" / "data_iconip_marwa" / "student_performance" / "Marwa"
OUTPUT_PATH = BASE_DIR / "Results"
ADULT_SCENARIOS = [
    "unmodified_to_debiased_25000",
    "unmodified_to_inflation_25000",
    "unmodified_to_less_marriage_25000",
    "unmodified_to_women_stem_25000",
]
STUDENT_SCENARIOS = [
    "unmodified_to_females_more_support_2000",
    "unmodified_to_grade_category_adjusted_2000",
    "unmodified_to_grade_inflation_2000",
    "unmodified_to_internet_era_2000",
    "unmodified_to_males_more_support_2000",
]

def _evaluate_fairness_stream(X_y, model, sens_att, switched_values, file_out, debug_first=False):
    eq_fpr = Equalized_FPR(sens_att)
    eq_opp = Equal_Opportunity(sens_att)
    disp_imp = Disparate_Impact(sens_att)
    dem_parity = Demographic_Parity(sens_att)
    fair_metrics = FMetrics((dem_parity, disp_imp, eq_opp, eq_fpr))

    acc = river.metrics.Accuracy()
    bAcc = river.metrics.BalancedAccuracy()
    recall = river.metrics.Recall(pos_val=1)
    kappa = river.metrics.CohenKappa()
    precision = river.metrics.Precision(pos_val=1)
    gmean = river.metrics.GeometricMean()
    f1 = river.metrics.F1(pos_val=1)
    metrics = river.metrics.base.Metrics((acc, bAcc, recall, kappa, precision, gmean, f1))

    indv_fairness = Fairness_Unawareness(sens_att)

    results = {m.__class__.__name__: [] for m in metrics}
    results.update({f.__class__.__name__: [] for f in fair_metrics})
    results["IndividualFairness"] = []
    results["time"] = []
    n=0

    for x, y in X_y:
        n += 1
        y_pred = model.predict_one(x)
        #print("y_true:", y, "y_pred:", y_pred)

        # individual fairness probe: flip sensitive attribute
        x_switched = x.copy()
        if x[sens_att[0]] == sens_att[1]:
            x_switched[sens_att[0]] = random.choice(switched_values)  # undeprived values
        else:
            x_switched[sens_att[0]] = sens_att[1]  # deprived value

        y_switched = model.predict_one(x_switched)

        model.learn_one(x, y)

        if y_pred is not None:
            fair_metrics.update(y_pred=y_pred, y_true=y, x=x)
            metrics.update(y_pred=y_pred, y_true=y)

            for m in metrics:
                results[m.__class__.__name__].append(m.get())
            for f in fair_metrics:
                results[f.__class__.__name__].append(f.get())

            indv_fairness.update(x=x, y_opp_pred=y_switched, y_pred=y_pred)
            results["IndividualFairness"].append(indv_fairness.get())
            results["time"].append(datetime.datetime.now())
        if debug_first and n % 1000 == 0:
            print("Reached 100 instances in first file")

    file_out = Path(file_out)
    file_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(file_out, index=False)

def run_samknn_dataset(file_in, file_out, *, target: str, converters: dict | None = None, limit: int | None = 500,
                       sensitive_attr: tuple[str, Hashable], switched_values: list[Hashable],
                       categorical_features: set[str] | None = None):
    converters = converters or {}

    def dataset_stream():
        for x, y in stream.iter_csv(file_in, target=target, converters=converters):
            x = {
                key: value
                for key, value in x.items()
                if not (isinstance(key, str) and key.startswith("Unnamed:"))
            }

            # Normalize binary labels to integers so metrics and model use the same class domain.
            try:
                y = int(float(y))
            except (TypeError, ValueError):
                pass

            # --- normalize sensitive attributes to 0/1 ---
            if "sex" in x:
                v = x["sex"]
                if v in ("F", "Female", 1, "1"):
                    x["sex"] = 1
                elif v in ("M", "Male", 0, "0"):
                    x["sex"] = 0

            if "race" in x:
                v = x["race"]
                if v in ("1", 1, "Black"):
                    x["race"] = 1
                elif v in ("0", 0, "White"):
                    x["race"] = 0
            yield x, y

    X_y = dataset_stream()
    if limit is not None:
        X_y = islice(X_y, 0, limit)

    model = RiverSAMKNNClassifier(
        n_neighbors=10,
        max_mem_size=100,
        max_ltm_size=50,
        min_stm_size=10,
        weighted=True,
        softmax_norm=False,
        recalculate_stm_error=False,
        sensitive_key="sex",
        balance_sensitive_neighbors=True,
        use_synthetic_stm=True,
        smote_update_every=5,
        smote_k_neighbors=None,
        smote_random_state=0,
        categorical_features=categorical_features,
    )
    sens_att = sensitive_attr
    _evaluate_fairness_stream(X_y, model, sens_att, switched_values, file_out, debug_first=True)

def run_test_samknn(repetition: int):
    print(f"Run {repetition} (SAMKNN)")


    for scenario in STUDENT_SCENARIOS:
        in_file = STUDENT_PATH / scenario / f"run_{repetition}.csv"
        out_file = OUTPUT_PATH / "student_performance"  / scenario / f"run_{repetition}.csv"

        if not in_file.exists():
            raise FileNotFoundError(f"Missing input file: {in_file}")

        print(f"Student scenario: {scenario}")
        run_samknn_dataset(
            in_file,
            out_file,
            target="G3",
            #converters = {'age': float, 'income': float, 'education-num': float, 'capital-gain': float, 'capital-loss': float, 'hours-per-week': float}
            converters={"absences": float},
            limit=None,
            sensitive_attr=("sex", 1),
            switched_values=[0],
            categorical_features={
                "school",
                "age",
                "address",
                "famsize",
                "Pstatus",
                "Medu",
                "Fedu",
                "Mjob",
                "Fjob",
                "reason",
                "guardian",
                "traveltime",
                "studytime",
                "failures",
                "schoolsup",
                "famsup",
                "paid",
                "activities",
                "nursery",
                "higher",
                "internet",
                "romantic",
                "famrel",
                "freetime",
                "goout",
                "Dalc",
                "Walc",
                "health"
            }
        )

    for scenario in ADULT_SCENARIOS:
        in_file = ADULT_PATH / scenario / f"run_{repetition}.csv"
        out_file = OUTPUT_PATH / "adult" / scenario / f"run_{repetition}.csv"

        if not in_file.exists():
            raise FileNotFoundError(f"Missing input file: {in_file}")

        print(f"Adult/Marwa scenario: {scenario}")
        run_samknn_dataset(in_file,
                           out_file,
                           target="income",
                           limit=None,
                           #converters = { 'income': float},
                           sensitive_attr=("sex", 1),
                           switched_values=[0],
                           categorical_features={
                               "hours-per-week",
                               "capital-loss",
                               "capital-gain",
                               "education-num",
                               "age",
                               "workclass",
                               "education",
                               "marital-status",
                               "occupation",
                               "relationship",
                               "race",
                               "native-country",
                           })

if __name__ == "__main__":
    i = int(sys.argv[1])
    run_test_samknn(i)
