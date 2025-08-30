
import numpy as np
from evaluation.utils import get_pickle_path, get_results, get_MAE_from_results
from tabulate import tabulate
import pandas as pd

def get_metrics_from_results(results):
    #return results["FFT"]["SNR"]
    return np.abs(np.array(results["Peak"]["hr_label"]) - np.array(results["Peak"]["hr_pred"]))

if __name__ == "__main__":
   

    datasets = ["validation", "test"]
    models = ["EfficientPhys",
                "Physnet",
                "TSCAN",
                "RythmFormer",
                "PhysFormer", 
                "Physnet_Quantile", 
                "Physnet_NegLogLikelihood"]

    overall_table = []
    fitzpatrick_values_label = [1, 2, 3, 4, 5]

    for model in models:

        metrics_by_fitzpatrick = {}

        for dataset in datasets:
            curr_path = get_pickle_path("DeadAlive", dataset, model)
            results = get_results(curr_path, dataset=dataset)

            fitzpatrick_values = results["fitzpatrick"]
            metrics = get_metrics_from_results(results)

            for fitzpatrick, metric in zip(fitzpatrick_values_label, metrics):
                if fitzpatrick not in metrics_by_fitzpatrick:
                    metrics_by_fitzpatrick[fitzpatrick] = []
                metrics_by_fitzpatrick[fitzpatrick].append(metric)

        for fitzpatrick in metrics_by_fitzpatrick:
            metrics_by_fitzpatrick[fitzpatrick] = sum(metrics_by_fitzpatrick[fitzpatrick]) / len(metrics_by_fitzpatrick[fitzpatrick])
            metrics_by_fitzpatrick[fitzpatrick] = round(metrics_by_fitzpatrick[fitzpatrick], 2)

        sorted_metrics_by_fitzpatrick = dict(sorted(metrics_by_fitzpatrick.items()))
        overall_table.append([model] + list(sorted_metrics_by_fitzpatrick.values()))

    print("Fitzpatrick Skin Type Comparison")
    headers = [f"Type {i}" for i in fitzpatrick_values_label]
    headers.insert(0, "Model")

    print(tabulate(overall_table, headers=headers, tablefmt="latex"))
    df = pd.DataFrame(overall_table)
    df.to_excel("csv/fitzpatrick_skin_type_comparison.xlsx", index=False)
    print("Results saved to csv/fitzpatrick_skin_type_comparison.xlsx")

