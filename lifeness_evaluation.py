from evaluation.utils import get_results, get_pickle_path
import numpy as np

datasets = ["test", "validation", "dead", "own_videos", "emergency", "test_shuffled", "validation_shuffled"]
dataset_shownames = ["Test set", "Validation set", "Ad Banner Videos", "Distribution Shift Videos", "Realistic Videos"]


print(r"\begin{tabular}{l|c}")
print(r"Dataset & Mean Lifeness \\ \hline")

for dataset, shownames in zip(datasets, dataset_shownames):
    outputs = get_pickle_path("Lifeness", dataset, model_name="Physnet_Lifeness")
    results = get_results(outputs, dataset=dataset)
    mean_lifeness = np.mean(results["lifenesses"])

    print(f"{shownames} & {mean_lifeness:.4f} \\\\")

print(r"\end{tabular}")
