import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
from evaluation.gt_visualize import GTVisualizer
from dataset.data_loader.VitalVideosLoader import get_raw_data_vital_videos

def plot_pie_chart(data_list, title, filename):
    print(data_list)
    labels = list(set(data_list))
    counts = [data_list.count(label) for label in labels]
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 20})
    plt.title(title, fontsize=25)
    plt.savefig(filename)
    plt.close()

def plot_histogram(data_list, title, filename):
    data_list = [int(age) for age in data_list if age.isdigit()]  # Filter out non-numeric ages
    plt.figure(figsize=(10, 6))
    plt.hist(data_list, bins=10, edgecolor='black')
    plt.title(title, fontsize=25)
    plt.xlabel('Age', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(filename)
    plt.close()

def evaluate_dataset_stats(data_dirs, dataset_name="Test"):
    """Evaluate dataset statistics."""
    num_samples = len(data_dirs)
    print(f"Number of samples in the dataset: {num_samples}")

    genders_list = []
    ages_list = []
    fitzpatricks_list = []
    locations_list = []

    # Example: Print first 5 entries
    for curr_dir in data_dirs:
        curr_video_path = curr_dir['path']
        curr_json_path = os.path.join(os.path.dirname(curr_video_path), f"{curr_dir['subject']}.json")

        gt_visualizer = GTVisualizer(curr_json_path, curr_video_path)
        location = gt_visualizer.get_location()['location']
        fitzpatrick = gt_visualizer.get_fitzpatrick()
        age = gt_visualizer.get_age()
        gender = gt_visualizer.get_gender()

        genders_list.append(gender.strip())
        ages_list.append(age.strip())
        fitzpatricks_list.append(fitzpatrick.strip())
        locations_list.append(location.strip())


    plot_pie_chart(genders_list, 'Gender Distribution', f'{dataset_name}_gender_distribution_pie.png')
    plot_pie_chart(fitzpatricks_list, 'Fitzpatrick Skin Type Distribution', f'{dataset_name}_fitzpatrick_distribution_pie.png')
    plot_pie_chart(locations_list, 'Location Distribution', f'{dataset_name}_location_distribution_pie.png')
    plot_histogram(ages_list, 'Age Distribution', f'{dataset_name}_age_distribution_pie.png')


dataset_path = "/mnt/data/vitalVideos"
train_split = range(0, 141)  # Example split, adjust as needed
valid_split = range(141, 173)  # Example split, adjust as needed
test_split = range(173, 202)  # Example split, adjust as needed
B_split = range(0, 14)  # Example split, adjust as needed

data_dirs = get_raw_data_vital_videos(dataset_path)
data_dirs_train = [d for d in data_dirs if d['index'] in train_split]
data_dirs_valid = [d for d in data_dirs if d['index'] in valid_split]
data_dirs_test = [d for d in data_dirs if d['index'] in test_split]
data_dirs_B = [d for d in data_dirs if d['index'] in B_split]

evaluate_dataset_stats(data_dirs_test, dataset_name="VitalVideos_Test")
evaluate_dataset_stats(data_dirs_train, dataset_name="VitalVideos_Train")
evaluate_dataset_stats(data_dirs_valid, dataset_name="VitalVideos_Valid")
evaluate_dataset_stats(data_dirs_B, dataset_name="VitalVideos_B")