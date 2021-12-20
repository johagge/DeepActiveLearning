import csv
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import sys

category_names = {"random": "random sampling",
                  "mean_confidence": "mean confidence",
                  "mean_confidence_no_boxes": "mean confidence with no predictions",
                  "min_confidence": "minimum confidence",
                  "lowest_max_confidence": "lowest maximum confidence",
                  "median": "median confidence",
                  "max": "maximum confidence",
                  "min_bb": "least predictions",
                  "most_bb": "most predictions",
                  "gaussian_mean_difference": "gaussian noise / mean difference",
                  "gaussian_map_mean": "gaussian noise / mAP",
                  "motion_blur_map_mean": "motion blur noise / mAP",
                  "image_2_vec_resnet": "clustering",
                  "vae": "Variational Autoencoder",
                  "learnloss": "Learning Loss",
                  }

first_table = ["used images", "random sampling", "mean confidence", "mean confidence with no predictions", "minimum confidence", "lowest maximum confidence", "median confidence", "maximum confidence", "least predictions"]
second_table = ["used images", "most predictions", "gaussian noise / mean difference", "gaussian noise / mAP", "motion blur noise / mAP", "clustering", "Variational Autoencoder", "Learning Loss"]

first_1000_table = ["used images", "random sampling", "mean confidence", "mean confidence with no predictions", "minimum confidence", "lowest maximum confidence"]
second_1000_table = ["used images", "maximum confidence", "most predictions", "gaussian noise / mAP", "clustering", "Variational Autoencoder"]

coco_table = ["used images", "random sampling", "mean confidence", "minimum confidence", "gaussian noise / mAP", "Variational Autoencoder"]

logdir = "log1000coco"

files = glob.glob(os.path.join(logdir, "*.txt"))
print(files)

results = {}
for file in files:
    method = ""
    mAPs = []
    with open(file, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            if not method:
                method = row[0]
                continue  # save method and go to next row
            if row[0] == '200':
                mAPs.append(float(row[1]))
    if method in results:
        tempresult = []
        # we need to handle it differently when we already have a list as first element (instead of floats)
        if isinstance(results[method][0], list):
            for result in results[method]:
                tempresult.append(result)
            tempresult.append(mAPs)
            results[method] = tempresult
        else:
            # do this case explicitly to avoid results[method] = [1,2,3,[4,5,6]]
            results[method] = [results[method], mAPs]
    else:
        results[method] = mAPs
# format the methods with multiple runs correctly with one tuple for each x
all_results = {}
for key, value in results.items():
    if isinstance(value[0], list):
        results = []
        for iteration in range(len(value[0])):
            results.append([result[iteration] for result in value])
        mean_results = []
        for value in results:
            mean_results.append(np.mean(value))

        result_frame = pd.DataFrame(mean_results)
        all_results[key] = result_frame
    else:
        result_frame = pd.DataFrame(value)
        all_results[key] = result_frame


# generate amounts of images list
run_list = []
for n in range(0, 10):
    run_list.append((n+1)*1000)
print(run_list)

tex_frame = pd.DataFrame(
    {"used images": run_list}
)
for key, value in all_results.items():
    tex_frame[category_names[key]] = value
# print(tex_frame)
# remove useless row count
print()
print(tex_frame.to_latex(index=False, multicolumn=True, columns=coco_table))