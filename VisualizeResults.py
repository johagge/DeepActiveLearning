import csv
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import sys

random = ['random']
confidence_based = ['mean_confidence', 'mean_confidence_no_boxes', 'min_confidence', 'lowest_max_confidence']
confidence_based2 = ['median', 'max']
bb_amount = ['min_bb', 'most_bb']
noise = ['gaussian_mean_difference', 'gaussian_map_mean', 'motion_blur_map_mean']
img2vec = ['image_2_vec_resnet', 'vae']
learnloss = ['learnloss']

categories = [random, confidence_based, confidence_based2, bb_amount, noise, img2vec, learnloss]

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

colors = {"random": "red",
          "mean_confidence": "blue",
          "mean_confidence_no_boxes": "orange",
          "min_confidence": "tab:purple",
          "lowest_max_confidence": "magenta",
          "median": "blue",
          "max": "orange",
          "min_bb": "blue",
          "most_bb": "orange",
          "gaussian_mean_difference": "blue",
          "gaussian_map_mean": "orange",
          "motion_blur_map_mean": "tab:purple",
          "image_2_vec_resnet": "blue",
          "vae": "orange",
          "learnloss": "blue",
          }

logdir = "log100"

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
for key, value in results.items():
    if isinstance(value[0], list):

        foo = pd.DataFrame(value)
        results[key] = foo
        # print(key)
        # print(foo[0])

        run_list = []
        result_list = []
        for n in range(0, 10):
            for run in foo[n]:
                run_list.append((n+1)*100)
                result_list.append(run)
        new_frame = pd.DataFrame(
            {
                "current_run": run_list,
                "results": result_list
            }
        )
        new_frame.columns = ['used_images', 'map_values']
        foo = new_frame
        results[key] = foo

        # for proper pandas dataframes:
        #   each column as one theme (e.g. random/vae)
        #   for multiple runs: first run, afterwards second run with same row names again
        # std = np.std(foo)
        # mean = np.mean(foo)
        # print(mean)
        # sys.exit()
    """
        data_in_columns = zip(*value)
        results[key] = data_in_columns
    """
for i, cat in enumerate(categories):

    fig, ax = plt.subplots()
    used_images = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    used_images = np.array(used_images)
    used_images = used_images
    for key, value in results.items():
        # print(key)
        if key not in cat and key != 'random':
            continue
        if isinstance(value, pd.DataFrame):
            # value = value.transpose()
            # print(value)
            sns.lineplot(data=value, x='used_images', y='map_values', label=key, color=colors[key])# y=key)
            continue  # previous way of plotting
            # 95% confidence interval
            ci = 1.96 * np.std(value) * np.mean(value)
            # print(key)
            # print(value[0])
            y = [y for y in np.mean(value)]
            # print(y)
            plt.fill_between(used_images, y - ci, y + ci, alpha=.3, label=category_names[key])
        else:
            ax.plot(used_images, value, "-o", label=category_names[key])
    ax.legend()
    plt.grid()
    plt.yticks(ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.xlim(used_images[0], used_images[-1])  # use first and last element of used images as scale for x
    plt.ylim(0, 1)
    plt.xlabel("number of images used in training")
    plt.ylabel("mAP on test data")

    '''
    # use different colors, because otherwise the colors are reused
    colormap = plt.cm.gist_ncar  # nipy_spectral, Set1,Paired
    # only up to 0.9, because 1 is almost white and hard to see
    colors = [colormap(i) for i in np.linspace(0, 0.9, len(ax.lines))]
    for i, j in enumerate(ax.lines):
        j.set_color(colors[i])
    '''



    plt.show(dpi=200)
    # put the legend outside of graph
    # do it after showing the figure, because for some reason the bbox inches tight only works in savefig
    ax.legend(bbox_to_anchor=(1, 1))
    fig.savefig(f"viz/{logdir}_{categories[i]}_graph.png", dpi=200, bbox_inches='tight')
