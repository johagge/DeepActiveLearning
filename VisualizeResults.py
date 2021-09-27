import csv
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import sys

# TODO alternatively use only files specified via command line

logdir = "log10"

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
        print(key)
        print(foo)
        # for proper pandas dataframes:
        #   each column as one theme (e.g. random/vae)
        #   for multiple runs: first run, afterwards second run with same row names again
        #std = np.std(foo)
        #mean = np.mean(foo)
        #print(mean)
        #sys.exit()
    """
        data_in_columns = zip(*value)
        results[key] = data_in_columns
    """

fig, ax = plt.subplots()
used_images = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
used_images = np.array(used_images)
used_images = used_images / 10
for key, value in results.items():
    if isinstance(value, pd.DataFrame):
        #value = value.transpose()
        #print(value)
        #sns.lineplot(data=value, x=used_images, y=key)
        #continue
        # 95% confidence interval
        ci = 1.96 * np.std(value) * np.mean(value)
        print(key)
        print(value[0])
        y = [y for y in np.mean(value)]
        print(y)
        plt.fill_between(used_images, y - ci, y + ci, alpha=.3, label=key)
    else:
        ax.plot(used_images, value, "-o", label=key)
ax.legend()
plt.xlim(used_images[0], used_images[-1])  # use first and last element of used images as scale for x
plt.ylim(0, 1)
plt.xlabel("number of images used in training")
plt.ylabel("mAP on test data")

# use different colors, because otherwise the colors are reused
colormap = plt.cm.gist_ncar  # nipy_spectral, Set1,Paired
# only up to 0.9, because 1 is almost white and hard to see
colors = [colormap(i) for i in np.linspace(0, 0.9, len(ax.lines))]
for i, j in enumerate(ax.lines):
    j.set_color(colors[i])

plt.show(dpi=200)
# put the legend outside of graph
# do it after showing the figure, because for some reason the bbox inches tight only works in savefig
ax.legend(bbox_to_anchor=(1, 1))
fig.savefig("graph.png", dpi=200, bbox_inches='tight')
