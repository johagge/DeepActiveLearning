import csv
import glob
import os
import matplotlib.pyplot as plt

# TODO alternatively use only files specified via command line

logdir = "log"

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
    results[method] = mAPs

fig, ax = plt.subplots()
used_images = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
for key, value in results.items():
    ax.plot(used_images, value, "-o", label=key)
ax.legend()
plt.xlim(100, 1000)
plt.ylim(0, 1)
plt.xlabel("amount of images used in training")
plt.ylabel("mAP")
plt.show()
fig.savefig("graph.png", dpi=200)