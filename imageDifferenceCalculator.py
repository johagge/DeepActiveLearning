import random

from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from tqdm import tqdm
import pickle

class DifferenceCalculator:

    def __init__(self, image_list):
        self.image_list = image_list

    def calculate_for_two(self, image1, image2):
        pass

    def load_image_pillow(self, img_path):
        return Image.open(img_path)

    def load_results(self):
        """
        Since the comparison only needs to be done once, we don't have to generate them each time
        """
        pass

    def cluster(self, amount_clusters):
        pass



class Image2Vector(DifferenceCalculator):

    def __init__(self, image_list):
        super(Image2Vector, self).__init__(image_list)
        self.img2vec = Img2Vec(cuda=False) # TODO set true
        self.vector_list = None
        self.pickle_name = "img2vec.pickle"
        self.amount_of_clusters = None
        self.kmeans = None
        self.reduced_data = None

    def calculate_for_two(self, image1, image2):
        super()
        vec = self.img2vec.get_vec(image1)
        vec2 = self.img2vec.get_vec(image2)
        similarity = cosine_similarity(vec.reshape((1, -1)), vec2.reshape((1, -1)))[0][0]
        print(similarity)

    def load_results(self):
        """
        If this fails, you first need to run the generate all image vectors function which generates the pickle
        :return:
        """
        with open(self.pickle_name, "rb") as f:
            self.vector_list = pickle.load(f)

    def generate_all_image_vectors(self):
        """
        This generates the pickle file including all the vectors generated from the images
        This saves a few minutes
        Should only have to run this once.
        :return:
        """
        vector_list = np.zeros((len(self.image_list), 512))  # 512 because resnet-18 is used as default with 512 output
        print('generating image vectors...')
        for i, image in tqdm(enumerate(self.image_list)):
            img = self.load_image_pillow(image)
            vector = self.img2vec.get_vec(img)
            vector_list[i, :] = vector
        self.vector_list = vector_list
        with open(self.pickle_name, "wb") as f:
            pickle.dump(vector_list, f)

    def cluster(self, amount_clusters):
        # inspired by https://github.com/christiansafka/img2vec/blob/master/example/test_clustering.py
        self.amount_of_clusters = amount_clusters
        self.load_results()
        print('Applying PCA...')
        reduced_data = PCA(n_components=2).fit_transform(self.vector_list)  # TODO try more than 2 components
        print('calculating kmeans')
        kmeans = KMeans(init='k-means++', n_clusters=amount_clusters, n_init=25)
        kmeans.fit(reduced_data)
        self.kmeans = kmeans
        self.reduced_data = reduced_data
        return kmeans, reduced_data

    def visualizeCluster(self):
        """
        visualizes the clusters. only works with 2 dimensional reduced data
        :param kmeans: sklearn kmeans
        :param reduced_data: PCA reduced data
        :return: None
        """
        import matplotlib.pyplot as plt

        if not self.kmeans or not self.reduced_data:
            print("Please run cluster() first, so we have kmeans and reduced data available")
            sys.exit()

        # the following code in this function is from: (slightly changed)
        # https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html

        # Step size of the mesh. Decrease to increase the quality of the VQ.
        h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].
        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
        y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Obtain labels for each point in mesh. Use last trained model.
        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1)
        plt.clf()
        plt.imshow(Z, interpolation="nearest",
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired, aspect="auto", origin="lower")

        plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
        # Plot the centroids as a white X
        centroids = kmeans.cluster_centers_
        plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=169, linewidths=3,
                    color="w", zorder=10)
        plt.title("K-means clustering on the TORSO-21 training set (PCA-reduced data)\n"
                  "Centroids are marked with white cross")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.show()
        #plt.savefig("kmeans.png", dpi=200)

    def images_by_cluster(self, amount_clusters):
        kmeans, reduced_data = self.cluster(amount_clusters)
        images_by_cluster = [list() for e in range(a.amount_of_clusters)]  # one sub list for each defined cluster
        for i, e in enumerate(reduced_data):
            cluster = kmeans.predict(e.reshape(1, -1))
            # this assumes that the order of the reduced data is the same as the image list
            images_by_cluster[cluster[0]].append(self.image_list[i])  # put the image path into the fitting cluster list
        return images_by_cluster

    # TODO remember to delete selected samples from
    #       self.reduced_data = None
    #       self.vector_list = None
    #       self.image_list = image_list
    # TODO delete cluster and cluster again after each run


if __name__ == "__main__":
    import os
    import glob
    # find all images in folder
    trainImagesPool = []
    datasets = [x[0] for x in os.walk("/home/jonas/Downloads/1076/")]  # a list of all subdirectories (including root directory)

    for d in datasets:
        trainImagesPool = glob.glob(f"{d}/*.png", recursive=True)
        trainImagesPool += glob.glob(f"{d}/*.PNG", recursive=True)
        trainImagesPool += glob.glob(f"{d}/*.jpg", recursive=True)
        trainImagesPool += glob.glob(f"{d}/*.JPG", recursive=True)

    a = Image2Vector(trainImagesPool)
    # a.generate_all_image_vectors()
    a.images_by_cluster(10)

    # skip currently unnecessary debug 
    import sys
    sys.exit()

    # Visualization
    kmeans, reduced_data = a.cluster(10)
    print(kmeans)
    a.visualizeCluster()

    # images by cluster initial
    print(kmeans.predict(reduced_data))  # returns assigned cluster for each value
    images_by_cluster = [list() for e in range(a.amount_of_clusters)]  # one sub list for each defined cluster
    for i, e in enumerate(reduced_data):
        cluster = kmeans.predict(e.reshape(1, -1))
        images_by_cluster[cluster[0]].append(a.image_list[i])  # put the image path into the fitting cluster list
    for e in images_by_cluster:
        print(len(e))

    # Some tests to get a feel for how different images compare in one to one comparison
    a = Image2Vector(["/home/jonas/Downloads/1076/1042-image00538.png", "/home/jonas/Downloads/1076/130-16_02_2018__11_16_34_0525_upper.png"])
    img1 = a.load_image_pillow("/home/jonas/Downloads/1076/1042-image00538.png")
    img2 = a.load_image_pillow("/home/jonas/Downloads/1076/130-16_02_2018__11_16_34_0525_upper.png")
    img3 = a.load_image_pillow("/home/jonas/Downloads/1076/1042-image00007.png")
    img4 = a.load_image_pillow("/home/jonas/Downloads/1076/1042-image00686.png")
    a.calculate_for_two(img1, img1)
    a.calculate_for_two(img1, img2)
    a.calculate_for_two(img1, img3)
    a.calculate_for_two(img1, img4)