from repr_partitions_cluster.src.kmean.kmean import kmean
from repr_partitions_cluster.src.generators import generate_simple1,random_centroids
from repr_partitions_cluster.src.utils import equals
from repr_partitions_cluster.src.visualize import show_clustering
import numpy as np
from typing import *
from repr_partitions_cluster.src.kmean.kmean import kmean

def centroid_check(
        points_assign,clust_coords,
        init_point_assign,init_clust_coords,
        points_coords
    ):
    show_clustering(points_coords,init_point_assign,init_clust_coords,title="Initial clustering")
    show_clustering(points_coords,points_assign,clust_coords,title="Clustering after improvement")
    input("ok?")
class KMeanTest:
    
    def test_simple1(self):
        for _ in range(10): 
            points_coords,points_init_assign,clust_init_coords = random_centroids(*generate_simple1(), num_clusters=2)
            points_assign,clust_coords = kmean(points_coords,np.copy(points_init_assign),np.copy(clust_init_coords))
            centroid_check(points_assign,clust_coords,
                           points_init_assign,clust_init_coords,
                           points_coords)
        

if __name__ == "__main__":
    test = KMeanTest()
    test.test_simple1()