from typing import *
import numpy as np
from repr_partitions_cluster.src.kmean.kmean import kmean
from repr_partitions_cluster.src.generators import generate_simple1,random_centroids
from repr_partitions_cluster.src.hmeans.hmeans import hmeans
from repr_partitions_cluster.src.hmeans.improvement_choice import CallbackBestImprovement,CallbackFirstImprovement,cost

class HMeanTest:
    def test_hmeans_simple1(self):
        points_coords,points_init_assign,clust_init_coords = random_centroids(*generate_simple1(), num_clusters=2)
        points_assign_impr,clust_coords_impr = kmean(points_coords,np.copy(points_init_assign),np.copy(clust_init_coords))
        for params,init_type in zip([(points_coords,points_init_assign,clust_init_coords),(points_coords,points_assign_impr,clust_coords_impr)],["random","kmean+"]):
            for ImprClass in [CallbackBestImprovement,CallbackFirstImprovement]:
                points_assign,clust_coords,fin_cost = hmeans(*params,type_improvement=ImprClass(),initial_cost=cost(*params))
                print(f"{init_type=} {ImprClass.__name__=} {fin_cost=}")
                
if __name__ == "__main__":
    HMeanTest().test_hmeans_simple1()