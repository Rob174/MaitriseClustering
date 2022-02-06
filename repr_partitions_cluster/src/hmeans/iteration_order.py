from abc import ABC, abstractmethod
from typing import *
import numpy as np


class AbstractIterationOrder(ABC):
    def get_points_order(self, src_clust_id: int, num_clusters: int) -> Iterable:
        assert num_clusters > 1, "We must have more than 1 cluster to choose a different cluster for the destination"
        return []


class CURR(AbstractIterationOrder):
    def get_points_order(self, src_clust_id: int, num_clusters: int) -> Iterable:
        super(CURR,self).get_points_order(src_clust_id, num_clusters)
        return (i % num_clusters for i in range(src_clust_id+1, src_clust_id+num_clusters) if i % num_clusters != src_clust_id)


class BACK(AbstractIterationOrder):
    def get_points_order(self, src_clust_id: int, num_clusters: int) -> Iterable:
        super(BACK,self).get_points_order(src_clust_id, num_clusters)
        return filter(lambda x: x != src_clust_id, range(num_clusters))


class RANDOM(AbstractIterationOrder):
    def get_points_order(self, src_clust_id: int, num_clusters: int) -> Iterable:
        super(RANDOM,self).get_points_order(src_clust_id, num_clusters)
        return filter(lambda x: x != src_clust_id, np.random.permutation(num_clusters))
