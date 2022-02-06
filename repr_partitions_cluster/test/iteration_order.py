import unittest

from repr_partitions_cluster.src.hmeans.iteration_order import *
import numpy as np


class TestIterationOrder(unittest.TestCase):
    def __init__(self,*args,**kwargs):
        super(TestIterationOrder,self).__init__(*args,**kwargs)
        self.classes_to_test = [CURR(), RANDOM(), BACK()]
        
    def test_no_src_clust_in_result(self):
        for class_obj in self.classes_to_test:
            num_clusters = np.random.randint(2, 100)
            src_clust_id = np.random.randint(0, num_clusters)
            iterable = class_obj.get_points_order(src_clust_id, num_clusters)
            self.assertNotIn(src_clust_id, iterable)


if __name__ == '__main__':
    unittest.main()
