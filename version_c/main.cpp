#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <tuple>
#include <iostream>
#include "clustering.h"
#include "operations.h"
#include "utils.h"
#include "iteration_order.h"

#define UNINITIALIZED -1

int main(int argc, char *argv[])
{
    srand((unsigned int)time(NULL));
    auto params = get_config(argc, argv);
    Config *config = std::get<0>(params);
    IterationOrder *order = std::get<1>(params);
    // Define initial clustering
    Clustering clustering;
    initialize(&clustering, config);
    double cost = initial_cost(&clustering, config);
    // Local seach
    int improvement = 1;
    int step = 0;
    while (improvement)
    {
        double vij = 0;
        int l = UNINITIALIZED, j = UNINITIALIZED, i = UNINITIALIZED;
        for (int point_moving_id = 0; point_moving_id < config->NUM_POINTS; point_moving_id++)
        {
            int from_cluster_id = clustering.c_a[point_moving_id];
            order->restart(from_cluster_id);
            for (int to_clust_id = 0; to_clust_id < config->NUM_CLUST; to_clust_id++)
            {
                if (to_clust_id == from_cluster_id)
                    continue;
                double modif = cost_modif(
                    &clustering,
                    from_cluster_id, to_clust_id,
                    clustering.p_c[point_moving_id], config);
                if (modif < vij)
                {
                    vij = modif;
                    i = point_moving_id;
                    l = from_cluster_id;
                    j = to_clust_id;
                    goto outloop;
                }
            }
        }
    outloop:
        step++;
        if (i == UNINITIALIZED)
            improvement = 0;
        else
        {
            // Update cluster assignements
            update(&clustering, l, j, i, ->config);
            cost += vij;
        }
        printf("iter\n");
    }
}