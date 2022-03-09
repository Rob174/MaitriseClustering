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

int main(int argc, char* argv[])
{
    srand((unsigned int)time(NULL));

    auto params = get_config(argc, argv);
    Config* config = std::get<0>(params);
    IterationOrder* order = std::get<1>(params);
    ImprovementChoice* impr = std::get<2>(params);
    Result* result = std::get<3>(params);

    // Define initial clustering
    Clustering clustering;
    initialize(&clustering, config);
    float cost = initial_cost(&clustering, config);
    result->set_init_cost(cost);
    // Local seach
    bool improvement = true;
    result->set_time_start();
    while (improvement)
    {
        float vij = 0;
        int l = UNINITIALIZED, j = UNINITIALIZED, i = UNINITIALIZED;
        for (int point_moving_id = 0; point_moving_id < config->NUM_POINTS; point_moving_id++)
        {
            int from_cluster_id = clustering.c_a[point_moving_id];
            order->restart(from_cluster_id);
            int to_clust_id = order->next();
            while (to_clust_id != -1)
            {
                float modif = cost_modif(
                    &clustering,
                    from_cluster_id, to_clust_id,
                    &(clustering.p_c[point_moving_id * config->NUM_DIM]), config
                );
                if (modif < vij)
                {
                    vij = modif;
                    i = point_moving_id;
                    l = from_cluster_id;
                    j = to_clust_id;
                    if(impr->stop_loop(vij))
                        goto outloop;
                }
                to_clust_id = order->next();
                result->set_next_iter_glob();
            }
        }
    outloop:
        if (i == UNINITIALIZED)
            improvement = false;
        else
        {
            // Update cluster assignements
            update(&clustering, l, j, i, config);
            cost += vij;
            result->set_next_iter();
        }
    }
    result->set_final_cost(cost);
    result->set_time_end();
    result->print_results();
}