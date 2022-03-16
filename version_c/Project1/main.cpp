#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <tuple>
#include <set>
#include <list>
#include <iostream>
#include "clustering.h"
#include "operations.h"
#include "utils.h"
#include "iteration_order.h"
#include <math.h>
#include <windows.h>
#include <limits>

#define UNINITIALIZED -1
void run(int argc, char *argv[])
{
    int seed = 86165187;// GetTickCount64();
    srand(seed);
    auto params = get_config(argc, argv, seed);
    Config *config = std::get<0>(params);
    IterationOrder *order = std::get<1>(params);
    ImprovementChoice *impr = std::get<2>(params);
    Initializer *initializer = std::get<3>(params);
    Result *result = std::get<4>(params);

    // Define initial clustering
    Clustering clustering;
    initialize(&clustering, config);
    float cost = initial_cost(&clustering, config);
    if (isinf(cost))
    {
        int b = 0;
    }
    result->set_init_cost(cost);
    // Local seach
    bool improvement = true;
    result->set_time_start();
    // KMeans improvement or no improvement
    initializer->initialize(&clustering, config);
    while (improvement)
    {
        ClusteringChoice *choice = new ClusteringChoice();
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
                    &(clustering.p_c[point_moving_id * config->NUM_DIM]), config);
                if (modif < choice->vij)
                {
                    impr->choose_solution(choice, &clustering, modif, point_moving_id, from_cluster_id, to_clust_id);
                    if (impr->stop_loop())
                    {
                        goto outloop;
                    }
                }
                to_clust_id = order->next();
                result->set_next_iter_glob();
            }
            order->end_loop();
        }
    outloop:
        if (choice->i == UNINITIALIZED)
            improvement = false;
        else
        {
            // Update cluster assignements
            update(&clustering, choice->l, choice->j, choice->i, config);
            cost += choice->vij;
            if (cost < 0) {
                std::cout << "Cost negative for seed" << config->SEED << std::endl;
                exit(1);
            }

            result->set_next_iter();
        }
        double time_micro_elapsed = result->get_time_elapsed();
        if (time_micro_elapsed > 1000000 * 60 * 5) {
            std::cout << "Error timeout ";
            //break;
        }
        delete choice;
    }
    result->set_final_cost(cost);
    result->set_time_end();
    result->print_results();
    clean(config, &clustering,order, result, impr, initializer);
}
int main(int argc, char *argv[])
{
    for (int i = 0; i < 3; i++)
        run(argc, argv);
}