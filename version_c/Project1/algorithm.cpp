#include "algorithm.h"

#define UNINITIALIZED -1
void run(int argc, char* argv[], int loop_id, bool verbose, long seed_points,long seed_assigns)
{
    auto params = get_config(argc, argv, seed_points, seed_assigns);
    Config* config = std::get<0>(params);
    IterationOrder* order = std::get<1>(params);
    ImprovementChoice* impr = std::get<2>(params);
    Initializer* initializer = std::get<3>(params);
    Result* result = std::get<4>(params);

    // Define initial clustering
    Clustering* clustering = new Clustering();
    initialize(clustering, config);
    double cost = initial_cost(clustering, config);
    result->set_init_cost(cost);
    impr->initialize(clustering);
    // Local seach
    bool improvement = true;
    result->set_time_start();
    // KMeans improvement or no improvement
    initializer->initialize(clustering, config);
    Clustering* initial_clustering = deepcopy(clustering, config);
    while (improvement)
    {
        ClusteringChoice* choice = new ClusteringChoice();
        for (int point_moving_id = 0; point_moving_id < config->NUM_POINTS; point_moving_id++)
        {
            int from_cluster_id = clustering->c_a[point_moving_id];
            order->restart(from_cluster_id);
            int to_clust_id = order->next();
            while (to_clust_id != -1)
            {
                double modif = cost_modif(
                    clustering,
                    from_cluster_id, to_clust_id,
                    &(clustering->p_c[point_moving_id * config->NUM_DIM]), config);
                if (modif < choice->vij)
                {
                    impr->choose_solution(choice, clustering, modif, point_moving_id, from_cluster_id, to_clust_id);
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
        if (choice->i == UNINITIALIZED || choice->vij > -0.001)
            improvement = false;
        else
        {
            // Update cluster assignements
            impr->after_choice();
            update(clustering, choice->l, choice->j, choice->i, config);
            cost += choice->vij;
            /*
            if (cost < 0) {
                std::cout << "Cost negative for seed" << config->SEED << std::endl;
                exit(1);
            }
            */

            result->set_next_iter();
        }
        double time_micro_elapsed = result->get_time_elapsed();
        /*if (time_micro_elapsed > 1000000 * 15) {
            std::cout << "Error timeout ";
            //break;
        }*/
        delete choice;
    }
    result->set_final_cost(cost);
    result->set_time_end();
    if(verbose)
        result->print_results();
    create_dataset(result, initial_clustering, clustering);
    clean(config, clustering, initial_clustering,order, result, impr, initializer);
}