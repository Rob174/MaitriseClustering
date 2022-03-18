#include "improvementChoice.h"

ImprovementChoice* ImprFactory::create(int identifier, Result* res,Config*config)
{
    switch (identifier)
    {
    case 0:
        return new BestImpr(res);
    case 1:
        return new FirstImpr(res);
    case 2:
        return new DelayedImpr1(res, config);
    default:
        std::cout << "Wrong argument ImprovementChoice " << identifier << std::endl;
        exit(1);
    }
}

void ImprFactory::print_doc()
{
}

bool BestImpr::stop_loop()
{
	return false;
}

bool FirstImpr::stop_loop()
{
	return true;
}


void ImprovementChoice::choose_solution(ClusteringChoice*choice, Clustering* clustering, double sugg_vij, int sugg_i, int sugg_l, int sugg_j)
{
    choice->vij = sugg_vij;
    choice->i = sugg_i;
    choice->l = sugg_l;
    choice->j = sugg_j;
}

bool ImprovementChoice::stop_loop()
{
    return false;
}

void DelayedImpr1::choose_solution(ClusteringChoice* choice, Clustering* clustering, double sugg_vij, int sugg_i, int sugg_l, int sugg_j)
{
    int counter_not_closest = 0;
    auto former_dest_clust = compute_new_centroids(clustering, sugg_l, sugg_j, sugg_i, this->config);
    double* new_f_center = std::get<0>(former_dest_clust);
    double* new_t_center = std::get<1>(former_dest_clust);
    for (int q = 0; q < this->config->NUM_POINTS; q++) {
        int min_centr = -1;
        double min_dist = std::numeric_limits<double>::max();
        for (int z = 0; z < this->config->NUM_CLUST; z++) {
            double* centroid;
            if (z == sugg_l) {
                centroid = new_f_center;
            }
            else if (z == sugg_j) {
                centroid = new_t_center;
            }
            else {
                centroid = &(clustering->c_c[z * this->config->NUM_DIM]);
            }
            double d = dist(centroid, &(clustering->p_c[q * this->config->NUM_DIM]), this->config);

            if (d < min_dist) {
                min_dist = d;
                min_centr = z;
            }
        }
        if ((q == sugg_i && min_centr != sugg_j) ||
            (q != sugg_i && min_centr != clustering->c_a[q]))
            counter_not_closest++;

    }
    delete new_f_center;
    delete new_t_center;
    if (counter_not_closest > choice->counter_not_closest) {
        ImprovementChoice::choose_solution(choice, clustering, sugg_vij, sugg_i, sugg_l, sugg_j);
        choice->counter_not_closest = counter_not_closest;
    }
}