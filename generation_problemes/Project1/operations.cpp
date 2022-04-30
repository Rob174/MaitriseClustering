#include "operations.h"
#include "utils.h"

double initial_cost(Clustering* clustering, Config* config)
{
    // compute cost
    double cost = 0;
    for (int i = 0; i < config->NUM_POINTS; i++)
    {
        double* point = &(clustering->p_c[i * config->NUM_DIM]);
        double* centroid = &(clustering->c_c[clustering->c_a[i] * config->NUM_DIM]);
        cost += dist(point, centroid, config);
    }
    return cost;
}

double cost_modif(Clustering* c, int f, int t, double* point, Config* config)
{
    // f: source cluster
    // t: destination cluster
    double part1 = (double)(c->n_p_p_c[t]) / (double)(c->n_p_p_c[t] + 1) * dist(&(c->c_c[t * config->NUM_DIM]), point, config);
    double part2 = 0;
    if (c->n_p_p_c[f] > 1)
    {
        part2 = (double)(c->n_p_p_c[f]) / (double)(c->n_p_p_c[f] - 1) * dist(&(c->c_c[f * config->NUM_DIM]), point, config);
    }
    /*
    if (fabs(part1 - part2) > 2*100*100) {
        printf("Error improvement too high\n");
        exit(1);
    }*/
    return part1 - part2;
}
void update(Clustering* c, int f, int t, int p, Config* config)
{
    // f: source cluster
    // t: destination cluster
    // update centroids
    double* point = &(c->p_c[p * config->NUM_DIM]);
    for (int dim = 0; dim < config->NUM_DIM; dim++)
    {
        if (c->n_p_p_c[f] <= 1) // If we had one point
            c->c_c[f * config->NUM_DIM + dim] = 0.;
        else
            c->c_c[f * config->NUM_DIM + dim] = (double)(c->n_p_p_c[f] * c->c_c[f * config->NUM_DIM + dim] - point[dim]) /
            (double)(c->n_p_p_c[f] - 1);
        c->c_c[t * config->NUM_DIM + dim] = (double)(c->n_p_p_c[t] * c->c_c[t * config->NUM_DIM + dim] + point[dim]) /
            (double)(c->n_p_p_c[t] + 1);

    }
    // update num_pts_per_clust
    c->n_p_p_c[t] = c->n_p_p_c[t]+1;
    c->n_p_p_c[f] = c->n_p_p_c[f]-1;
    /*
    if (c->n_p_p_c[f]<0 || c->n_p_p_c[t] > config->NUM_POINTS) {
        printf("Error num points: n_p_p_c[t]=%d,n_p_p_c[f]=%d\n", c->n_p_p_c[t], c->n_p_p_c[f]);
        exit(1);
    }
    */
    // update assignement
    c->c_a[p] = t;
}
std::tuple<double*, double*> compute_new_centroids(Clustering const * c, int f, int t, int p, Config* config) {
    // Does not modify clustering
    double* point = &(c->p_c[p * config->NUM_DIM]);
    double* new_f_center = new double[config->NUM_DIM];
    double* new_t_center = new double[config->NUM_DIM];
    for (int dim = 0; dim < config->NUM_DIM; dim++)
    {
        if (c->n_p_p_c[f] <= 1) // If we had one point
            new_f_center[dim] = 0.;
        else
            new_f_center[dim] = (double)(c->n_p_p_c[f] * c->c_c[f * config->NUM_DIM + dim] - point[dim]) /
            (double)(c->n_p_p_c[f] - 1);
        new_t_center[dim] = (double)(c->n_p_p_c[t] * c->c_c[t * config->NUM_DIM + dim] + point[dim]) /
            (double)(c->n_p_p_c[t] + 1);

    }
    return std::make_tuple(new_f_center, new_t_center);
}