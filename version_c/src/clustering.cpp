#include "clustering.h"
#include "utils.h"
#define GRID_COORD_MIN 0.
#define GRID_COORD_MAX 100.

std::tuple<Config*,IterationOrder*> get_config(int argc, char **argv)
{
    Config *config = new Config;
    if (argc != 5)
    {
        printf("4 arguments expected, found %d\n", argc - 1);
        printf("Expected arguments: NUM_POINTS NUM_DIM NUM_CLUST ITORDER\n");
        IterationOrderFactory::print_doc();
        exit(1);
    }
    config->NUM_POINTS = atoi(argv[1]);
    config->NUM_DIM = atoi(argv[2]);
    config->NUM_CLUST = atoi(argv[3]);
    int iteration_type = atoi(argv[4]);
    IterationOrder* iteration_order = IterationOrderFactory::create(config, iteration_type);
    return std::make_tuple(config, iteration_order);
}
void initialize(Clustering *clustering, Config *config)
{
    // Initialize points coordinates
    clustering->p_c = new double[config->NUM_POINTS*config->NUM_DIM];
    for (int i = 0; i < config->NUM_POINTS*config->NUM_DIM; i++)
        clustering->p_c[i] = prandom(GRID_COORD_MIN, GRID_COORD_MAX);
    // Initialize cluster intial assignements
    clustering->c_a = new int[config->NUM_POINTS];
    for (int i = 0; i < config->NUM_POINTS; i++)
        clustering->c_a[i] = (int)prandom(0, config->NUM_CLUST);
    // Initialize number of points per cluster
    clustering->n_p_p_c =new int[config->NUM_CLUST];
    for (int i = 0; i < config->NUM_CLUST; i++)
        clustering->n_p_p_c[i] = 0;
    // Initialize cluster centroids
    clustering->c_c = new double[config->NUM_CLUST*config->NUM_DIM];
    for (int i = 0; i < config->NUM_CLUST*config->NUM_DIM; i++)
        clustering->c_c[i] = 0.;
    // Compute cluster centroids
    for (int i = 0; i < config->NUM_POINTS; i++)
    {
        int cluster_id = clustering->c_a[i*config->NUM_DIM];
        for (int j = 0; j < config->NUM_DIM; j++)
        {
            clustering->c_c[cluster_id*config->NUM_CLUST+j] += clustering->p_c[i*config->NUM_POINTS+j];
            clustering->n_p_p_c[cluster_id]++;
        }
    }
    for (int i = 0; i < config->NUM_CLUST; i++)
    {
        for (int j = 0; j < config->NUM_DIM; j++)
        {
            clustering->c_c[i*config->NUM_CLUST+j] /= clustering->n_p_p_c[i];
        }
    }
}
