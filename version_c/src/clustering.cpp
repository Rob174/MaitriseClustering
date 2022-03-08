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
    clustering->p_c = (double**)malloc(config->NUM_DIM * sizeof(double*));
    for (int i = 0; i < config->NUM_POINTS; i++)
    {
        clustering->p_c[i] = (double*)malloc(config->NUM_DIM * sizeof(double));
        for (int d = 0; d < config->NUM_DIM; d++)
        {
            clustering->p_c[i][d] = prandom(GRID_COORD_MIN, GRID_COORD_MAX);
        }
    }
    // Initialize cluster intial assignements
    clustering->c_a = (int*)malloc(config->NUM_POINTS * sizeof(int));
    for (int i = 0; i < config->NUM_POINTS; i++)
        clustering->c_a[i] = (int)prandom(0, config->NUM_CLUST);
    // Initialize number of points per cluster
    clustering->n_p_p_c = (int*)malloc(config->NUM_CLUST * sizeof(int));
    for (int i = 0; i < config->NUM_CLUST; i++)
        clustering->n_p_p_c[i] = 0;
    // Initialize cluster centroids
    clustering->c_c = (double**)malloc(config->NUM_CLUST * sizeof(double *));
    for (int i = 0; i < config->NUM_CLUST; i++)
    {
        clustering->c_c[i] = (double*)malloc(config->NUM_DIM * sizeof(double));
        for (int j = 0; j < config->NUM_DIM; j++)
        {
            clustering->c_c[i][j] = 0.;
        }
    }
    // Compute cluster centroids
    for (int i = 0; i < config->NUM_POINTS; i++)
    {
        int cluster_id = clustering->c_a[i];
        for (int j = 0; j < config->NUM_DIM; j++)
        {
            clustering->c_c[cluster_id][j] += clustering->p_c[i][j];
            clustering->n_p_p_c[cluster_id]++;
        }
    }
    for (int i = 0; i < config->NUM_CLUST; i++)
    {
        for (int j = 0; j < config->NUM_DIM; j++)
        {
            clustering->c_c[i][j] /= clustering->n_p_p_c[i];
        }
    }
}
