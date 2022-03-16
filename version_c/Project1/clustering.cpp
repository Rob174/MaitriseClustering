#include "clustering.h"

#define GRID_COORD_MIN 0.
#define GRID_COORD_MAX 100.

std::tuple<Config*, IterationOrder*, ImprovementChoice*, Initializer* ,Result*> get_config(int argc, char** argv, int seed)
{
    Config* config = new Config;
    if (argc != 7)
    {
        printf("4 arguments expected, found %d\n", argc - 1);
        printf("Expected arguments: NUM_POINTS NUM_DIM NUM_CLUST ITORDER IMPRCHOICE INITCHOICE\n");
        IterationOrderFactory::print_doc();
        ImprFactory::print_doc();
        InitializerFactory::print_doc();
        exit(1);
    }
    config->NUM_POINTS = atoi(argv[1]);
    config->NUM_DIM = atoi(argv[2]);
    config->NUM_CLUST = atoi(argv[3]);
    config->IT_ORDER = atoi(argv[4]);
    config->IMPR_CLASS = atoi(argv[5]);
    config->INIT_CHOICE = atoi(argv[6]);
    config->SEED = seed;
    IterationOrder* iteration_order = IterationOrderFactory::create(config, config->IT_ORDER);
    Result* result = new Result(config);
    ImprovementChoice* impr = ImprFactory::create(config->IMPR_CLASS, result, config);
    Initializer* initializer = InitializerFactory::create(config->INIT_CHOICE);
    return std::make_tuple(config, iteration_order,impr, initializer,result);
}
void clean(Config* config, Clustering*clust,IterationOrder* iteration_order, Result* result, ImprovementChoice* impr, Initializer* initializer) {
    int a = 0;
    delete clust->p_c;
    delete clust->n_p_p_c;
    delete clust->c_a;
    delete clust->c_c;
    delete config;
    delete iteration_order;
    delete result;
    delete impr;
    delete initializer;
}
void initialize(Clustering* clustering, Config* config)
{
    // Initialize points coordinates
    clustering->p_c = new float[config->NUM_POINTS * config->NUM_DIM];
    for (int i = 0; i < config->NUM_POINTS * config->NUM_DIM; i++)
        clustering->p_c[i] = prandom(GRID_COORD_MIN, GRID_COORD_MAX);
    // Initialize number of points per cluster
    clustering->n_p_p_c = new int[config->NUM_CLUST];
    for (int i = 0; i < config->NUM_CLUST; i++)
        clustering->n_p_p_c[i] = 0;
    // Initialize cluster intial assignements
    clustering->c_a = new int[config->NUM_POINTS];
    for (int i = 0; i < config->NUM_POINTS; i++)
        clustering->c_a[i] = (int)prandom(0, (int)config->NUM_CLUST-1);
    // Initialize cluster centroids
    clustering->c_c = new float[config->NUM_CLUST * config->NUM_DIM];
    for (int i = 0; i < config->NUM_CLUST * config->NUM_DIM; i++)
        clustering->c_c[i] = 0.;
    // Compute cluster centroids
    for (int i = 0; i < config->NUM_POINTS; i++)
    {
        int cluster_id = clustering->c_a[i];
        for (int j = 0; j < config->NUM_DIM; j++)
        {
            clustering->c_c[cluster_id * config->NUM_DIM + j] += clustering->p_c[i * config->NUM_DIM + j];
        }
        clustering->n_p_p_c[cluster_id]++;
    }
    for (int i = 0; i < config->NUM_CLUST; i++)
    {
        for (int j = 0; j < config->NUM_DIM; j++)
        {
            clustering->c_c[i * config->NUM_DIM + j] /= clustering->n_p_p_c[i];
        }
    }
}
