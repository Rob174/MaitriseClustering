
#ifndef CLUSTERING_H
#define CLUSTERING_H
#include <stdio.h>
#include <stdlib.h>
#include <tuple>
#include <iostream>
#include "iteration_order.h"

typedef struct
{
    double **p_c; // points coor
    int *c_a;     // cluster assignements
    int *n_p_p_c; // num_pts_per_clust
    double **c_c; // cluster_centroids
} Clustering;
typedef struct
{
    int NUM_POINTS;
    int NUM_DIM;
    int NUM_CLUST;
} Config;
std::tuple<Config *, IterationOrder *> get_config(int argc, char **argv);
void initialize(Clustering *clustering, Config *config);
#endif