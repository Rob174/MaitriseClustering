#ifndef CONSTANTS_H
#define CONSTANTS_H

typedef struct _Config Config;
typedef struct _Clustering Clustering;
typedef struct _Clustering
{
    float* p_c; // points coor
    int* c_a;     // cluster assignements
    int* n_p_p_c; // num_pts_per_clust
    float* c_c; // cluster_centroids
}Clustering;
typedef struct _Config
{
    int NUM_POINTS;
    int NUM_DIM;
    int NUM_CLUST;
    int IMPR_CLASS;
    int IT_ORDER;
    int INIT_CHOICE;
    int SEED;
}Config;

#endif