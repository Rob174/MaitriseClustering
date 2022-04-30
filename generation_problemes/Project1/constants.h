#ifndef CONSTANTS_H
#define CONSTANTS_H
#include <string>
typedef struct _Config Config;
class Clustering {
public:
    double* p_c; // points coor
    int* c_a;     // cluster assignements
    int* n_p_p_c; // num_pts_per_clust
    double* c_c; // cluster_centroids
};
typedef struct _Config
{
    int NUM_POINTS;
    int NUM_DIM;
    int NUM_CLUST;
    int IMPR_CLASS;
    int IT_ORDER;
    int INIT_CHOICE;
    long SEED_POINTS;
    long SEED_ASSIGN;
}Config;

#endif