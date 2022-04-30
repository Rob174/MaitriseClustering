#ifndef OPERATIONS_H
#define OPERATIONS_H
#include "constants.h"
#include <tuple>

double initial_cost(Clustering* clustering, Config* config);
double cost_modif(Clustering* c, int f, int t, double* point, Config* config);
void update(Clustering* c, int f, int t, int p, Config* config);
std::tuple<double*, double*> compute_new_centroids(Clustering const* c, int f, int t, int p, Config* config);
#endif
