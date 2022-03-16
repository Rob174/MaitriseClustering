#ifndef OPERATIONS_H
#define OPERATIONS_H
#include "constants.h"
#include <tuple>

float initial_cost(Clustering* clustering, Config* config);
float cost_modif(Clustering* c, int f, int t, float* point, Config* config);
void update(Clustering* c, int f, int t, int p, Config* config);
std::tuple<float*, float*> compute_new_centroids(Clustering const* c, int f, int t, int p, Config* config);
#endif
