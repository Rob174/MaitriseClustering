#ifndef OPERATIONS_H
#define OPERATIONS_H
#include "clustering.h"

float initial_cost(Clustering* clustering, Config* config);
float cost_modif(Clustering* c, int f, int t, float* point, Config* config);
void update(Clustering* c, int f, int t, int p, Config* config);
#endif
