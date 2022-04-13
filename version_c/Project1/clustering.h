
#ifndef CLUSTERING_H
#define CLUSTERING_H
#include <stdio.h>
#include <stdlib.h>
#include <tuple>
#include <functional>
#include <iostream>
#include "constants.h"
#include "results.h"
#include "iteration_order.h"
#include "improvementChoice.h"
#include "utils.h"
#include "kmeans.h"
#include <math.h> 
#include <string.h> 
#include <random>
class ImprovementChoice;
class Initializer;
std::tuple<Config*, IterationOrder*, ImprovementChoice*, Initializer*, Result*> get_config(int argc, char** argv, long seed_points, long seed_assign);
void clean(Config* config, Clustering* clust, Clustering* initial_clustering, IterationOrder* iteration_order, Result* result, ImprovementChoice* impr, Initializer* initializer);
void initialize(Clustering* clustering, Config* config);
Clustering* deepcopy(Clustering* clust, Config* config);
std::tuple<int, char**> random_argv(int loop_id, int init);
#endif