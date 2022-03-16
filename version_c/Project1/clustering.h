
#ifndef CLUSTERING_H
#define CLUSTERING_H
#include <stdio.h>
#include <stdlib.h>
#include <tuple>
#include <iostream>
#include "constants.h"
#include "results.h"
#include "iteration_order.h"
#include "improvementChoice.h"
#include "utils.h"
#include "kmeans.h"
class ImprovementChoice;
class Initializer;
std::tuple<Config*, IterationOrder*, ImprovementChoice*, Initializer*, Result*> get_config(int argc, char** argv,int seed);
void clean(Config* config, Clustering* clust, IterationOrder* iteration_order, Result* result, ImprovementChoice* impr, Initializer* initializer);
void initialize(Clustering* clustering, Config* config);
#endif