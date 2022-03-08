
#ifndef CLUSTERING_H
#define CLUSTERING_H
#include <stdio.h>
#include <stdlib.h>
#include <tuple>
#include <iostream>
#include "iteration_order.h"
#include "constants.h"

std::tuple<Config *, IterationOrder *> get_config(int argc, char **argv);
void initialize(Clustering *clustering, Config *config);
#endif