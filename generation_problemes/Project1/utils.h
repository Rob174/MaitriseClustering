#ifndef UTILS_H
#define UTILS_H
#include <time.h>
#include <random>
#include "clustering.h"
#include <process.h>

double dist(double* p1, double* p2, Config* config);
bool same_points(double* p1, double* p2, Config* config);
double prandom(double min, double max);
long get_seed(int loop_index);
#endif