#ifndef UTILS_H
#define UTILS_H
#include <time.h>
#include "clustering.h"

double dist(double* p1, double* p2, Config* config);
bool same_points(double* p1, double* p2, Config* config);
double prandom(double min, double max);
#endif