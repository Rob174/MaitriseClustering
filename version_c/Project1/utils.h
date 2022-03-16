#ifndef UTILS_H
#define UTILS_H
#include <time.h>
#include "clustering.h"

float dist(float* p1, float* p2, Config* config);
bool same_points(float* p1, float* p2, Config* config);
float prandom(float min, float max);
#endif