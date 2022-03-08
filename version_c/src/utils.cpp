#include "utils.h"

double dist(double *p1, double *p2, Config *config)
{
    double dist = 0.0;
    for (int i = 0; i < config->NUM_DIM; i++)
    {
        dist += (p1[i] - p2[i]) * (p1[i] - p2[i]);
    }
    return dist;
}

double prandom(double min, double max)
{
    return min + ((double)rand() / (double)(RAND_MAX)) * (max - min);
}