#include "utils.h"

float dist(float* p1, float* p2, Config* config)
{
    float dist = 0.0;
    for (int i = 0; i < config->NUM_DIM; i++)
    {
        dist += (p1[i] - p2[i]) * (p1[i] - p2[i]);
    }
    return dist;
}

bool same_points(float* p1, float* p2, Config* config)
{
    bool issame = true;
    for (int i = 0; i < config->NUM_DIM; i++) {
        if (p1[i] != p2[i]) {
            issame = false;
            break;
        }
    }
    return issame;
}

float prandom(float min, float max)
{
    return min + ((float)rand() / (float)(RAND_MAX)) * (max - min);
}
