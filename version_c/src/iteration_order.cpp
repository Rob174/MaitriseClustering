#include "iteration_order.h"

void IterationOrder::restart(int curr_clust_id)
{
    this->i = 0;
    this->curr_clust_id = curr_clust_id;
};
int BACK::next()
{
    if (this->i == this->config->NUM_POINTS)
        return -1;
    else if (this->i == this->curr_clust_id)
    {
        this->i++;
        return next();
    }
    else
    {
        int i = this->i;
        this->i++;
        return i;
    }
}
int CURR::next()
{
    if (this->i == this->config->NUM_POINTS)
        return -1;
    int i = (this->i + this->curr_clust_id + 1) % this->config->NUM_POINTS;
    this->i++;
    return i;
}
int* RANDOM::random_permutation(int n)
{
    int *perm = new int[n];
    for (int i = 0; i < n; i++)
        perm[i] = i;
    for (int i = 0; i < n; i++)
    {
        int r = rand() % n;
        int tmp = perm[i];
        perm[i] = perm[r];
        perm[r] = tmp;
    }
    return perm;
}
void RANDOM::restart(int curr_clust_id)
{
    IterationOrder::restart(curr_clust_id);
    this->perm = random_permutation(this->config->NUM_POINTS);
}
int RANDOM::next()
{
    if (this->i == this->config->NUM_POINTS)
        return -1;
    int i = this->perm[this->i];
    if (i == this->curr_clust_id)
    {
        this->i++;
        return next();
    }
    else
    {
        this->i++;
        return i;
    }
}
void IterationOrderFactory::print_doc()
{
    std::cout << "IterationOrder:BACK (0) CURR (1) RANDOM (2)" << std::endl;
}
IterationOrder *IterationOrderFactory::create(Config *config, int type)
{
    switch (type)
    {
    case 0:
        return new BACK(config);
    case 1:
        return new CURR(config);
    case 2:
        return new RANDOM(config);
    default:
        return new BACK(config);
    }
}