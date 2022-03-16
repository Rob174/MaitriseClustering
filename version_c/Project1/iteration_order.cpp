#include "iteration_order.h"

void IterationOrder::restart(int curr_clust_id)
{
    this->i = 0;
    this->curr_clust_id = curr_clust_id;
};
int BACK::next()
{
    if (this->i >= this->config->NUM_CLUST)
        return -1;
    else if (this->i == this->curr_clust_id)
    {
        this->i++;
        return next();
    }
    else
    {
        int a = this->i;
        this->i++;
        return a;
    }
}
int CURR::next()
{
    if (this->i >= this->config->NUM_CLUST-1)
    {
        return -1;
    }
    int a = (this->i + this->curr_clust_id + 1) % this->config->NUM_CLUST;
    this->i++;
    return a;
}
int* RANDOM::random_permutation(int size)
{
    int* perm = new int[size];
    for (int i = 0; i < size; i++)
        perm[i] = i;
    for (int i = 0; i < size; i++)
    {
        int r = rand() % size;
        int tmp = perm[i];
        perm[i] = perm[r];
        perm[r] = tmp;
    }
    return perm;
}
void RANDOM::restart(int curr_clust_id)
{
    IterationOrder::restart(curr_clust_id);
    this->perm = random_permutation(this->config->NUM_CLUST);
}
int RANDOM::next()
{
    if (this->i >= this->config->NUM_CLUST)
        return -1;
    int a = this->perm[this->i];
    if (a == this->curr_clust_id)
    {
        this->i++;
        return next();
    }
    else
    {
        this->i++;
        return a;
    }
}
void RANDOM::end_loop() {
    delete this->perm;
}
void IterationOrderFactory::print_doc()
{
    std::cout << "IterationOrder:BACK (0) CURR (1) RANDOM (2)" << std::endl;
}
IterationOrder* IterationOrderFactory::create(Config* config, int type)
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
        std::cout << "Wrong argument IterationOrder " << type << std::endl;
        exit(1);
    }
}