#ifndef KMEANS_H
#define KMEANS_H
#include "constants.h"
#include "utils.h"
#include <math.h>
#include <iostream>
#include <tuple>
#include <set>
#include <list>

class Initializer
{
public:
	virtual void initialize(Clustering* clustering, Config* config);
};
class KMeansPlusInitializer : public Initializer {
public:
	void initialize(Clustering* clustering, Config* config);
	void update_centroid(Clustering* clustering, Config* config);
};
class InitializerFactory {
public:
	static Initializer* create(int type);
	static void print_doc();
};
#endif
