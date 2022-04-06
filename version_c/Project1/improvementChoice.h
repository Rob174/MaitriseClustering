#ifndef IMPROVEMENTCHOICE_H
#define IMPROVEMENTCHOICE_H
#include <iostream>
#include <limits>
#include <vector>
#include <tuple>
#include "results.h"
#include "operations.h"
#include "utils.h"

#define UNINITIALIZED -1
class ClusteringChoice {
public:
	int i;
	int l;
	int j;
	double vij;
	int counter_not_closest;
	ClusteringChoice() : i(UNINITIALIZED), l(UNINITIALIZED), j(UNINITIALIZED), vij(0), counter_not_closest(UNINITIALIZED){};
};
class ImprovementChoice {
private:
	Result* result;
public:
	ImprovementChoice(Result* result) : result(result) {};
	virtual void choose_solution(ClusteringChoice* choice, Clustering* clustering, double sugg_vij, int sugg_i, int sugg_l, int sugg_j);
	virtual bool stop_loop();
	virtual void initialize(Clustering*clustering) { return; };
	virtual void after_choice();
};
class BestImpr : public ImprovementChoice {
public:
	BestImpr(Result* result) : ImprovementChoice(result) {};
	bool stop_loop();
};
class FirstImpr : public ImprovementChoice {
public:
	FirstImpr(Result* result) : ImprovementChoice(result) {};
	bool stop_loop();
};
class DelayedImpr1 : public BestImpr {
private:
	Config* config;
public:
	DelayedImpr1(Result* result,Config*config) : BestImpr(result),config(config) {};
	void choose_solution(ClusteringChoice* choice, Clustering* clustering, double sugg_vij, int sugg_i, int sugg_l, int sugg_j);
};
class ClosestCentr {
public:
	double dist;
	int centr_id;
	ClosestCentr() : dist(std::numeric_limits<double>::max()), centr_id(-1) {};
	ClosestCentr(double dist,int centroid) : dist(dist), centr_id(centroid) {};
};
class DelayedImpr2 : public BestImpr {
private:
	Config* config;
	ClosestCentr* d_closest_centr;
	std::vector<std::tuple<int, int, double>> *modifications;
	void apply_closestCentr_modif();
	void distanceClosestCentroid(Clustering* clustering);
public:
	DelayedImpr2(Result* result, Config* config) : BestImpr(result), config(config), d_closest_centr(nullptr), 
		modifications() {
		this->d_closest_centr = new ClosestCentr[config->NUM_POINTS];
	};
	~DelayedImpr2() {
		delete[] this->d_closest_centr;
		delete this->modifications;
	}
	void initialize(Clustering* clustering) { this->distanceClosestCentroid(clustering); }
	void choose_solution(ClusteringChoice* choice, Clustering* clustering, double sugg_vij, int sugg_i, int sugg_l, int sugg_j);
	void after_choice();
};
class ImprFactory {
public:
	static ImprovementChoice* create(int identifier,Result*result, Config*config);
	static void print_doc();
};
#endif
