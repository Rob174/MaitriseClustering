#ifndef IMPROVEMENTCHOICE_H
#define IMPROVEMENTCHOICE_H
#include <iostream>
#include "results.h"
#include "operations.h"
#include "utils.h"
#define UNINITIALIZED -1
class ClusteringChoice {
public:
	int i;
	int l;
	int j;
	float vij;
	int counter_not_closest;
	ClusteringChoice() : i(UNINITIALIZED), l(UNINITIALIZED), j(UNINITIALIZED), vij(0), counter_not_closest(UNINITIALIZED){};
};
class ImprovementChoice {
private:
	Result* result;
public:
	ImprovementChoice(Result* result) : result(result) {};
	virtual void choose_solution(ClusteringChoice* choice, Clustering* clustering, float sugg_vij, int sugg_i, int sugg_l, int sugg_j);
	virtual bool stop_loop();
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
	void choose_solution(ClusteringChoice* choice, Clustering* clustering, float sugg_vij, int sugg_i, int sugg_l, int sugg_j);
};
class ImprFactory {
public:
	static ImprovementChoice* create(int identifier,Result*result, Config*config);
	static void print_doc();
};
#endif
