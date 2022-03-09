#ifndef IMPROVEMENTCHOICE_H
#define IMPROVEMENTCHOICE_H
#include "results.h"
#include <iostream>
class ImprovementChoice {
private:
	Result* result;
public:
	ImprovementChoice(Result* result) : result(result) {};
	virtual bool stop_loop(float vij);
};
class BestImpr : public ImprovementChoice {
public:
	BestImpr(Result* result) : ImprovementChoice(result) {};
	bool stop_loop(float vij);
};
class FirstImpr : public ImprovementChoice {
public:
	FirstImpr(Result* result) : ImprovementChoice(result) {};
	bool stop_loop(float vij);
};
class ImprFactory {
public:
	static ImprovementChoice* create(int identifier,Result*result);
	static void print_doc();
};
#endif
