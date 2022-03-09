#ifndef RESULT_H
#define RESULT_H
#include <chrono>
#include <iostream>
using hrclk_t = std::chrono::high_resolution_clock;
using timepoint_t = std::chrono::high_resolution_clock::time_point;
class Result {
private:
	int num_iter;
	int num_iter_glob;
	double init_cost;
	double final_cost;
	std::chrono::steady_clock::time_point time_start;
	double duration;
public:
	Result();
	void set_next_iter();
	void set_next_iter_glob();
	void set_final_cost(double final_cost);
	void set_init_cost(double init_cost);
	void set_time_start();
	void set_time_end();
	void print_results();
};
#endif
