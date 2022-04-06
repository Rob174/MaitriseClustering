#ifndef RESULT_H
#define RESULT_H
#include <chrono>
#include <iostream>
#include "constants.h"
#include <iomanip>
#include <vector>
using hrclk_t = std::chrono::high_resolution_clock;
using timepoint_t = std::chrono::high_resolution_clock::time_point;
class Result {
private:
	Config* config;
	int num_iter;
	int num_iter_glob;
	double init_cost;
	double final_cost;
	std::chrono::steady_clock::time_point time_start;
	double duration;
public:
	Result(Config* config) : config(config) {};
	Result(Config* config, int num_iter, int num_iter_glob, double init_cost, double final_cost, double duration) : 
		config(config),num_iter(num_iter),num_iter_glob(num_iter_glob), init_cost(init_cost), final_cost(final_cost), duration(duration) {};
	void set_next_iter();
	void set_next_iter_glob();
	void set_final_cost(double final_cost);
	void set_init_cost(double init_cost);
	void set_time_start();
	void set_time_end();
	double get_time_elapsed();
	void print_results();
	std::vector<double>* get_result();
	Config* get_config() { return this->config; }
};
#endif
