#include "results.h"
Result::Result() : num_iter(0), num_iter_glob(0), init_cost(0), final_cost(0), time_start(std::chrono::steady_clock::now()), duration(-1)
{
}
void Result::set_next_iter()
{
	this->num_iter++;
}

void Result::set_next_iter_glob()
{
	this->num_iter_glob++;
}

void Result::set_final_cost(double final_cost)
{
	this->final_cost = final_cost;
}
void Result::set_init_cost(double init_cost)
{
	this->init_cost = init_cost;
}

void Result::set_time_end()
{
	this->duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - this->time_start).count();
}

void Result::set_time_start()
{
	this->time_start = std::chrono::steady_clock::now();
}
void Result::print_results() {
	std::cout << "num_iter:" << this->num_iter << ",num_iter_glob:" << this->num_iter_glob << ",init_cost:" << this->init_cost << ",final_cost:" << this->final_cost << ",duration:" << this->duration << std::endl;
}
