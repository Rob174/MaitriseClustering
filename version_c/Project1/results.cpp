#include "results.h"

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
double Result::get_time_elapsed() {
	return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - this->time_start).count();;
}
void Result::set_time_end()
{
	this->duration = this->get_time_elapsed();
}

void Result::set_time_start()
{
	this->time_start = std::chrono::steady_clock::now();
}
void Result::print_results() {
	std::cout << "seed:" << this->config->SEED
			  << ",num_clusters:" << this->config->NUM_CLUST
		      << ",num_points:" << this->config->NUM_POINTS
		      << ",init_type:" << this->config->INIT_CHOICE 
		      << ",ImprClass:" << this->config->IMPR_CLASS 
		      << ",iteration_order:" << this->config->IT_ORDER 
		      << ",init_cost:" << this->init_cost 
		      << ",end_cost:" << this->final_cost 
		      << ",num_iter:" << this->num_iter 
		      << ",num_iter_tot:" << this->num_iter_glob 
		      << ",time:" << this->duration 
		<< std::endl;
}
