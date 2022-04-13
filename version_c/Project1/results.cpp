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
	std::cout << std::endl;
	std::cout << "seed_points:" << this->config->SEED_POINTS
			  << ",seed_assign:" << this->config->SEED_ASSIGN
			  << ",num_clusters:" << this->config->NUM_CLUST
		      << ",num_points:" << this->config->NUM_POINTS
		      << ",init_type:" << this->config->INIT_CHOICE 
		      << ",ImprClass:" << this->config->IMPR_CLASS 
		      << ",iteration_order:" << this->config->IT_ORDER 
		      << ",init_cost:" << std::fixed << std::setprecision(3) << this->init_cost
		      << ",end_cost:" << this->final_cost 
		      << ",num_iter:" << this->num_iter 
		      << ",num_iter_tot:" << this->num_iter_glob 
		      << ",time:" << this->duration 
		<< std::endl;
	//printf("seed:%d,num_clusters:%d,num_points:%d,init_type:%d,ImprClass:%d,iteration_order:%d,init_cost:%.3f,end_cost:%.3f,num_iter:%d,num_iter_tot:%d,duration:%.3f\n", this->config->SEED, this->config->NUM_CLUST, this->config->NUM_POINTS, this->config->INIT_CHOICE, this->config->IMPR_CLASS, this->config->IT_ORDER, this->init_cost, this->final_cost, this->num_iter, this->num_iter_glob, this->duration);
}
std::vector<double>* Result::get_result() {
	std::vector<double>* to_backup = new std::vector<double>();
	to_backup->push_back((double)this->config->SEED_POINTS);
	to_backup->push_back((double)this->config->SEED_ASSIGN);
	to_backup->push_back((double)this->config->NUM_CLUST);
	to_backup->push_back((double)this->config->NUM_POINTS);
	to_backup->push_back((double)this->config->INIT_CHOICE);
	to_backup->push_back((double)this->config->IMPR_CLASS);
	to_backup->push_back((double)this->config->IT_ORDER);
	to_backup->push_back((double)this->init_cost);
	to_backup->push_back((double)this->final_cost);
	to_backup->push_back((double)this->num_iter);
	to_backup->push_back((double)this->num_iter_glob);
	to_backup->push_back((double)this->duration);
	return to_backup;
}
