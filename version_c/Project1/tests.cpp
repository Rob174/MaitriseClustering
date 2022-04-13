#include "tests.h"
#include "utils.h"
#include <set>
#include <chrono>
#include <thread>
#include <iostream>
#include <array>
#include<sstream>
#include <string>
//0 : pass ; 1 : fail
void print(std::string name) {
	std::cout << "Test " << name<<": ";
}
void succ_result() {
	std::cout << "\x1B[32mPASS\033[0m" << std::endl;
}
int test_seed_diff() {
	print("Seed different (guaranted resolution 1ms)");

	std::set<long> already_seen;
	for (int i = 0; i < 100; i++) {
		for (int rep = 0; rep < 3; rep++) {
			long seed = get_seed(rep);
			if (already_seen.size() > 1 && already_seen.find(seed) != already_seen.end()) {
				std::cout << "\x1B[31mFAILED:\033[0m Duplicated key " << seed << " for " << already_seen.size() << " seed seen" << std::endl;
				return 1;
			}
			already_seen.insert(seed);
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	}
	succ_result();
	return 0;
}
int test_algos() {
	print("BI FI");
	std::cout << std::endl;
	std::array<int, 2> num_pts = {20,1000};
	std::array<int, 2> arr_init = { 0,1};
	std::array<int, 3> num_clusters = { 2,25,50 };
	std::array<int, 3> it_orders = { 0,1 };
	std::array<int, 3> algos = { 0,1 };

	int i = 0;
	for (auto& num_pt : num_pts) {
		for (auto& init : arr_init) {
			for (int i_clust = 0; i_clust < 3; i_clust++) {
				int num_clust = i_clust == 0 ? num_clusters.at(i_clust) : (int)((float)(num_clusters.at(i_clust))/100. *(float)(num_pt));
				for (auto& it_order : it_orders) {
					for (auto& algo : algos) {
						if (num_clust == 0 || num_clust == 1 || num_clust >= num_pt)
							continue;
						int argc = 7;
						std::string
							s_num_pt = std::to_string(num_pt),
							s_init = std::to_string(init),
							s_num_clust = std::to_string(num_clust),
							s_it_order = std::to_string(it_order),
							s_algo = std::to_string(algo);
						//NUM_POINTS NUM_DIM NUM_CLUST ITORDER IMPRCHOICE INITCHOICE
						char* argv[] = {
							(char*)"prog_name",
							const_cast<char*>(s_num_pt.c_str()),
							(char*)"2",
							const_cast<char*>(s_num_clust.c_str()),
							const_cast<char*>(s_it_order.c_str()),
							const_cast<char*>(s_algo.c_str()),
							const_cast<char*>(s_init.c_str())
						};
						std::cout << "\x1B[33m--> subtest " << s_num_pt << " pts;" << s_num_clust 
							<< " clusters;" << s_it_order << " order;" << s_algo << " algorithm;" << s_init << " initialization\033[0m " << std::flush;
						std::random_device rd;
						run(argc, argv, 0, false, rd(), rd());
						succ_result();
						i++;
					}
				}
			}
		}
	}
	std::cout << "END TEST " << "BI FI: ";
	succ_result();
	return 0;
}
int test_DI() {
	print("DI");
	std::cout << std::endl;
	std::array<int, 2> num_pts = { 20,1000 };
	std::array<int, 2> arr_init = { 0,1 };
	std::array<int, 2> num_clusters = { 2,10 };
	std::array<int, 3> it_orders = { 0,1 };

	int i = 0;
	for (auto& num_pt : num_pts) {
		for (auto& init : arr_init) {
			for (int i_clust = 0; i_clust < 3; i_clust++) {
				int num_clust = num_clusters.at(i_clust);
				for (auto& it_order : it_orders) {
					if (num_clust == 0 || num_clust == 1 || num_clust >= num_pt)
						continue;
					int argc = 7;
					std::string
						s_num_pt = std::to_string(num_pt),
						s_init = std::to_string(init),
						s_num_clust = std::to_string(num_clust),
						s_it_order = std::to_string(it_order);
					//NUM_POINTS NUM_DIM NUM_CLUST ITORDER IMPRCHOICE INITCHOICE
					char* argv[] = {
						(char*)"prog_name",
						const_cast<char*>(s_num_pt.c_str()),
						(char*)"2",
						const_cast<char*>(s_num_clust.c_str()),
						const_cast<char*>(s_it_order.c_str()),
						(char*)"2",
						const_cast<char*>(s_init.c_str())
					};
					std::cout << "\x1B[33m--> subtest " << s_num_pt << " pts;" << s_num_clust << " clusters;" << s_it_order << " order;" << "DI;" << s_init << " initialization\033[0m "<< std::flush;
					std::random_device rd;
					run(argc, argv, 0, false, rd(), rd());
					succ_result();
					i++;
				}
			}
		}
	}
	std::cout << "END TEST " << "DI: ";
	succ_result();
	return 0;
}
int test_DI_opti() {
	print("DI optimized version");
	std::cout << std::endl;
	std::array<int, 2> num_pts = { 20,1000 };
	std::array<int, 2> arr_init = { 0,1 };
	std::array<int, 3> num_clusters = { 2,10 ,100};
	std::array<int, 3> it_orders = { 0,1 };

	int i = 0;
	for (auto& num_pt : num_pts) {
		for (auto& init : arr_init) {
			for (int i_clust = 0; i_clust < 3; i_clust++) {
				int num_clust = num_clusters.at(i_clust);
				for (auto& it_order : it_orders) {
					if (num_clust == 0 || num_clust == 1 || num_clust >= num_pt)
						continue;
					int argc = 7;
					std::string
						s_num_pt = std::to_string(num_pt),
						s_init = std::to_string(init),
						s_num_clust = std::to_string(num_clust),
						s_it_order = std::to_string(it_order);
					//NUM_POINTS NUM_DIM NUM_CLUST ITORDER IMPRCHOICE INITCHOICE
					char* argv[] = {
						(char*)"prog_name",
						const_cast<char*>(s_num_pt.c_str()),
						(char*)"2",
						const_cast<char*>(s_num_clust.c_str()),
						const_cast<char*>(s_it_order.c_str()),
						(char*)"3",
						const_cast<char*>(s_init.c_str())
					};
					std::cout << "\x1B[33m--> subtest " << s_num_pt << " pts;" << s_num_clust << " clusters;" << s_it_order << " order;" << "DI optimized;" << s_init << " initialization\033[0m " << std::flush;

					std::random_device rd; 
					run(argc, argv, 0, true, rd(),rd());
					succ_result();
					i++;
				}
			}
		}
	}
	std::cout << "END TEST " << "DI: ";
	succ_result();
	return 0;
}
int test_write_to_hdf5() {
	Config conf = { 1000,2,4,0,0,0,0 };
	Result res(&conf, 100, 1000, 256.2, 201.1, 214569874.3);
	Clustering *init_clust = new Clustering();
	init_clust->c_a = new int[conf.NUM_POINTS];
	for (int i = 0; i < conf.NUM_POINTS; i++) {
		init_clust->c_a[i] = i;
	}
	const int num_poss = conf.NUM_POINTS * conf.NUM_DIM;
	init_clust->p_c = new double[num_poss];
	for (int i = 0; i < conf.NUM_POINTS*conf.NUM_DIM; i++) {
		init_clust->c_a[i] = num_poss - i;
	}

	Clustering* final_clust = new Clustering();
	final_clust->c_a = new int[conf.NUM_POINTS];
	for (int i = 0; i < conf.NUM_POINTS; i++) {
		final_clust->c_a[i] = i;
	}
	final_clust->p_c = new double[num_poss];
	for (int i = 0; i < conf.NUM_POINTS * conf.NUM_DIM; i++) {
		final_clust->c_a[i] = num_poss -i;
	}
	std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;
	create_dataset(&res, init_clust, final_clust);
	return 1;
}
void run_tests() {
	std::vector<int (*)()> test_functions = { 
		//test_seed_diff,
		//test_algos,
		//test_DI,
		//test_DI_opti,
		test_write_to_hdf5
	};
	for (auto& f : test_functions) {
		if (f() != 0)
			break;
	}
	//TODO : check DI opti ttes methodes appelées et ajouter le test.
}