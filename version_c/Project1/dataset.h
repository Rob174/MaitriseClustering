#ifndef DATASET_H
#define DATASET_H
#include "H5Cpp.h"
#include "clustering.h"
#include "results.h"
#include <string>
#include <list>
#include <iostream>
#include <filesystem>
#include<sstream>
using namespace H5;
const H5std_string FILE_NAME("dataset.hdf5");

void create_dataset(Result* res, Clustering* init_clust, Clustering* final_clust);
#endif // !DATASET_H
