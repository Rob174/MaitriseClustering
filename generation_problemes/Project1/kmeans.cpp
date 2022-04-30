#include "kmeans.h"

void Initializer::initialize(Clustering* clustering, Config* config)
{
	return;
}
void KMeansPlusInitializer::update_centroid(Clustering* clustering, Config* config) {
    //Reinitialize centroids
    for (int i = 0; i < config->NUM_CLUST; i++) {
        for (int d = 0; d < config->NUM_DIM; d++) {
            clustering->c_c[i * config->NUM_DIM + d] = 0.;
        }
    }
    // Update centroids
    // 1. sum points coordinates
    for (int i = 0; i < config->NUM_POINTS; i++) {
        for (int d = 0; d < config->NUM_DIM; d++) {
            clustering->c_c[clustering->c_a[i] * config->NUM_DIM + d] += clustering->p_c[i * config->NUM_DIM + d];
        }
    }
    //2. Divide by number of points (already kept up to date in the algorithm)
    for (int i = 0; i < config->NUM_CLUST; i++) {
        for (int d = 0; d < config->NUM_DIM; d++) {
            clustering->c_c[i * config->NUM_DIM + d] = clustering->c_c[i * config->NUM_DIM + d] / (double)(clustering->n_p_p_c[i]);
        }
    }
}
void KMeansPlusInitializer::initialize(Clustering* clustering, Config* config) {
    bool assignement_changed = true;
    int iter = 0;
    int iter2 = 0;
    while (assignement_changed) {
        assignement_changed = false;
        double* previous_assignement = new double[config->NUM_POINTS];
        for (int i = 0; i < config->NUM_POINTS; i++)
            previous_assignement[i] = clustering->c_a[i];
        for (int i = 0; i < config->NUM_POINTS; i++) {
            double* point = &(clustering->p_c[i * config->NUM_DIM]);
            //We assign the point to its closest centroid
            double best_d = std::numeric_limits<double>::max();
            int best_c = -1;
            for (int c = 0; c < config->NUM_CLUST; c++) {
                double d = dist(point, &(clustering->p_c[c]), config);
                if (d < best_d) {
                    best_d = d;
                    best_c = c;
                }
            }
            // If better centroid: assign the point to this centroid
            if (best_c != clustering->c_a[i]) {
                if (clustering->n_p_p_c[clustering->c_a[i]] <= 0)
                    exit(1);
                clustering->n_p_p_c[clustering->c_a[i]]--;
                clustering->n_p_p_c[best_c]++;
                clustering->c_a[i] = best_c;
                iter++;
            }
        }
        iter2++;
        this->update_centroid(clustering, config);
        // KMeans+ : Check if empty clusters and assign them to the more distant clusters
        //1. Which and How many empty clusters
        std::list<int> empty_clust;
        int num_empty = 0;
        for (int i = 0; i < config->NUM_CLUST; i++) {
            if (clustering->n_p_p_c[i] == 0) {
                empty_clust.push_back(i);
                num_empty++;
            }
        }
        //2. Get biggest top n distances to centroids, n number of empty clusters
        auto cmp = [](std::tuple<int, double> t1, std::tuple<int, double> t2) {
            //with first corresponding point index and then distance
            const auto& [i1, d1] = t1;
            const auto& [i2, d2] = t2;
            return d1 > d2;
        };
        // REMINDER: set sorted : biggest element first : (binary tree)
        std::set<std::tuple<int, double>, decltype(cmp)> distances;
        for (int i = 0; i < config->NUM_POINTS; i++) {
            for (int j = 0; j < config->NUM_CLUST; j++) {
                double distance = dist(&(clustering->p_c[i * config->NUM_DIM]), &(clustering->c_c[j * config->NUM_DIM]), config);
                if ((int)distances.size() < num_empty || (distances.begin() != distances.end() && distance > (double)std::get<1>(*std::prev(distances.end()))))
                {
                    distances.insert(std::make_tuple(i, distance));
                    if ((int)distances.size() > num_empty)
                        distances.erase(std::prev(distances.end()));
                }
            }
        }
        //3. Affect most distant points to empty centroids
        auto it_d = distances.begin();
        auto it_c = empty_clust.begin();
        while (it_d != distances.end())
        {
            //affect point to the empty clust
            clustering->n_p_p_c[clustering->c_a[std::get<0>(*it_d)]]--;
            clustering->n_p_p_c[*it_c]++;
            clustering->c_a[std::get<0>(*it_d)] = *it_c;
            //next
            ++it_d;
            ++it_c;
        }
        this->update_centroid(clustering, config);
        for (int i = 0; i < config->NUM_POINTS; i++) {
            if (clustering->c_a[i] != previous_assignement[i]) {
                assignement_changed = true;
                break;
            }
        }
        delete previous_assignement;
    }
}


Initializer* InitializerFactory::create(int type)
{
    switch (type)
    {
    case(0):
        return new Initializer();
    case(1):
        return new KMeansPlusInitializer();
    default:
        std::cout << "Wrong argument Initializer "<< type << std::endl;
        exit(1);
    }
}
void InitializerFactory::print_doc() {
    return;
}
