#include "improvementChoice.h"

ImprovementChoice* ImprFactory::create(int identifier, Result* res,Config*config)
{
    switch (identifier)
    {
    case 0:
        return new BestImpr(res);
    case 1:
        return new FirstImpr(res);
    case 2:
        return new DelayedImpr1(res, config);
    case 3:
        return new DelayedImpr2(res, config);
    default:
        std::cout << "Wrong argument ImprovementChoice " << identifier << std::endl;
        exit(1);
    }
}

void ImprFactory::print_doc()
{
}

bool BestImpr::stop_loop()
{
	return false;
}

bool FirstImpr::stop_loop()
{
	return true;
}


void ImprovementChoice::choose_solution(ClusteringChoice*choice, Clustering* clustering, double sugg_vij, int sugg_i, int sugg_l, int sugg_j)
{
    choice->vij = sugg_vij;
    choice->i = sugg_i;
    choice->l = sugg_l;
    choice->j = sugg_j;
}

bool ImprovementChoice::stop_loop()
{
    return false;
}
void ImprovementChoice::after_choice()
{
    return;
}
void DelayedImpr1::choose_solution(ClusteringChoice* choice, Clustering* clustering, double sugg_vij, int sugg_i, int sugg_l, int sugg_j)
{
    int counter_not_closest = 0;
    auto former_dest_clust = compute_new_centroids(clustering, sugg_l, sugg_j, sugg_i, this->config);
    double* new_f_center = std::get<0>(former_dest_clust);
    double* new_t_center = std::get<1>(former_dest_clust);

    for (int q = 0; q < this->config->NUM_POINTS; q++) {
        int min_centr = -1;
        double min_dist = std::numeric_limits<double>::max();
        for (int z = 0; z < this->config->NUM_CLUST; z++) {
            double* centroid;
            if (z == sugg_l) {
                centroid = new_f_center;
            }
            else if (z == sugg_j) {
                centroid = new_t_center;
            }
            else {
                centroid = &(clustering->c_c[z * this->config->NUM_DIM]);
            }
            double d = dist(centroid, &(clustering->p_c[q * this->config->NUM_DIM]), this->config);

            if (d < min_dist) {
                min_dist = d;
                min_centr = z;
            }
        }
        if ((q == sugg_i && min_centr != sugg_j) ||
            (q != sugg_i && min_centr != clustering->c_a[q])) {
            counter_not_closest++;
            std::cout << "q:" << q << ";min_centr:" << min_centr << std::endl;
        }

    }
    delete new_f_center;
    delete new_t_center;
    if (counter_not_closest > choice->counter_not_closest) { // First breakpoint  counter_not_closest = 932
        ImprovementChoice::choose_solution(choice, clustering, sugg_vij, sugg_i, sugg_l, sugg_j);
        choice->counter_not_closest = counter_not_closest;
    }
}

void DelayedImpr2::distanceClosestCentroid(Clustering* clustering) {
    this->d_closest_centr = new ClosestCentr[this->config->NUM_POINTS];
    for (int i = 0; i < this->config->NUM_POINTS; i++) {
        for (int j = 0; j < this->config->NUM_CLUST; j++) {
            double d = dist(&(clustering->p_c[i * this->config->NUM_DIM]), &(clustering->c_c[j*this->config->NUM_DIM]), this->config);
            if (d < this->d_closest_centr[i].dist) {
                this->d_closest_centr[i].dist = d;
                this->d_closest_centr[i].centr_id = j;
            }
        }
    }
}
void DelayedImpr2::choose_solution(ClusteringChoice* choice, Clustering* clustering, double sugg_vij, int sugg_i, int sugg_l, int sugg_j) {
    // TO BE DEBUGGED
    auto former_dest_clust = compute_new_centroids(clustering, sugg_l, sugg_j, sugg_i, this->config);
    double* new_f_center = std::get<0>(former_dest_clust);
    double* new_t_center = std::get<1>(former_dest_clust);
    std::vector<std::tuple<int, int, double>>* modifications = new std::vector<std::tuple<int, int, double>>;

    int counter_not_closest = 0;
    for (int i = 0; i < this->config->NUM_POINTS; i++) {
        //OPTI: enlever le premier cas
        if (this->d_closest_centr[i].centr_id != sugg_l && this->d_closest_centr[i].centr_id != sugg_j // closest center not moved
            && clustering->c_a[i] != sugg_l && clustering->c_a[i] != sugg_j) { // assigned cluster not moved
            counter_not_closest += (int)(this->d_closest_centr[i].centr_id != clustering->c_a[i]);
            if(this->d_closest_centr[i].centr_id != clustering->c_a[i])
                std::cout << "q:" << i << ";min_centr:" << this->d_closest_centr[i].centr_id << std::endl;
        }
        else {
            // Recompute closest cluster centroid
            double min_dist = std::numeric_limits<double>::max();
            int min_clust = -1;
            for (int c = 0; c < config->NUM_CLUST; c++) {//OPTI: def structure où peut facilement extraire un voisinage
                //OU ordonner les n plus proches clusters (de toute façon le calcul est effectué pour tous les centroids)
                // puis ici au lieu de parcourir de l'index 0 au dernier
                // parcourir de l'ancien plus proche index au plus lointain 
                // JUSTIF: ne change pas la position des points et modifie que peu de centroids.
                // FAUX : de toute façon doit tout reparcourir pour vérifier que bien le centroid le plus proche
                double* centroid;
                if (c == sugg_l)
                    centroid = new_f_center;
                else if (c == sugg_j)
                    centroid = new_t_center;
                else {
                    centroid = &(clustering->c_c[c * config->NUM_DIM]);
                }
                double d = dist(centroid, &(clustering->p_c[i]), config);
                if (d < min_dist) {
                    min_dist = d;
                    min_clust = c;
                }
            }
            modifications->push_back(std::make_tuple(i, min_clust, min_dist));
            counter_not_closest += (int)(min_clust != clustering->c_a[i]);
            if(min_clust != clustering->c_a[i])
                std::cout << "q:" << i << ";min_centr:" << min_clust << std::endl;
        }
    }
    if (counter_not_closest > choice->counter_not_closest) {// First breakpoint  counter_not_closest = 919
        this->modifications = modifications;
        ImprovementChoice::choose_solution(choice, clustering, sugg_vij, sugg_i, sugg_l, sugg_j);
        choice->counter_not_closest = counter_not_closest;
    }
    else {
        delete modifications;
    }
}
void DelayedImpr2::apply_closestCentr_modif() {
    for (auto& closest : *this->modifications) {
        this->d_closest_centr[std::get<0>(closest)].centr_id = std::get<1>(closest);
        this->d_closest_centr[std::get<0>(closest)].dist = std::get<2>(closest);
    }
    delete this->modifications;
}
void DelayedImpr2::after_choice() {
    this->apply_closestCentr_modif();
}