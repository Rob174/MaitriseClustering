#include "dataset.h"
void create_dataset(Result*res, Clustering*init_clust, Clustering* final_clust) {
    try {
        // Turn off the auto-printing when failure occurs so that we can
        // handle the errors appropriately
        Exception::dontPrint();

        // Create a new file using the default property lists.
        //check if exists and if not create
        if (!std::filesystem::exists(FILE_NAME)) {
            std::cout << "file does not exist" << std::endl;
            exit(1);
        }
        std::string identifier;
        std::ostringstream os;
        os << res->get_config()->SEED_POINTS << "," << res->get_config()->SEED_ASSIGN << "," << res->get_config()->IMPR_CLASS << "," << res->get_config()->NUM_CLUST << "," << res->get_config()->INIT_CHOICE << "," << res->get_config()->IT_ORDER;
        identifier = os.str();

        H5File file(FILE_NAME, H5F_ACC_RDWR);
        // Save input output of algorithm
        Group group1(file.openGroup("metadata"));
        const int RANK1 = 1;
        hsize_t dims1[RANK1];
        dims1[0] = 12;
        DataSpace dataspace1(RANK1, dims1);
        DataSet dataset1 = group1.createDataSet(identifier, PredType::NATIVE_DOUBLE, dataspace1);
        std::vector<double> *results = res->get_result();
        double *data = new double[results->size()];
        for (std::size_t i = 0; i < results->size(); ++i) {
            data[i] = (*results)[i];
        }
        dataset1.write(data, PredType::NATIVE_DOUBLE);
        delete results;

        // Save points coordinates
        Group group2(file.openGroup("points_coords"));
        const int RANK2 = 1;
        hsize_t dims2[RANK2];
        dims2[0] = res->get_config()->NUM_POINTS * res->get_config()->NUM_DIM;
        DataSpace dataspace2(RANK2, dims2);

        DataSet dataset2 = group2.createDataSet(identifier, PredType::NATIVE_DOUBLE, dataspace2);
        dataset2.write(init_clust->p_c, PredType::NATIVE_DOUBLE);

        //Save initial assignements
        Group group3(file.openGroup("init_assignements"));
        const int RANK3 = 1;
        hsize_t dims3[RANK3];
        dims3[0] = res->get_config()->NUM_POINTS;
        DataSpace dataspace3(RANK3, dims3);

        DataSet dataset3 = group3.createDataSet(identifier, PredType::NATIVE_INT, dataspace3);
        dataset3.write(init_clust->c_a, PredType::NATIVE_INT);

        //Save final assignements
        Group group4(file.openGroup("final_assignements"));
        const int RANK4 = 1;
        hsize_t dims4[RANK4];
        dims4[0] = res->get_config()->NUM_POINTS;
        DataSpace dataspace4(RANK4, dims4);

        DataSet dataset4 = group4.createDataSet(identifier, PredType::NATIVE_INT, dataspace4);
        dataset4.write(final_clust->c_a, PredType::NATIVE_INT);
        delete data;
    } 
    catch (FileIException error) {
        error.printErrorStack();
        std::cout << "Error writing dataset: H5File operations" << std::endl;
        exit(-1);
    }
    catch (DataSetIException error) {
        error.printErrorStack();
        std::cout << "Error writing dataset: DataSet operations" << std::endl;
        exit(-1);
    }
    catch (DataSpaceIException error) {
        error.printErrorStack();
        std::cout << "Error writing dataset: DataSpace operations" << std::endl;
        exit(-1);
    }
}

