#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "tests.h"
#include "algorithm.h"
#include "clustering.h"
#include "utils.h"
int main(int argc, char *argv[])
{
    
    //  External run
    /*
    for (int i = 0; i < 1; i++)
        run(argc, argv,i, true,1);
    */
    
    //Internal run,
    
    long seed = 0;
    for (int i = 0; i < 16000; i++) {
        seed = i/2;
        auto args = random_argv(i, seed);
        int argc = std::get<0>(args);
        char** argv = std::get<1>(args);
        run(argc, argv, i,false,seed= seed);
        for (int a = 0; a < 7; a++)
            delete argv[a];
        delete argv;
    }
    
    //run_tests();
    return 0;
}