#include "improvementChoice.h"

bool BestImpr::stop_loop(float vij)
{
	return false;
}

bool FirstImpr::stop_loop(float vij)
{
	return true;
}

ImprovementChoice* ImprFactory::create(int identifier,Result*res)
{
	switch (identifier)
    {
    case 0:
        return new BestImpr(res);
    case 1:
        return new FirstImpr(res);
    default:
        return new BestImpr(res);
    }
}

void ImprFactory::print_doc()
{
	std::cout << "IMPR: BEST(0), FIRST(1), DELAYED(2)" << std::endl;
}

bool ImprovementChoice::stop_loop(float vij)
{
    return false;
}
