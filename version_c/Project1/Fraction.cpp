#include "Fraction.h"
int Fraction::gcd(int a, int b) {
    if (b == 0)
        return a;
    else 
        return this->gcd(b, a % b);
}
