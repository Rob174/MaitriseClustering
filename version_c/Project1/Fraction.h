#pragma once
class Fraction
{
private:
	double num;
	double denom;
	int gcd(int num, int denom);
public:
	Fraction(int num, int denom) : num(num), denom(denom) {};

};

