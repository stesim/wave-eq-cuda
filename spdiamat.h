#pragma once

#ifndef real
#ifdef REAL_FLOAT
typedef float real;
#else
typedef double real;
#endif
#endif

struct SpDiaMat
{
	unsigned int n;
	unsigned int diags;
	int* offsets;
	real* values;
	unsigned int valuesSize;
	unsigned int constPos;
};
