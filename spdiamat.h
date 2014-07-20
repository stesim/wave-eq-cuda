#pragma once

//#define REAL_FLOAT

#ifdef REAL_FLOAT
#define real float
#define real2 float2
#else
#define real double
#define real2 double2
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
