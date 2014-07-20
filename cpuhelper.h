#pragma once

#include "spdiamat.h"

// matrix / vector helper functions
class CpuHelper
{
public:
	template<typename T>
	inline static T min( T x, T y )
	{
		return ( x < y ? x : y );
	}

	template<typename T>
	inline static T max( T x, T y )
	{
		return ( x > y ? x : y );
	}
	
	template<typename T>
	inline static T abs( T x )
	{
		return ( x >= 0 ? x : -x );
	}

	inline static void mulMatVec( const SpDiaMat& mat, const real* v,
			real* res )
	{
		for( unsigned int i = 0; i < mat.n; ++i )
		{
			res[ i ] = 0.0;
		}
		const real* curDiagValues = mat.values;
		for( unsigned int k = 0; k < mat.diags; ++k )
		{
			int offset = mat.offsets[ k ];
			int resIdx = -min( offset, 0 );
			int vIdx = max( offset, 0 );
			unsigned int diagSize = mat.n - abs( offset );
			for( unsigned int i = 0; i < diagSize; ++i )
			{
				res[ i + resIdx ] +=
					curDiagValues[ i ] * v[ i + vIdx ];
			}
			curDiagValues += diagSize;
		}
	}
};
