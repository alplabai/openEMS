/*
*	Copyright (C) 2025-2026 alpLab (alplabai)
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*
*	This program is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef CUDA_CHECK_H
#define CUDA_CHECK_H

#ifdef CUDA_SUPPORT

#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include "openems_error.h"

#define CUDA_CHECK(call) do { \
	cudaError_t err = (call); \
	if (err != cudaSuccess) { \
		std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
		          << cudaGetErrorString(err) << std::endl; \
		throw openEMS_InternalError(std::string("CUDA error: ") + cudaGetErrorString(err)); \
	} \
} while(0)

#endif // CUDA_SUPPORT
#endif // CUDA_CHECK_H
