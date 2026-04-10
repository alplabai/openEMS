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

#ifndef ENGINE_CUDA_KERNELS_H
#define ENGINE_CUDA_KERNELS_H

#ifdef CUDA_SUPPORT

void LaunchUpdateVoltages(
	float* d_volt, const float* d_curr,
	const float* d_vv, const float* d_vi,
	unsigned int Nx, unsigned int Ny, unsigned int Nz);

void LaunchUpdateCurrents(
	float* d_curr, const float* d_volt,
	const float* d_ii, const float* d_iv,
	unsigned int Nx, unsigned int Ny, unsigned int Nz);

#endif // CUDA_SUPPORT
#endif // ENGINE_CUDA_KERNELS_H
