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

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include "engine_cuda_kernels.h"

/**
 * Memory layout (ArrayNIJK, N-I-J-K ordering):
 *   index = n * Nx*Ny*Nz + x * Ny*Nz + y * Nz + z
 *
 * Z is innermost (stride=1) → consecutive threads access consecutive Z values
 * → perfect coalesced GPU memory access when thread index maps to Z.
 *
 * Thread mapping: 1D grid of 1D blocks.
 *   linear_idx = blockIdx.x * blockDim.x + threadIdx.x
 *   x = linear_idx / (Ny * Nz)
 *   y = (linear_idx / Nz) % Ny
 *   z = linear_idx % Nz
 */

/**
 * E-field (voltage) update kernel.
 *
 * Standard Yee cell update:
 *   E_x = vv_x * E_x + vi_x * (H_z(y) - H_z(y-1) - H_y(z) + H_y(z-1))
 *   E_y = vv_y * E_y + vi_y * (H_x(z) - H_x(z-1) - H_z(x) + H_z(x-1))
 *   E_z = vv_z * E_z + vi_z * (H_y(x) - H_y(x-1) - H_x(y) + H_x(y-1))
 *
 * Boundary handling: at pos=0, the shift is 0 (no subtraction of neighbor),
 * matching the CPU code's `shift[n] = pos[n]` pattern.
 */
__global__ void UpdateVoltages_kernel(
	float* __restrict__ volt,
	const float* __restrict__ curr,
	const float* __restrict__ vv,
	const float* __restrict__ vi,
	unsigned int Nx, unsigned int Ny, unsigned int Nz)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int NyNz = Ny * Nz;
	unsigned int total = Nx * NyNz;

	if (idx >= total)
		return;

	unsigned int x = idx / NyNz;
	unsigned int y = (idx / Nz) % Ny;
	unsigned int z = idx % Nz;

	unsigned int stride_n = total;  // stride for field component dimension

	// Shift flags: at boundary (pos=0), don't subtract neighbor
	unsigned int sy = (y > 0) ? 1 : 0;
	unsigned int sz = (z > 0) ? 1 : 0;
	unsigned int sx = (x > 0) ? 1 : 0;

	unsigned int pos = x * NyNz + y * Nz + z;

	// E_x update: curl_H_x = dHz/dy - dHy/dz
	float curl_x = curr[2 * stride_n + pos]                       // Hz(x,y,z)
	             - curr[2 * stride_n + x * NyNz + (y - sy) * Nz + z]  // Hz(x,y-1,z)
	             - curr[1 * stride_n + pos]                       // Hy(x,y,z)
	             + curr[1 * stride_n + x * NyNz + y * Nz + (z - sz)]; // Hy(x,y,z-1)
	volt[0 * stride_n + pos] = vv[0 * stride_n + pos] * volt[0 * stride_n + pos]
	                         + vi[0 * stride_n + pos] * curl_x;

	// E_y update: curl_H_y = dHx/dz - dHz/dx
	float curl_y = curr[0 * stride_n + pos]                       // Hx(x,y,z)
	             - curr[0 * stride_n + x * NyNz + y * Nz + (z - sz)]  // Hx(x,y,z-1)
	             - curr[2 * stride_n + pos]                       // Hz(x,y,z)
	             + curr[2 * stride_n + (x - sx) * NyNz + y * Nz + z]; // Hz(x-1,y,z)
	volt[1 * stride_n + pos] = vv[1 * stride_n + pos] * volt[1 * stride_n + pos]
	                         + vi[1 * stride_n + pos] * curl_y;

	// E_z update: curl_H_z = dHy/dx - dHx/dy
	float curl_z = curr[1 * stride_n + pos]                       // Hy(x,y,z)
	             - curr[1 * stride_n + (x - sx) * NyNz + y * Nz + z]  // Hy(x-1,y,z)
	             - curr[0 * stride_n + pos]                       // Hx(x,y,z)
	             + curr[0 * stride_n + x * NyNz + (y - sy) * Nz + z]; // Hx(x,y-1,z)
	volt[2 * stride_n + pos] = vv[2 * stride_n + pos] * volt[2 * stride_n + pos]
	                         + vi[2 * stride_n + pos] * curl_z;
}

/**
 * H-field (current) update kernel.
 *
 * Standard Yee cell update:
 *   H_x = ii_x * H_x + iv_x * (E_z(y) - E_z(y+1) - E_y(z) + E_y(z+1))
 *   H_y = ii_y * H_y + iv_y * (E_x(z) - E_x(z+1) - E_z(x) + E_z(x+1))
 *   H_z = ii_z * H_z + iv_z * (E_y(x) - E_y(x+1) - E_x(y) + E_x(y+1))
 *
 * Note: H-field update range is [0, Nx-1) x [0, Ny-1) x [0, Nz-1),
 * i.e. one less than E-field in each dimension.
 */
__global__ void UpdateCurrents_kernel(
	float* __restrict__ curr,
	const float* __restrict__ volt,
	const float* __restrict__ ii,
	const float* __restrict__ iv,
	unsigned int Nx, unsigned int Ny, unsigned int Nz)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int NxH = Nx - 1;  // H-field update: [0, Nx-2]
	unsigned int NyH = Ny - 1;  // [0, Ny-2]
	unsigned int NzH = Nz - 1;  // [0, Nz-2]
	unsigned int totalH = NxH * NyH * NzH;

	if (idx >= totalH)
		return;

	// Map linear index to 3D position within H-field update range
	unsigned int x = idx / (NyH * NzH);
	unsigned int y = (idx / NzH) % NyH;
	unsigned int z = idx % NzH;

	unsigned int NyNz = Ny * Nz;
	unsigned int stride_n = Nx * NyNz;
	unsigned int pos = x * NyNz + y * Nz + z;

	// H_x update
	float curl_x = volt[2 * stride_n + pos]                          // Ez(x,y,z)
	             - volt[2 * stride_n + x * NyNz + (y + 1) * Nz + z]  // Ez(x,y+1,z)
	             - volt[1 * stride_n + pos]                          // Ey(x,y,z)
	             + volt[1 * stride_n + x * NyNz + y * Nz + (z + 1)]; // Ey(x,y,z+1)
	curr[0 * stride_n + pos] = ii[0 * stride_n + pos] * curr[0 * stride_n + pos]
	                         + iv[0 * stride_n + pos] * curl_x;

	// H_y update
	float curl_y = volt[0 * stride_n + pos]                          // Ex(x,y,z)
	             - volt[0 * stride_n + x * NyNz + y * Nz + (z + 1)]  // Ex(x,y,z+1)
	             - volt[2 * stride_n + pos]                          // Ez(x,y,z)
	             + volt[2 * stride_n + (x + 1) * NyNz + y * Nz + z]; // Ez(x+1,y,z)
	curr[1 * stride_n + pos] = ii[1 * stride_n + pos] * curr[1 * stride_n + pos]
	                         + iv[1 * stride_n + pos] * curl_y;

	// H_z update
	float curl_z = volt[1 * stride_n + pos]                          // Ey(x,y,z)
	             - volt[1 * stride_n + (x + 1) * NyNz + y * Nz + z]  // Ey(x+1,y,z)
	             - volt[0 * stride_n + pos]                          // Ex(x,y,z)
	             + volt[0 * stride_n + x * NyNz + (y + 1) * Nz + z]; // Ex(x,y+1,z)
	curr[2 * stride_n + pos] = ii[2 * stride_n + pos] * curr[2 * stride_n + pos]
	                         + iv[2 * stride_n + pos] * curl_z;
}

// Launch wrappers

void LaunchUpdateVoltages(
	float* d_volt, const float* d_curr,
	const float* d_vv, const float* d_vi,
	unsigned int Nx, unsigned int Ny, unsigned int Nz)
{
	size_t total = (size_t)Nx * Ny * Nz;
	if (total == 0) return;
	unsigned int blockSize = 256;
	unsigned int gridSize = (unsigned int)((total + blockSize - 1) / blockSize);
	UpdateVoltages_kernel<<<gridSize, blockSize>>>(d_volt, d_curr, d_vv, d_vi, Nx, Ny, Nz);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		throw std::runtime_error(std::string("UpdateVoltages kernel launch failed: ") + cudaGetErrorString(err));
}

void LaunchUpdateCurrents(
	float* d_curr, const float* d_volt,
	const float* d_ii, const float* d_iv,
	unsigned int Nx, unsigned int Ny, unsigned int Nz)
{
	if (Nx < 2 || Ny < 2 || Nz < 2) return;
	size_t totalH = (size_t)(Nx - 1) * (Ny - 1) * (Nz - 1);
	unsigned int blockSize = 256;
	unsigned int gridSize = (unsigned int)((totalH + blockSize - 1) / blockSize);
	UpdateCurrents_kernel<<<gridSize, blockSize>>>(d_curr, d_volt, d_ii, d_iv, Nx, Ny, Nz);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		throw std::runtime_error(std::string("UpdateCurrents kernel launch failed: ") + cudaGetErrorString(err));
}
