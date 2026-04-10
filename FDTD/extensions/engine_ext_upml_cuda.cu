/*
*	Copyright (C) 2025-2026 alpLab (alplabai)
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*/

#include "engine_ext_upml_cuda.h"

#ifdef CUDA_SUPPORT

#include <cuda_runtime.h>
#include "operator_ext_upml.h"
#include "FDTD/engine_cuda.h"
#include "tools/cuda_check.h"

// ============================================================
// CUDA Kernels for PML flux updates
// ============================================================

__global__ void UPML_PreVolt_kernel(
	float* __restrict__ d_volt,
	float* __restrict__ d_volt_flux,
	const float* __restrict__ d_vv,
	const float* __restrict__ d_vvfo,
	unsigned int pNx, unsigned int pNy, unsigned int pNz,
	unsigned int startX, unsigned int startY, unsigned int startZ,
	unsigned int gNy, unsigned int gNz, unsigned int g_stride_n)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int pml_total = pNx * pNy * pNz;
	if (idx >= pml_total) return;

	unsigned int lx = idx / (pNy * pNz);
	unsigned int ly = (idx / pNz) % pNy;
	unsigned int lz = idx % pNz;

	unsigned int gx = lx + startX;
	unsigned int gy = ly + startY;
	unsigned int gz = lz + startZ;

	unsigned int p_stride_n = pml_total;
	unsigned int g_pos = gx * gNy * gNz + gy * gNz + gz;

	for (int n = 0; n < 3; ++n)
	{
		unsigned int pi = n * p_stride_n + idx;
		unsigned int gi = n * g_stride_n + g_pos;

		float f_help = d_vv[pi] * d_volt[gi] - d_vvfo[pi] * d_volt_flux[pi];
		d_volt[gi] = d_volt_flux[pi];
		d_volt_flux[pi] = f_help;
	}
}

__global__ void UPML_PostVolt_kernel(
	float* __restrict__ d_volt,
	float* __restrict__ d_volt_flux,
	const float* __restrict__ d_vvfn,
	unsigned int pNx, unsigned int pNy, unsigned int pNz,
	unsigned int startX, unsigned int startY, unsigned int startZ,
	unsigned int gNy, unsigned int gNz, unsigned int g_stride_n)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int pml_total = pNx * pNy * pNz;
	if (idx >= pml_total) return;

	unsigned int lx = idx / (pNy * pNz);
	unsigned int ly = (idx / pNz) % pNy;
	unsigned int lz = idx % pNz;

	unsigned int gx = lx + startX;
	unsigned int gy = ly + startY;
	unsigned int gz = lz + startZ;

	unsigned int p_stride_n = pml_total;
	unsigned int g_pos = gx * gNy * gNz + gy * gNz + gz;

	for (int n = 0; n < 3; ++n)
	{
		unsigned int pi = n * p_stride_n + idx;
		unsigned int gi = n * g_stride_n + g_pos;

		float old_flux = d_volt_flux[pi];
		float old_volt = d_volt[gi];
		d_volt_flux[pi] = old_volt;
		d_volt[gi] = old_flux + d_vvfn[pi] * old_volt;
	}
}

__global__ void UPML_PreCurr_kernel(
	float* __restrict__ d_curr,
	float* __restrict__ d_curr_flux,
	const float* __restrict__ d_ii,
	const float* __restrict__ d_iifo,
	unsigned int pNx, unsigned int pNy, unsigned int pNz,
	unsigned int startX, unsigned int startY, unsigned int startZ,
	unsigned int gNy, unsigned int gNz, unsigned int g_stride_n)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int pml_total = pNx * pNy * pNz;
	if (idx >= pml_total) return;

	unsigned int lx = idx / (pNy * pNz);
	unsigned int ly = (idx / pNz) % pNy;
	unsigned int lz = idx % pNz;

	unsigned int gx = lx + startX;
	unsigned int gy = ly + startY;
	unsigned int gz = lz + startZ;

	unsigned int p_stride_n = pml_total;
	unsigned int g_pos = gx * gNy * gNz + gy * gNz + gz;

	for (int n = 0; n < 3; ++n)
	{
		unsigned int pi = n * p_stride_n + idx;
		unsigned int gi = n * g_stride_n + g_pos;

		float f_help = d_ii[pi] * d_curr[gi] - d_iifo[pi] * d_curr_flux[pi];
		d_curr[gi] = d_curr_flux[pi];
		d_curr_flux[pi] = f_help;
	}
}

__global__ void UPML_PostCurr_kernel(
	float* __restrict__ d_curr,
	float* __restrict__ d_curr_flux,
	const float* __restrict__ d_iifn,
	unsigned int pNx, unsigned int pNy, unsigned int pNz,
	unsigned int startX, unsigned int startY, unsigned int startZ,
	unsigned int gNy, unsigned int gNz, unsigned int g_stride_n)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int pml_total = pNx * pNy * pNz;
	if (idx >= pml_total) return;

	unsigned int lx = idx / (pNy * pNz);
	unsigned int ly = (idx / pNz) % pNy;
	unsigned int lz = idx % pNz;

	unsigned int gx = lx + startX;
	unsigned int gy = ly + startY;
	unsigned int gz = lz + startZ;

	unsigned int p_stride_n = pml_total;
	unsigned int g_pos = gx * gNy * gNz + gy * gNz + gz;

	for (int n = 0; n < 3; ++n)
	{
		unsigned int pi = n * p_stride_n + idx;
		unsigned int gi = n * g_stride_n + g_pos;

		float old_flux = d_curr_flux[pi];
		float old_curr = d_curr[gi];
		d_curr_flux[pi] = old_curr;
		d_curr[gi] = old_flux + d_iifn[pi] * old_curr;
	}
}

// ============================================================
// Engine_Ext_UPML_CUDA implementation
// ============================================================

Engine_Ext_UPML_CUDA::Engine_Ext_UPML_CUDA(Operator_Ext_UPML* op_ext)
	: Engine_Ext_UPML(op_ext)
{
	d_vv = d_vvfo = d_vvfn = nullptr;
	d_ii = d_iifo = d_iifn = nullptr;
	d_volt_flux = d_curr_flux = nullptr;
	m_Eng_CUDA = nullptr;
	m_pml_totalSize = 0;
}

Engine_Ext_UPML_CUDA::~Engine_Ext_UPML_CUDA()
{
	FreeCUDA();
}

void Engine_Ext_UPML_CUDA::FreeCUDA()
{
	if (d_vv)        { cudaFree(d_vv);        d_vv = nullptr; }
	if (d_vvfo)      { cudaFree(d_vvfo);      d_vvfo = nullptr; }
	if (d_vvfn)      { cudaFree(d_vvfn);      d_vvfn = nullptr; }
	if (d_ii)        { cudaFree(d_ii);         d_ii = nullptr; }
	if (d_iifo)      { cudaFree(d_iifo);       d_iifo = nullptr; }
	if (d_iifn)      { cudaFree(d_iifn);       d_iifn = nullptr; }
	if (d_volt_flux) { cudaFree(d_volt_flux);  d_volt_flux = nullptr; }
	if (d_curr_flux) { cudaFree(d_curr_flux);  d_curr_flux = nullptr; }
}

void Engine_Ext_UPML_CUDA::InitCUDA()
{
	m_Eng_CUDA = dynamic_cast<Engine_CUDA*>(m_Eng);
	if (!m_Eng_CUDA)
		throw openEMS_SetupError("Engine_Ext_UPML_CUDA: Engine is not CUDA");

	m_pml_Nx = m_Op_UPML->m_numLines[0];
	m_pml_Ny = m_Op_UPML->m_numLines[1];
	m_pml_Nz = m_Op_UPML->m_numLines[2];
	m_startX = m_Op_UPML->m_StartPos[0];
	m_startY = m_Op_UPML->m_StartPos[1];
	m_startZ = m_Op_UPML->m_StartPos[2];

	const Operator* op = m_Eng_CUDA->GetOperator();
	m_global_Nx = op->GetNumberOfLines(0, true);
	m_global_Ny = op->GetNumberOfLines(1, true);
	m_global_Nz = op->GetNumberOfLines(2, true);

	m_pml_totalSize = 3 * (size_t)m_pml_Nx * m_pml_Ny * m_pml_Nz;
	size_t bytes = m_pml_totalSize * sizeof(float);

	try
	{
		// Allocate device arrays for PML coefficients
		CUDA_CHECK(cudaMalloc(&d_vv,   bytes));
		CUDA_CHECK(cudaMalloc(&d_vvfo, bytes));
		CUDA_CHECK(cudaMalloc(&d_vvfn, bytes));
		CUDA_CHECK(cudaMalloc(&d_ii,   bytes));
		CUDA_CHECK(cudaMalloc(&d_iifo, bytes));
		CUDA_CHECK(cudaMalloc(&d_iifn, bytes));

		// Allocate device arrays for flux state
		CUDA_CHECK(cudaMalloc(&d_volt_flux, bytes));
		CUDA_CHECK(cudaMalloc(&d_curr_flux, bytes));

		// Upload PML coefficients (read-only on GPU)
		CUDA_CHECK(cudaMemcpy(d_vv,   m_Op_UPML->vv.data(),   bytes, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_vvfo, m_Op_UPML->vvfo.data(), bytes, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_vvfn, m_Op_UPML->vvfn.data(), bytes, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_ii,   m_Op_UPML->ii.data(),   bytes, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_iifo, m_Op_UPML->iifo.data(), bytes, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_iifn, m_Op_UPML->iifn.data(), bytes, cudaMemcpyHostToDevice));

		// Initialize flux to zero
		CUDA_CHECK(cudaMemset(d_volt_flux, 0, bytes));
		CUDA_CHECK(cudaMemset(d_curr_flux, 0, bytes));
	}
	catch (...)
	{
		FreeCUDA();
		throw;
	}
}

void Engine_Ext_UPML_CUDA::DoPreVoltageUpdates()
{
	if (!m_Eng_CUDA)
		InitCUDA();

	unsigned int pml_cells = m_pml_Nx * m_pml_Ny * m_pml_Nz;
	unsigned int blockSize = 256;
	unsigned int gridSize = (pml_cells + blockSize - 1) / blockSize;
	unsigned int g_stride_n = m_global_Nx * m_global_Ny * m_global_Nz;

	UPML_PreVolt_kernel<<<gridSize, blockSize>>>(
		m_Eng_CUDA->GetDeviceVolt(), d_volt_flux,
		d_vv, d_vvfo,
		m_pml_Nx, m_pml_Ny, m_pml_Nz,
		m_startX, m_startY, m_startZ,
		m_global_Ny, m_global_Nz, g_stride_n);
	CUDA_CHECK(cudaGetLastError());
}

void Engine_Ext_UPML_CUDA::DoPostVoltageUpdates()
{
	if (!m_Eng_CUDA) InitCUDA();

	unsigned int pml_cells = m_pml_Nx * m_pml_Ny * m_pml_Nz;
	unsigned int blockSize = 256;
	unsigned int gridSize = (pml_cells + blockSize - 1) / blockSize;
	unsigned int g_stride_n = m_global_Nx * m_global_Ny * m_global_Nz;

	UPML_PostVolt_kernel<<<gridSize, blockSize>>>(
		m_Eng_CUDA->GetDeviceVolt(), d_volt_flux,
		d_vvfn,
		m_pml_Nx, m_pml_Ny, m_pml_Nz,
		m_startX, m_startY, m_startZ,
		m_global_Ny, m_global_Nz, g_stride_n);
	CUDA_CHECK(cudaGetLastError());
}

void Engine_Ext_UPML_CUDA::DoPreCurrentUpdates()
{
	if (!m_Eng_CUDA) InitCUDA();

	unsigned int pml_cells = m_pml_Nx * m_pml_Ny * m_pml_Nz;
	unsigned int blockSize = 256;
	unsigned int gridSize = (pml_cells + blockSize - 1) / blockSize;
	unsigned int g_stride_n = m_global_Nx * m_global_Ny * m_global_Nz;

	UPML_PreCurr_kernel<<<gridSize, blockSize>>>(
		m_Eng_CUDA->GetDeviceCurr(), d_curr_flux,
		d_ii, d_iifo,
		m_pml_Nx, m_pml_Ny, m_pml_Nz,
		m_startX, m_startY, m_startZ,
		m_global_Ny, m_global_Nz, g_stride_n);
	CUDA_CHECK(cudaGetLastError());
}

void Engine_Ext_UPML_CUDA::DoPostCurrentUpdates()
{
	if (!m_Eng_CUDA) InitCUDA();

	unsigned int pml_cells = m_pml_Nx * m_pml_Ny * m_pml_Nz;
	unsigned int blockSize = 256;
	unsigned int gridSize = (pml_cells + blockSize - 1) / blockSize;
	unsigned int g_stride_n = m_global_Nx * m_global_Ny * m_global_Nz;

	UPML_PostCurr_kernel<<<gridSize, blockSize>>>(
		m_Eng_CUDA->GetDeviceCurr(), d_curr_flux,
		d_iifn,
		m_pml_Nx, m_pml_Ny, m_pml_Nz,
		m_startX, m_startY, m_startZ,
		m_global_Ny, m_global_Nz, g_stride_n);
	CUDA_CHECK(cudaGetLastError());
}

#endif // CUDA_SUPPORT
