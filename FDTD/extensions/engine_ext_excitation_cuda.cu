/*
*	Copyright (C) 2025 alpLab (alplabai)
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*/

#include "engine_ext_excitation_cuda.h"

#ifdef CUDA_SUPPORT

#include <cuda_runtime.h>
#include "operator_ext_excitation.h"
#include "FDTD/excitation.h"
#include "FDTD/engine_cuda.h"
#include "tools/openems_error.h"

#define CUDA_CHECK(call) do { \
	cudaError_t err = (call); \
	if (err != cudaSuccess) \
		throw openEMS_InternalError(std::string("CUDA error: ") + cudaGetErrorString(err)); \
} while(0)

// ============================================================
// CUDA Kernels for sparse excitation
// ============================================================

__global__ void Excitation_ApplyVolt_kernel(
	float* __restrict__ d_volt,
	const float* __restrict__ d_amp,
	const unsigned int* __restrict__ d_delay,
	const unsigned short* __restrict__ d_dir,
	const unsigned int* __restrict__ d_idx_x,
	const unsigned int* __restrict__ d_idx_y,
	const unsigned int* __restrict__ d_idx_z,
	const float* __restrict__ d_signal,
	unsigned int count,
	int numTS, unsigned int signal_length, int period,
	unsigned int gNy, unsigned int gNz, unsigned int g_stride_n)
{
	unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
	if (n >= count) return;

	int exc_pos = numTS - (int)d_delay[n];
	exc_pos *= (exc_pos > 0);
	exc_pos %= period;
	exc_pos *= (exc_pos < (int)signal_length);

	unsigned int ny = d_dir[n];
	unsigned int gi = ny * g_stride_n + d_idx_x[n] * gNy * gNz + d_idx_y[n] * gNz + d_idx_z[n];

	d_volt[gi] += d_amp[n] * d_signal[exc_pos];
}

__global__ void Excitation_ApplyCurr_kernel(
	float* __restrict__ d_curr,
	const float* __restrict__ d_amp,
	const unsigned int* __restrict__ d_delay,
	const unsigned short* __restrict__ d_dir,
	const unsigned int* __restrict__ d_idx_x,
	const unsigned int* __restrict__ d_idx_y,
	const unsigned int* __restrict__ d_idx_z,
	const float* __restrict__ d_signal,
	unsigned int count,
	int numTS, unsigned int signal_length, int period,
	unsigned int gNy, unsigned int gNz, unsigned int g_stride_n)
{
	unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
	if (n >= count) return;

	int exc_pos = numTS - (int)d_delay[n];
	exc_pos *= (exc_pos > 0);
	exc_pos %= period;
	exc_pos *= (exc_pos < (int)signal_length);

	unsigned int ny = d_dir[n];
	unsigned int gi = ny * g_stride_n + d_idx_x[n] * gNy * gNz + d_idx_y[n] * gNz + d_idx_z[n];

	d_curr[gi] += d_amp[n] * d_signal[exc_pos];
}

// ============================================================
// Implementation
// ============================================================

Engine_Ext_Excitation_CUDA::Engine_Ext_Excitation_CUDA(Operator_Ext_Excitation* op_ext)
	: Engine_Ext_Excitation(op_ext)
{
	m_cuda_init = false;
	m_Eng_CUDA = nullptr;
	d_Volt_amp = d_Curr_amp = nullptr;
	d_Volt_delay = d_Curr_delay = nullptr;
	d_Volt_dir = d_Curr_dir = nullptr;
	d_Volt_index_x = d_Volt_index_y = d_Volt_index_z = nullptr;
	d_Curr_index_x = d_Curr_index_y = d_Curr_index_z = nullptr;
	d_exc_volt_signal = d_exc_curr_signal = nullptr;
	m_signal_length = 0;
	m_signal_period = 0;
}

Engine_Ext_Excitation_CUDA::~Engine_Ext_Excitation_CUDA()
{
	FreeCUDA();
}

void Engine_Ext_Excitation_CUDA::FreeCUDA()
{
	if (d_Volt_amp) cudaFree(d_Volt_amp);
	if (d_Volt_delay) cudaFree(d_Volt_delay);
	if (d_Volt_dir) cudaFree(d_Volt_dir);
	if (d_Volt_index_x) cudaFree(d_Volt_index_x);
	if (d_Volt_index_y) cudaFree(d_Volt_index_y);
	if (d_Volt_index_z) cudaFree(d_Volt_index_z);
	if (d_Curr_amp) cudaFree(d_Curr_amp);
	if (d_Curr_delay) cudaFree(d_Curr_delay);
	if (d_Curr_dir) cudaFree(d_Curr_dir);
	if (d_Curr_index_x) cudaFree(d_Curr_index_x);
	if (d_Curr_index_y) cudaFree(d_Curr_index_y);
	if (d_Curr_index_z) cudaFree(d_Curr_index_z);
	if (d_exc_volt_signal) cudaFree(d_exc_volt_signal);
	if (d_exc_curr_signal) cudaFree(d_exc_curr_signal);
	d_Volt_amp = d_Curr_amp = nullptr;
	m_cuda_init = false;
}

void Engine_Ext_Excitation_CUDA::InitCUDA()
{
	m_Eng_CUDA = dynamic_cast<Engine_CUDA*>(m_Eng);
	if (!m_Eng_CUDA)
		throw openEMS_SetupError("Engine_Ext_Excitation_CUDA: Engine is not CUDA");

	m_signal_length = m_Op_Exc->m_Exc->GetLength();
	if (m_Op_Exc->m_Exc->GetSignalPeriod() > 0)
		m_signal_period = (int)(m_Op_Exc->m_Exc->GetSignalPeriod() / m_Op_Exc->m_Exc->GetTimestep());
	else
		m_signal_period = m_signal_length + 1;

	// Upload voltage excitation data
	if (m_Op_Exc->Volt_Count > 0)
	{
		size_t vc = m_Op_Exc->Volt_Count;
		CUDA_CHECK(cudaMalloc(&d_Volt_amp,     vc * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&d_Volt_delay,   vc * sizeof(unsigned int)));
		CUDA_CHECK(cudaMalloc(&d_Volt_dir,     vc * sizeof(unsigned short)));
		CUDA_CHECK(cudaMalloc(&d_Volt_index_x, vc * sizeof(unsigned int)));
		CUDA_CHECK(cudaMalloc(&d_Volt_index_y, vc * sizeof(unsigned int)));
		CUDA_CHECK(cudaMalloc(&d_Volt_index_z, vc * sizeof(unsigned int)));

		CUDA_CHECK(cudaMemcpy(d_Volt_amp,     m_Op_Exc->Volt_amp,      vc * sizeof(float),          cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_Volt_delay,   m_Op_Exc->Volt_delay,    vc * sizeof(unsigned int),   cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_Volt_dir,     m_Op_Exc->Volt_dir,      vc * sizeof(unsigned short), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_Volt_index_x, m_Op_Exc->Volt_index[0], vc * sizeof(unsigned int),   cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_Volt_index_y, m_Op_Exc->Volt_index[1], vc * sizeof(unsigned int),   cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_Volt_index_z, m_Op_Exc->Volt_index[2], vc * sizeof(unsigned int),   cudaMemcpyHostToDevice));
	}

	// Upload current excitation data
	if (m_Op_Exc->Curr_Count > 0)
	{
		size_t cc = m_Op_Exc->Curr_Count;
		CUDA_CHECK(cudaMalloc(&d_Curr_amp,     cc * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&d_Curr_delay,   cc * sizeof(unsigned int)));
		CUDA_CHECK(cudaMalloc(&d_Curr_dir,     cc * sizeof(unsigned short)));
		CUDA_CHECK(cudaMalloc(&d_Curr_index_x, cc * sizeof(unsigned int)));
		CUDA_CHECK(cudaMalloc(&d_Curr_index_y, cc * sizeof(unsigned int)));
		CUDA_CHECK(cudaMalloc(&d_Curr_index_z, cc * sizeof(unsigned int)));

		CUDA_CHECK(cudaMemcpy(d_Curr_amp,     m_Op_Exc->Curr_amp,      cc * sizeof(float),          cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_Curr_delay,   m_Op_Exc->Curr_delay,    cc * sizeof(unsigned int),   cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_Curr_dir,     m_Op_Exc->Curr_dir,      cc * sizeof(unsigned short), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_Curr_index_x, m_Op_Exc->Curr_index[0], cc * sizeof(unsigned int),   cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_Curr_index_y, m_Op_Exc->Curr_index[1], cc * sizeof(unsigned int),   cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_Curr_index_z, m_Op_Exc->Curr_index[2], cc * sizeof(unsigned int),   cudaMemcpyHostToDevice));
	}

	// Upload signal waveforms
	CUDA_CHECK(cudaMalloc(&d_exc_volt_signal, m_signal_length * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_exc_curr_signal, m_signal_length * sizeof(float)));
	CUDA_CHECK(cudaMemcpy(d_exc_volt_signal, m_Op_Exc->m_Exc->GetVoltageSignal(), m_signal_length * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_exc_curr_signal, m_Op_Exc->m_Exc->GetCurrentSignal(), m_signal_length * sizeof(float), cudaMemcpyHostToDevice));

	m_cuda_init = true;
}

void Engine_Ext_Excitation_CUDA::Apply2Voltages()
{
	if (!m_cuda_init) InitCUDA();
	if (m_Op_Exc->Volt_Count == 0) return;

	const Operator* op = m_Eng_CUDA->GetOperator();
	unsigned int gNy = op->GetNumberOfLines(1, true);
	unsigned int gNz = op->GetNumberOfLines(2, true);
	unsigned int gNx = op->GetNumberOfLines(0, true);
	unsigned int g_stride_n = gNx * gNy * gNz;

	int numTS = m_Eng->GetNumberOfTimesteps();
	int period = (m_signal_period > 0) ? m_signal_period : numTS + 1;

	unsigned int blockSize = 256;
	unsigned int gridSize = (m_Op_Exc->Volt_Count + blockSize - 1) / blockSize;

	Excitation_ApplyVolt_kernel<<<gridSize, blockSize>>>(
		m_Eng_CUDA->GetDeviceVolt(),
		d_Volt_amp, d_Volt_delay, d_Volt_dir,
		d_Volt_index_x, d_Volt_index_y, d_Volt_index_z,
		d_exc_volt_signal,
		m_Op_Exc->Volt_Count,
		numTS, m_signal_length, period,
		gNy, gNz, g_stride_n);
}

void Engine_Ext_Excitation_CUDA::Apply2Current()
{
	if (!m_cuda_init) InitCUDA();
	if (m_Op_Exc->Curr_Count == 0) return;

	const Operator* op = m_Eng_CUDA->GetOperator();
	unsigned int gNy = op->GetNumberOfLines(1, true);
	unsigned int gNz = op->GetNumberOfLines(2, true);
	unsigned int gNx = op->GetNumberOfLines(0, true);
	unsigned int g_stride_n = gNx * gNy * gNz;

	int numTS = m_Eng->GetNumberOfTimesteps();
	int period = (m_signal_period > 0) ? m_signal_period : numTS + 1;

	unsigned int blockSize = 256;
	unsigned int gridSize = (m_Op_Exc->Curr_Count + blockSize - 1) / blockSize;

	Excitation_ApplyCurr_kernel<<<gridSize, blockSize>>>(
		m_Eng_CUDA->GetDeviceCurr(),
		d_Curr_amp, d_Curr_delay, d_Curr_dir,
		d_Curr_index_x, d_Curr_index_y, d_Curr_index_z,
		d_exc_curr_signal,
		m_Op_Exc->Curr_Count,
		numTS, m_signal_length, period,
		gNy, gNz, g_stride_n);
}

#endif // CUDA_SUPPORT
