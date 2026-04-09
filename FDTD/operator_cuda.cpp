/*
*	Copyright (C) 2025 alpLab (alplabai)
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

#include "operator_cuda.h"

#ifdef CUDA_SUPPORT

#include "engine_cuda.h"

Operator_CUDA::Operator_CUDA() : Operator()
{
}

Operator_CUDA::~Operator_CUDA()
{
}

Operator_CUDA* Operator_CUDA::New()
{
	std::cout << "Create CUDA operator..." << std::endl;
	Operator_CUDA* op = new Operator_CUDA();
	op->Init();
	return op;
}

Engine* Operator_CUDA::CreateEngine()
{
	m_Engine = Engine_CUDA::New(this);
	return m_Engine;
}

#endif // CUDA_SUPPORT
