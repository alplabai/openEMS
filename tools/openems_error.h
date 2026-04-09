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

#ifndef OPENEMS_ERROR_H
#define OPENEMS_ERROR_H

#include <stdexcept>
#include <string>

/** Base exception for all openEMS errors (safe for shared-library use). */
class openEMS_Exception : public std::runtime_error
{
public:
	explicit openEMS_Exception(const std::string& msg) : std::runtime_error(msg) {}
};

/** Fatal setup/configuration error — the simulation cannot proceed. */
class openEMS_SetupError : public openEMS_Exception
{
public:
	explicit openEMS_SetupError(const std::string& msg) : openEMS_Exception(msg) {}
};

/** Memory allocation failure. */
class openEMS_AllocationError : public openEMS_Exception
{
public:
	explicit openEMS_AllocationError(const std::string& msg) : openEMS_Exception(msg) {}
};

/** Internal logic error — "should not happen" conditions. */
class openEMS_InternalError : public openEMS_Exception
{
public:
	explicit openEMS_InternalError(const std::string& msg) : openEMS_Exception(msg) {}
};

#endif // OPENEMS_ERROR_H
