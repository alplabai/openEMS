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

#ifndef SCOPED_LOCALE_H
#define SCOPED_LOCALE_H

#include <clocale>
#include <cstdlib>
#include <cstring>

/**
 * RAII guard that sets LC_NUMERIC to "C" for the current scope and restores
 * the previous locale on destruction. This prevents global locale corruption
 * when multiple threads or library callers have different locale expectations.
 *
 * Uses "C" locale instead of "en_US.UTF-8" because "C" is always available
 * on all platforms and produces the same decimal-point behavior needed for
 * XML/number parsing (dot as decimal separator).
 */
class ScopedNumericLocale
{
public:
	ScopedNumericLocale()
	{
		const char* prev = std::setlocale(LC_NUMERIC, nullptr);
		if (prev)
			std::strncpy(m_prev, prev, sizeof(m_prev) - 1);
		else
			m_prev[0] = '\0';
		m_prev[sizeof(m_prev) - 1] = '\0';
		std::setlocale(LC_NUMERIC, "C");
	}

	~ScopedNumericLocale()
	{
		if (m_prev[0] != '\0')
			std::setlocale(LC_NUMERIC, m_prev);
	}

	ScopedNumericLocale(const ScopedNumericLocale&) = delete;
	ScopedNumericLocale& operator=(const ScopedNumericLocale&) = delete;

private:
	char m_prev[128];
};

#endif // SCOPED_LOCALE_H
