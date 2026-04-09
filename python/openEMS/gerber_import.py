# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 alpLab (alplabai)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

"""
Gerber RS-274X and Excellon drill file import for openEMS PCB simulation.

Parses Gerber copper layers and drill files, then generates CSXCAD geometry
from a PCB stackup definition (JSON).

Usage::

    from openEMS.gerber_import import PCBStackup

    stackup = PCBStackup.from_json('stackup.json')
    stackup.build_csxcad(CSX, mesh)
"""

import json
import math
import re
import numpy as np


class GerberParser:
    """Parse a Gerber RS-274X file into polygon geometry.

    Supports the RS-274X subset covering ~95% of PCB Gerber files:
    - Aperture definitions (%AD): Circle, Rectangle, Obround, Polygon
    - D01 (draw), D02 (move), D03 (flash)
    - G01 (linear), G02/G03 (circular arc) interpolation
    - G36/G37 (region fill)
    - G74/G75 (single/multi-quadrant arc mode)
    - %FS (format spec), %MO (units), %LP (polarity)
    """

    def __init__(self):
        self.polygons = []  # list of (vertices_Nx2, is_dark)
        self._apertures = {}
        self._x = 0.0
        self._y = 0.0
        self._fmt_int_x = 2
        self._fmt_dec_x = 4
        self._fmt_int_y = 2
        self._fmt_dec_y = 4
        self._unit_scale = 1e-3  # mm to meters
        self._current_aperture = None
        self._interp_mode = 1  # 1=linear, 2=CW arc, 3=CCW arc
        self._region_mode = False
        self._region_points = []
        self._polarity_dark = True
        self._multi_quadrant = True
        self._arc_segments = 32

    def parse(self, filename):
        """Parse a Gerber file and return polygons.

        Returns
        -------
        list of (ndarray, bool)
            Each element is (vertices as Nx2 array in meters, is_dark).
        """
        self.polygons = []
        self._apertures = {}
        self._x = 0.0
        self._y = 0.0

        with open(filename, 'r') as f:
            content = f.read()

        # Process extended commands (%...%)
        for match in re.finditer(r'%([^%]*)%', content):
            self._process_extended(match.group(1))

        # Remove extended commands and process data blocks
        data = re.sub(r'%[^%]*%', '', content)
        # Split into blocks by * delimiter
        blocks = [b.strip() for b in data.split('*') if b.strip()]

        for block in blocks:
            self._process_block(block)

        return self.polygons

    def _process_extended(self, cmd):
        """Process extended Gerber command."""
        if cmd.startswith('FS'):
            # Format specification: %FSLAX24Y24*%
            m = re.match(r'FS([LT])([AI])X(\d)(\d)Y(\d)(\d)', cmd)
            if m:
                self._fmt_int_x = int(m.group(3))
                self._fmt_dec_x = int(m.group(4))
                self._fmt_int_y = int(m.group(5))
                self._fmt_dec_y = int(m.group(6))

        elif cmd.startswith('MO'):
            # Unit mode
            if 'MM' in cmd:
                self._unit_scale = 1e-3  # mm to meters
            elif 'IN' in cmd:
                self._unit_scale = 25.4e-3  # inches to meters

        elif cmd.startswith('AD'):
            # Aperture definition: %ADD10C,0.254*%
            self._parse_aperture_def(cmd)

        elif cmd.startswith('LP'):
            # Layer polarity
            self._polarity_dark = cmd[2] == 'D'

    def _parse_aperture_def(self, cmd):
        """Parse aperture definition."""
        m = re.match(r'ADD(\d+)([CROP]),?([\d.X]*)', cmd)
        if not m:
            return
        d_code = int(m.group(1))
        shape = m.group(2)
        params = [float(p) for p in m.group(3).split('X') if p] if m.group(3) else []

        if shape == 'C':  # Circle
            r = params[0] / 2.0 * self._unit_scale if params else 0
            pts = self._circle_polygon(r, self._arc_segments)
            self._apertures[d_code] = pts
        elif shape == 'R':  # Rectangle
            w = params[0] * self._unit_scale if len(params) > 0 else 0
            h = params[1] * self._unit_scale if len(params) > 1 else w
            self._apertures[d_code] = np.array([
                [-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]
            ])
        elif shape == 'O':  # Obround
            w = params[0] * self._unit_scale if len(params) > 0 else 0
            h = params[1] * self._unit_scale if len(params) > 1 else w
            self._apertures[d_code] = self._obround_polygon(w, h)
        elif shape == 'P':  # Regular polygon
            od = params[0] * self._unit_scale if len(params) > 0 else 0
            n_vert = int(params[1]) if len(params) > 1 else 4
            rot = params[2] if len(params) > 2 else 0
            self._apertures[d_code] = self._regular_polygon(od/2, n_vert, rot)

    def _process_block(self, block):
        """Process a single data block."""
        # G-code detection
        g_match = re.match(r'G(\d+)', block)
        if g_match:
            g = int(g_match.group(1))
            if g == 1:
                self._interp_mode = 1
            elif g == 2:
                self._interp_mode = 2
            elif g == 3:
                self._interp_mode = 3
            elif g == 36:
                self._region_mode = True
                self._region_points = []
                return
            elif g == 37:
                if self._region_points:
                    pts = np.array(self._region_points)
                    self.polygons.append((pts, self._polarity_dark))
                self._region_mode = False
                self._region_points = []
                return
            elif g == 74:
                self._multi_quadrant = False
            elif g == 75:
                self._multi_quadrant = True
            # Remove G code and continue processing
            block = re.sub(r'G\d+', '', block).strip()
            if not block:
                return

        # D-code at end
        d_match = re.search(r'D(\d+)$', block)
        d_code = int(d_match.group(1)) if d_match else None

        # Parse coordinates
        x_new, y_new = self._x, self._y
        i_off, j_off = 0.0, 0.0

        x_match = re.search(r'X([+-]?\d+)', block)
        y_match = re.search(r'Y([+-]?\d+)', block)
        i_match = re.search(r'I([+-]?\d+)', block)
        j_match = re.search(r'J([+-]?\d+)', block)

        if x_match:
            x_new = self._parse_coord(x_match.group(1), self._fmt_int_x, self._fmt_dec_x)
        if y_match:
            y_new = self._parse_coord(y_match.group(1), self._fmt_int_y, self._fmt_dec_y)
        if i_match:
            i_off = self._parse_coord(i_match.group(1), self._fmt_int_x, self._fmt_dec_x)
        if j_match:
            j_off = self._parse_coord(j_match.group(1), self._fmt_int_y, self._fmt_dec_y)

        if d_code is not None and d_code >= 10:
            # Aperture select
            self._current_aperture = d_code

        elif d_code == 1:
            # Draw (interpolate)
            if self._region_mode:
                if not self._region_points:
                    self._region_points.append([self._x, self._y])
                if self._interp_mode == 1:
                    self._region_points.append([x_new, y_new])
                else:
                    arc_pts = self._arc_to_points(self._x, self._y, x_new, y_new,
                                                  i_off, j_off, self._interp_mode == 2)
                    self._region_points.extend(arc_pts)
            else:
                # Trace: sweep aperture from current to new position
                if self._current_aperture in self._apertures:
                    trace_poly = self._trace_to_polygon(
                        self._x, self._y, x_new, y_new,
                        self._apertures[self._current_aperture])
                    if trace_poly is not None:
                        self.polygons.append((trace_poly, self._polarity_dark))
            self._x, self._y = x_new, y_new

        elif d_code == 2:
            # Move
            self._x, self._y = x_new, y_new

        elif d_code == 3:
            # Flash
            if self._current_aperture in self._apertures:
                ap = self._apertures[self._current_aperture]
                flash = ap + np.array([x_new, y_new])
                self.polygons.append((flash, self._polarity_dark))
            self._x, self._y = x_new, y_new

    def _parse_coord(self, s, n_int, n_dec):
        """Parse coordinate string to float in file units, then to meters."""
        sign = 1
        if s.startswith('-'):
            sign = -1
            s = s[1:]
        elif s.startswith('+'):
            s = s[1:]
        # Pad to expected length
        total = n_int + n_dec
        s = s.zfill(total)
        integer_part = s[:len(s)-n_dec]
        decimal_part = s[len(s)-n_dec:]
        val = sign * (int(integer_part) + int(decimal_part) / (10 ** n_dec))
        return val * self._unit_scale

    def _circle_polygon(self, radius, segments=32):
        """Generate circle polygon vertices."""
        angles = np.linspace(0, 2*np.pi, segments, endpoint=False)
        return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

    def _obround_polygon(self, w, h, segments=16):
        """Generate obround (stadium) polygon."""
        if w > h:
            r = h / 2
            straight = w / 2 - r
            angles_right = np.linspace(-np.pi/2, np.pi/2, segments)
            angles_left = np.linspace(np.pi/2, 3*np.pi/2, segments)
            pts = []
            for a in angles_right:
                pts.append([straight + r*np.cos(a), r*np.sin(a)])
            for a in angles_left:
                pts.append([-straight + r*np.cos(a), r*np.sin(a)])
        else:
            r = w / 2
            straight = h / 2 - r
            angles_top = np.linspace(0, np.pi, segments)
            angles_bot = np.linspace(np.pi, 2*np.pi, segments)
            pts = []
            for a in angles_top:
                pts.append([r*np.cos(a), straight + r*np.sin(a)])
            for a in angles_bot:
                pts.append([r*np.cos(a), -straight + r*np.sin(a)])
        return np.array(pts)

    def _regular_polygon(self, radius, n_sides, rotation_deg=0):
        """Generate regular polygon vertices."""
        rot = np.radians(rotation_deg)
        angles = np.linspace(0, 2*np.pi, n_sides, endpoint=False) + rot
        return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

    def _arc_to_points(self, x0, y0, x1, y1, i_off, j_off, clockwise, segments=None):
        """Convert arc interpolation to polyline points."""
        if segments is None:
            segments = self._arc_segments
        cx = x0 + i_off
        cy = y0 + j_off
        r = math.sqrt((x0 - cx)**2 + (y0 - cy)**2)
        if r < 1e-12:
            return [[x1, y1]]

        a_start = math.atan2(y0 - cy, x0 - cx)
        a_end = math.atan2(y1 - cy, x1 - cx)

        if clockwise:
            if a_end >= a_start:
                a_end -= 2 * math.pi
        else:
            if a_end <= a_start:
                a_end += 2 * math.pi

        n = max(2, int(abs(a_end - a_start) / (2 * math.pi) * segments))
        angles = np.linspace(a_start, a_end, n + 1)[1:]
        pts = [[cx + r * math.cos(a), cy + r * math.sin(a)] for a in angles]
        return pts

    def _trace_to_polygon(self, x0, y0, x1, y1, aperture):
        """Convert a drawn trace to a polygon by sweeping aperture."""
        dx = x1 - x0
        dy = y1 - y0
        length = math.sqrt(dx*dx + dy*dy)
        if length < 1e-12:
            return aperture + np.array([x0, y0])

        # For circular apertures, create a rectangle with semicircular endcaps
        # For simplicity, use offset polygon approach
        nx = -dy / length
        ny = dx / length

        # Approximate aperture radius as max distance from center
        r = np.max(np.sqrt(aperture[:, 0]**2 + aperture[:, 1]**2))

        # Simple rectangle trace
        pts = np.array([
            [x0 + nx*r, y0 + ny*r],
            [x1 + nx*r, y1 + ny*r],
            [x1 - nx*r, y1 - ny*r],
            [x0 - nx*r, y0 - ny*r],
        ])
        return pts


class ExcellonParser:
    """Parse Excellon drill files for via positions.

    Supports:
    - Tool definitions (T01C0.3)
    - Drill hits (X/Y coordinates)
    - Metric/Imperial units
    """

    def __init__(self):
        self.drills = []  # list of (x, y, diameter) in meters

    def parse(self, filename):
        """Parse drill file and return drill hit list.

        Returns
        -------
        list of (float, float, float)
            Each element is (x_meters, y_meters, diameter_meters).
        """
        self.drills = []
        tools = {}
        unit_scale = 1e-3  # default mm
        current_tool = None
        header = True
        fmt_int = 2
        fmt_dec = 4

        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line == '%':
                    header = not header
                    continue

                if line == 'M30' or line == 'M00':
                    break

                if 'METRIC' in line:
                    unit_scale = 1e-3
                elif 'INCH' in line:
                    unit_scale = 25.4e-3

                # Format hint
                fmt_match = re.match(r'FMAT,?(\d)?', line)

                # Tool definition
                t_match = re.match(r'T(\d+)C([\d.]+)', line)
                if t_match:
                    t_num = int(t_match.group(1))
                    diameter = float(t_match.group(2)) * unit_scale
                    tools[t_num] = diameter
                    continue

                # Tool select
                t_sel = re.match(r'T(\d+)$', line)
                if t_sel:
                    current_tool = int(t_sel.group(1))
                    continue

                # Drill hit
                hit = re.match(r'X([+-]?[\d.]+)Y([+-]?[\d.]+)', line)
                if hit and current_tool is not None and current_tool in tools:
                    x = float(hit.group(1)) * unit_scale
                    y = float(hit.group(2)) * unit_scale
                    self.drills.append((x, y, tools[current_tool]))

        return self.drills


class StackupLayer:
    """Definition of one PCB stackup layer."""

    def __init__(self, name, layer_type, thickness,
                 epsilon_r=1.0, loss_tangent=0.0,
                 conductivity=56e6, gerber_file=None,
                 roughness_sr=0, roughness_rf=1.0):
        self.name = name
        self.layer_type = layer_type  # 'signal', 'dielectric', 'plane'
        self.thickness = thickness    # meters
        self.epsilon_r = epsilon_r
        self.loss_tangent = loss_tangent
        self.conductivity = conductivity  # S/m (copper default: 56e6)
        self.gerber_file = gerber_file
        self.roughness_sr = roughness_sr  # Huray RMS roughness height (meters)
        self.roughness_rf = roughness_rf  # Huray roughness factor


class PCBStackup:
    """Complete PCB stackup definition.

    Can be loaded from a JSON file or constructed programmatically.
    """

    def __init__(self, layers=None, unit=1e-3, drill_file=None,
                 via_plating_thickness=25e-6):
        self.layers = layers or []
        self.unit = unit  # drawing unit (default mm)
        self.drill_file = drill_file
        self.via_plating_thickness = via_plating_thickness

    @classmethod
    def from_json(cls, json_path):
        """Load stackup from JSON file.

        Expected format::

            {
                "unit": "mm",
                "layers": [
                    {"name": "Top", "type": "signal", "thickness": 35e-6,
                     "conductivity": 56e6, "gerber": "top.gbr",
                     "roughness_sr": 0.5e-6, "roughness_rf": 2.0},
                    {"name": "PrePreg1", "type": "dielectric", "thickness": 0.2e-3,
                     "epsilon_r": 4.3, "loss_tangent": 0.02},
                    ...
                ],
                "drill_file": "drill.drl",
                "via_plating_thickness": 25e-6
            }
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        unit_map = {'mm': 1e-3, 'um': 1e-6, 'mil': 25.4e-6, 'in': 25.4e-3, 'm': 1.0}
        unit = unit_map.get(data.get('unit', 'mm'), 1e-3)

        layers = []
        for ldata in data.get('layers', []):
            layers.append(StackupLayer(
                name=ldata['name'],
                layer_type=ldata['type'],
                thickness=ldata['thickness'],
                epsilon_r=ldata.get('epsilon_r', 1.0),
                loss_tangent=ldata.get('loss_tangent', 0.0),
                conductivity=ldata.get('conductivity', 56e6),
                gerber_file=ldata.get('gerber', None),
                roughness_sr=ldata.get('roughness_sr', 0),
                roughness_rf=ldata.get('roughness_rf', 1.0),
            ))

        return cls(
            layers=layers,
            unit=unit,
            drill_file=data.get('drill_file', None),
            via_plating_thickness=data.get('via_plating_thickness', 25e-6),
        )

    def compute_z_positions(self):
        """Compute z-position for each layer (bottom-up stacking).

        Returns
        -------
        list of (z_bottom, z_top) in meters
        """
        z = 0.0
        positions = []
        for layer in self.layers:
            positions.append((z, z + layer.thickness))
            z += layer.thickness
        return positions

    def build_csxcad(self, CSX, mesh=None, priority_base=100, gerber_dir='.'):
        """Generate CSXCAD geometry from the stackup.

        Parameters
        ----------
        CSX : ContinuousStructure
            The CSXCAD structure to add geometry to.
        mesh : CSRectGrid, optional
            If provided, adds z-mesh lines at layer boundaries.
        priority_base : int
            Base priority (dielectric=base, copper=base+100, via=base+200).
        gerber_dir : str
            Directory containing Gerber files referenced in the stackup.

        Returns
        -------
        dict
            Mapping of layer names to CSXCAD properties.
        """
        import os

        z_positions = self.compute_z_positions()
        properties = {}
        board_bounds = None  # will be set from first Gerber parse

        # Determine board bounds from Gerber files
        x_min = float('inf')
        x_max = float('-inf')
        y_min = float('inf')
        y_max = float('-inf')

        parsed_gerbers = {}
        parser = GerberParser()

        for layer, (z_bot, z_top) in zip(self.layers, z_positions):
            if layer.gerber_file:
                gerber_path = os.path.join(gerber_dir, layer.gerber_file)
                if os.path.exists(gerber_path):
                    polys = parser.parse(gerber_path)
                    parsed_gerbers[layer.name] = polys
                    for pts, _ in polys:
                        if len(pts) > 0:
                            x_min = min(x_min, pts[:, 0].min())
                            x_max = max(x_max, pts[:, 0].max())
                            y_min = min(y_min, pts[:, 1].min())
                            y_max = max(y_max, pts[:, 1].max())

        # Add margin
        if x_min < float('inf'):
            margin = max(x_max - x_min, y_max - y_min) * 0.1
            board_bounds = [x_min - margin, x_max + margin,
                            y_min - margin, y_max + margin]

        z_total = z_positions[-1][1] if z_positions else 0

        # Build layers
        for layer, (z_bot, z_top) in zip(self.layers, z_positions):
            if layer.layer_type == 'dielectric':
                # Add dielectric substrate
                mat = CSX.AddMaterial(layer.name, epsilon=layer.epsilon_r)
                if layer.loss_tangent > 0:
                    mat.SetMaterialProperty(name='kappa',
                        value=layer.loss_tangent * layer.epsilon_r * 2 * np.pi * 1e9 * 8.854e-12)
                if board_bounds:
                    mat.AddBox([board_bounds[0], board_bounds[2], z_bot],
                               [board_bounds[1], board_bounds[3], z_top],
                               priority=priority_base)
                properties[layer.name] = mat

            elif layer.layer_type in ('signal', 'plane'):
                # Add copper layer from Gerber polygons
                copper = CSX.AddConductingSheet(layer.name,
                    conductivity=layer.conductivity,
                    thickness=layer.thickness)

                if layer.name in parsed_gerbers:
                    for pts, is_dark in parsed_gerbers[layer.name]:
                        if is_dark and len(pts) >= 3:
                            # Convert to CSXCAD units
                            pts_scaled = pts / self.unit
                            z_mid = (z_bot + z_top) / 2.0 / self.unit
                            copper.AddPolygon(
                                pts_scaled.tolist(), 'z', z_mid,
                                priority=priority_base + 100)

                properties[layer.name] = copper

            # Add z-mesh lines
            if mesh is not None:
                mesh.AddLine('z', z_bot / self.unit)
                mesh.AddLine('z', z_top / self.unit)

        # Add vias from drill file
        if self.drill_file:
            drill_path = os.path.join(gerber_dir, self.drill_file)
            if os.path.exists(drill_path):
                excellon = ExcellonParser()
                drills = excellon.parse(drill_path)

                via_metal = CSX.AddMetal('via')
                for x, y, diameter in drills:
                    x_s = x / self.unit
                    y_s = y / self.unit
                    r_s = (diameter / 2.0) / self.unit
                    z_bot_s = z_positions[0][0] / self.unit
                    z_top_s = z_positions[-1][1] / self.unit
                    via_metal.AddCylinder(
                        [x_s, y_s, z_bot_s],
                        [x_s, y_s, z_top_s],
                        r_s,
                        priority=priority_base + 200)

                properties['via'] = via_metal

        return properties
