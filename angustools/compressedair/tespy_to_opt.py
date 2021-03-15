# -*- coding: utf-8

"""Calculate heating .


This file is part of project angustools (github.com/fwitte/angustools). It's
copyrighted by the contributors recorded in the version control history of the
file, available from its original location
angustools/heat/minimize_residual_load.py

SPDX-License-Identifier: MIT
"""

import pandas as pd
import numpy as np


def generate_lut_power_pressure(
        nwk, design_path, mass_obj, power_obj, pressure_obj,
        power_range, pressure_range, power_design, pressure_design):
    r"""
    Generate a lookup table over power and pressure inputs.

    The function loops over the given power and pressure ranges, sets the
    power value on the power_obj, the pressure value on the pressure_obj and
    calculates the resulting mass flow on the mass_obj as well as the values
    of all busses of the network. A dictionary containing the data is returned.

    Parameters
    ----------
    nwk : tespy.networks.networks.network
        TESPy network object to generate the lookup table on.

    design_path : str
        Path to the networks design state.

    mass_obj : tespy.connections.connection
        TESPy connection to calculate the mass flow.

    power_obj : tespy.connections.bus
        TESPy bus to set the power values.

    pressure_obj : tespy.connections.connection
        TESPy connection to set the pressure values.

    power_range : ndarray
        numpy array of the power the lookup table should span over.

    pressure_range : ndarray
        numpy array of the pressure the lookup table should span over.

    Returns
    -------
    df : dict
        Dictionary of DataFrames containing the calculated lookup tables.
    """
    df = {}
    busses = {}

    df['mass flow'] = pd.DataFrame(columns=pressure_range)

    for key in nwk.busses.keys():
        df[key] = pd.DataFrame(columns=pressure_range)
        busses[key] = []

    for power in power_range:
        mass_flow = []
        for key in busses.keys():
            busses[key] = []
        power_obj.set_attr(P=power * 1e6)
        for p in pressure_range:
            pressure_obj.set_attr(p=p)
            if p == pressure_range[0]:
                pressure_obj.set_attr(p=pressure_design)
                power_obj.set_attr(P=power_design)
                nwk.solve(
                    'offdesign',
                    design_path=design_path,
                    init_path=design_path)
                for power_step in np.linspace(power * 1e6, power_design, 3, endpoint=False)[::-1]:
                    power_obj.set_attr(P=power_step)
                    nwk.solve(
                        'offdesign',
                        design_path=design_path)

                for p_step in np.linspace(p, pressure_design, 3, endpoint=False)[::-1]:
                    pressure_obj.set_attr(p=p_step)
                    nwk.solve(
                        'offdesign',
                        design_path=design_path)
            else:
                nwk.solve('offdesign', design_path=design_path)

            if nwk.res[-1] > 1e-3 or nwk.lin_dep is True:
                mass_flow += [np.nan]
                for key in nwk.busses.keys():
                    busses[key] += [np.nan]

                print('Error calculating the network for input pair power, '
                      'pressure (' + str(power) + '; ' + str(p) + ').')
            else:
                mass_flow += [mass_obj.m.val_SI]
                for key, value in nwk.busses.items():
                    busses[key] += [value.P.val / 1e6]

        for key in nwk.busses.keys():
            df[key].loc[abs(power)] = busses[key]

        df['mass flow'].loc[abs(power)] = mass_flow

    return df


def linearise_lut(df):
    r"""
    Calculate parameters for linearisation of tabular 3-D surface data.

    The least squares algorithm is applied: x-data (DataFrame.index), y-data
    (DataFrame.columns) and z-data (DataFrame.values).

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        DataFrame containing the 3-D tabular data.

    Returns
    -------
    x : ndarray
        Array with paramaters of the linearised plane.
    """
    x_array = df.index.values[::-1]
    y_array = pd.to_numeric(df.columns.values)[::-1]
    z_matrix = df.values

    grid_num = len(x_array)

    x_mod = np.repeat(x_array, grid_num)
    y_mod = np.tile(y_array, grid_num)
    z_mod = z_matrix.flatten()[::-1]

    A = np.ones((3, 3))
    A[0, 0] = (x_mod * x_mod).sum()
    A[0, 1] = (x_mod * y_mod).sum()
    A[0, 2] = x_mod.sum()
    A[1, 0] = A[0, 1]
    A[1, 1] = (y_mod * y_mod).sum()
    A[1, 2] = y_mod.sum()
    A[2, 0] = A[0, 2]
    A[2, 1] = A[1, 2]
    A[2, 2] = len(x_mod)

    b = np.ones(3)
    b[0] = (x_mod * z_mod).sum()
    b[1] = (y_mod * z_mod).sum()
    b[2] = z_mod.sum()

    x = b.dot(np.linalg.inv(A))

    return x
