# -*- coding: utf-8

"""Minimize residual heat load given a technology's minimum relative load
restriction.


This file is part of project angustools (github.com/fwitte/angustools). It's
copyrighted by the contributors recorded in the version control history of the
file, available from its original location
angustools/heat/minimize_residual_load.py

SPDX-License-Identifier: MIT
"""

import pandas as pd
import numpy as np


def maximise_thermal_energy_output(ts, min_val_rel):
    r"""
    Search for nominal heat output.

    The nominal heat output value must satisfy the condition of maximum area
    under demand curve, given a minimum relative load. This means, the largest
    possible amount of thermal energy should be delivered by this plant.

    Calculation of the thermal energy delivered:

    .. math::

        \dot{Q}_\mathrm{min} = \dot{Q}_\mathrm{nom} \cdot f_\mathrm{min,rel}\\

        t_1 = t_\mathrm{max}\left(\dot{Q} > \dot{Q}_\mathrm{nom}\right)\\

        t_2 = t_\mathrm{max}\left(\dot{Q} <= \dot{Q}_\mathrm{nom} \land
        \dot{Q} >= \dot{Q}_\mathrm{min} \right)\\

        A = \dot{Q}_\mathrm{nom} \cdot t_1 + \sum_{t=t_1 + 1}^{t_2} \dot{Q}_t
        \cdot \tau

    Parameters
    ----------
    ts : pandas.core.series.Series
        Series containting the heat load demand.

    min_val_rel : float
        Factor for minimum heat output relative to nominal heat output.

    Returns
    -------
    Q_N : float
        Nominal heat output.

    ts : pandas.core.series.Series
        Series containting the residual heat load demand.
    """
    ts = ts.copy()
    area = 0
    counter = 1
    Q_old = np.nan
    for Q in ts.values:
        if Q == Q_old:
            counter += 1
            continue

        area_full_load = Q * (counter - 1)
        area_part_load = ts[(ts <= Q) & (ts >= Q * min_val_rel)].sum()

        if area < area_full_load + area_part_load:
            area = area_full_load + area_part_load
            Q_nom = Q

        counter += 1
        Q_old = Q

    ts[(ts <= Q_nom) & (ts >= Q_nom * min_val_rel)] = 0
    ts[(ts > Q_nom)] = ts - Q_nom

    return Q_nom, ts


def calculate_nominal_heat_by_tech(technologies, ts):
    """
    Calculate the nominal heat transfer capacity of different technologies.

    Parameters
    ----------
    technologies : pandas.core.frame.DataFrame
        DataFrame containing the information of the technologies.

    ts : pandas.core.series.Series
        Series containting the heat load demand.

    Returns
    -------
    technologies : pandas.core.frame.DataFrame
        DataFrame containing the calculated information of the technologies.

    Example
    -------
    >>> from angustools.heat import heating_system_design as hsd
    >>> import pandas as pd
    >>> ts = pd.read_csv(sys.argv[1]).dropna()
    >>> ts.set_index('Datum', inplace=True)
    >>> ts['Gesamt'] = ts['Gesamt'].astype(float)
    >>> ts.index = pd.to_datetime(ts.index)
    >>> ts.sort_index(inplace=True)
    >>> ts.sort_values(by='Gesamt', ascending=False, inplace=True)
    >>> ts.reset_index(inplace=True)
    >>> ts = ts['Gesamt']
    >>> technologien = pd.DataFrame(columns=['Q_N', 'Q_min_rel'])
    >>> technologien.loc['BHKW'] = [np.nan, 0.55]
    >>> technologien.loc['GuD'] = [np.nan, 0.7]
    >>> technologien.loc['WP'] = [np.nan, 0.3]
    >>> technologien.loc['EHK'] = [np.nan, 0.0]
    >>> technologien = hsd.calculate_nominal_heat_by_tech(technologien, ts)
    >>> technologien.to_csv('technologien.csv')
    """
    for technology in technologies.index:
        ts = ts[ts > 0].round(5)
        ts.sort_values(ascending=False, inplace=True)
        ts = ts.reset_index(drop=True)
        technologies.loc[technology, 'Q_nom'], ts = (
            maximise_thermal_energy_output(
                ts, technologies.loc[technology, 'Q_min_rel']))

    technologies['Q_min'] = technologies['Q_nom'] * technologies['Q_min_rel']
    return technologies
