# -*- coding: utf-8 -*-
"""
Created on Mon May 27 10:23:38 2023

@author: bksat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cluster_tools as ct
import sklearn.cluster as cluster
import sklearn.metrics as skmet



def FilterData(df):
    """
    This function filters the dataframe for the required countries such as US,
    UK, France, India, China, Germany, Russia and keeps only the required
    fields.
    """

    # required indicators
    indicator = ['SP.URB.TOTL', 'SP.POP.TOTL', 'SH.DYN.MORT', 'SH.DYN.MORT',
                 'ER.H2O.FWTL.K3', 'EN.ATM.GHGT.KT.CE', 'EN.ATM.CO2E.KT',
                 'EN.ATM.CO2E.SF.KT', 'EN.ATM.CO2E.LF.KT', 'EN.ATM.CO2E.GF.KT',
                 'EG.USE.ELEC.KH.PC', 'EG.ELC.RNEW.ZS', 'AG.LND.FRST.K2',
                 'AG.LND.ARBL.ZS', 'AG.LND.AGRI.K2']
    df = df.loc[df['Indicator Code'].isin(indicator)]
    return df


def Preprocess(df):
    """
    This function preprocesses the data by removing the country code and
    fill NaN with zeroes
    """
    df.drop('Country Code', axis=1, inplace=True)
    df.fillna(0, inplace=True)
    return df


def VariableTrend(df, indicator):
    """
    This function filters the trend for particular indicator
    """
    # filter the given indicator
    var_df = df.loc[df['Indicator Code'] == indicator]
    # drops unneccessary columns
    var_df.drop(['Indicator Name', 'Indicator Code'],
                axis=1, inplace=True)
    var_df.reset_index(drop=True, inplace=True)
    # transpose the dataframe and renamed the column names
    var_df = var_df.T
    var_df = var_df.rename(columns=var_df.iloc[0])
    var_df.drop(labels=['Country Name'], axis=0, inplace=True)
    var_df.rename(columns={'United Kingdom': 'UK',
                           'Russian Federation': 'Russia',
                           'United States': 'US'}, inplace=True)
    var_df.reset_index(inplace=True)
    var_df.drop(var_df.tail(1).index, inplace=True)

    # change the data type of the dataframe to float
    columns = list(var_df.columns)
    for col in columns:
        if col != 'Year':
            var_df[col] = var_df[col].astype('float64')

    return var_df


def HeatmapPreprocess(df, country):
    """
    This function processes the given data frame so that resulting data frame
    can be used to generate the heatmap
    """
    hdf = df.loc[df['Country Name'] == country]

    # filters required indicators
    indicator = ['SP.URB.TOTL', 'SP.POP.TOTL', 'SH.DYN.MORT', 'ER.H2O.FWTL.K3',
                 'AG.LND.FRST.K2', 'AG.LND.ARBL.ZS', 'AG.LND.AGRI.K2',
                 'EN.ATM.CO2E.KT']
    hdf = hdf.loc[hdf['Indicator Code'].isin(indicator)]

    # removes the unnecessary columns
    hdf.drop(['Country Name', 'Indicator Code'], axis=1, inplace=True)
    hdf.reset_index(drop=True, inplace=True)

    # transpose the data
    hdf = hdf.T
    hdf.reset_index(inplace=True)
    hdf = hdf.rename(columns=hdf.iloc[0])
    hdf.drop(0, inplace=True)

    # removes the last three rows as there is no data for those years
    hdf.drop(hdf.tail(3).index, inplace=True)

    # rename the columns names
    hdf.rename(columns={'Indicator Name': 'Year',
                        'Population, total': 'Total Population',
                        'Mortality rate, under-5 (per 1,000 live births)':
                        'Mortality rate',
                        'Annual freshwater withdrawals, total (billion cubic meters)': 'Annual freshwater withdrawals',
                        'CO2 emissions (kt)': 'CO2 emissions',
                        'Forest area (sq. km)': 'Forest area',
                        'Arable land (% of land area)': 'Arable land',
                        'Agricultural land (sq. km)': 'Agricultural land'},
               inplace=True)

    hdf['Year'] = hdf['Year'].astype('int')
    # keeps data after 1990 as the data is available from 1990
    hdf = hdf[hdf['Year'] >= 1990]
    hdf.reset_index(drop=True, inplace=True)
    hdf['Year'] = hdf['Year'].astype('object')

    # type casting of the dataframe columns
    columns = list(hdf.columns)
    for col in columns:
        if col != 'Year':
            hdf[col] = hdf[col].astype('float64')

    return hdf


# main code
# Reads data from the world bank climate data
df = pd.read_csv('API_19_DS2_en_csv_v2_4902199.csv', skiprows=4)

# filters the data for the required countries
df = FilterData(df)
df = Preprocess(df)


# heatmap for Germany
hdf_Germany = HeatmapPreprocess(df, 'Germany')
hdf_Germany.describe()

plt.figure(dpi=500)
ct.map_corr(hdf_Germany)
plt.title('Heatmap for Germany', fontweight="bold")
plt.savefig('hmap_Germany.png')
plt.show()

# heatmap for Germany
hdf_China = HeatmapPreprocess(df, 'China')
hdf_China.describe()

plt.figure(dpi=500)
ct.map_corr(hdf_China)
plt.title('Heatmap for China', fontweight="bold")
plt.savefig('hmap_China.png')
plt.show()
