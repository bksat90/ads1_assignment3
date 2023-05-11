# -*- coding: utf-8 -*-
"""
Created on Mon May 01 10:23:38 2023

@author: bksat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cluster_tools as ct
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt


def FilterData(df):
    """
    This function keeps only the required
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


def HeatmapPreprocess(df, year):
    """
    This function processes the given data frame so that resulting data frame
    can be used to generate the heatmap
    """
    hdf = df.loc[:, df.columns.isin(['Country Name', 'Indicator Name',
                                     'Indicator Code', str(year)])]
    indicators = ['SP.URB.TOTL', 'SP.POP.TOTL', 'SH.DYN.MORT',
                  'ER.H2O.FWTL.K3', 'AG.LND.FRST.K2', 'AG.LND.ARBL.ZS',
                  'AG.LND.AGRI.K2', 'EN.ATM.CO2E.KT']
    hdf = hdf.loc[hdf['Indicator Code'].isin(indicators)]
    hdf.to_csv('myfile.csv')
    
    # find number of countries
    countries = hdf['Country Name'].unique()
    
    # find number of indicator
    ind_count = len(indicators)

    # columns names for the empty data frame
    col_list = ['Country', 'SP.URB.TOTL', 'SP.POP.TOTL', 'SH.DYN.MORT',
                'ER.H2O.FWTL.K3', 'EN.ATM.CO2E.KT', 'AG.LND.FRST.K2',
                'AG.LND.ARBL.ZS', 'AG.LND.AGRI.K2']

    # creating an empty dataframe with columns as col_list
    mod_df = pd.DataFrame(columns=col_list)

    # transform the data for heatmap
    for count in range(len(countries)):
        # temporary dictionary with only country
        temp = {'Country': countries[count]}
        for ind in range(ind_count):
            # finding the key name
            cname = hdf.iloc[(count*ind_count)+ind, 2]
            # finding the value name
            value = hdf.iloc[(count*ind_count)+ind, 3]
            temp[cname] = value

        # append the dict temp to modified dictionary
        mod_df = mod_df.append(temp, ignore_index=True)

    return mod_df


def linfunc(x, a, b):
    """ Function for fitting
    x: independent variable
    a, b: parameters to be fitted
    """
    return a*x + b


# main code
# Reads data from the world bank climate data
df = pd.read_csv('API_19_DS2_en_csv_v2_4902199.csv', skiprows=4)

# filters the data for the required countries
df = FilterData(df)
df = Preprocess(df)

# year for which analysis to be done
year = 2015
# process the data for the heat map
hdf = HeatmapPreprocess(df, year)

# display the heat map
plt.figure(figsize=(8, 6))
ct.map_corr(hdf)
plt.title('Heat Map')
plt.show()

# display the scatter matrix
plt.figure(dpi=600)
axes = pd.plotting.scatter_matrix(hdf, figsize=(9.0, 9.0))
for ax in axes.flatten():
    ax.xaxis.label.set_rotation(90)
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha('right')
plt.tight_layout()    # helps to avoid overlap of labels
plt.show()


# finding n clusters
# extract columns for fitting.
hdf_fit = hdf[['SH.DYN.MORT', 'AG.LND.ARBL.ZS']].copy()

# normalise dataframe and inspect result
hdf_fit, df_min, df_max = ct.scaler(hdf_fit)
print(hdf_fit.describe())

print("n score using silhouette score:")
# loop over trial numbers
for ic in range(2, 15):
    # set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(hdf_fit)

    # extract labels and calculate silhoutte score
    labels = kmeans.labels_
    print(ic, skmet.silhouette_score(hdf_fit, labels))

# display clusters in the graph
nc = 3
kmeans = cluster.KMeans(n_clusters=nc)
kmeans.fit(hdf_fit)

# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_

plt.figure(figsize=(6.0, 6.0), dpi=600)
# scatter plot with colours selected using the cluster numbers
plt.scatter(hdf_fit["SH.DYN.MORT"],
            hdf_fit["AG.LND.ARBL.ZS"],
            c=labels, cmap="tab10")

# show cluster centres
xc = cen[:, 0]
yc = cen[:, 1]
plt.scatter(xc, yc, c="k", marker="d", s=80)

# display the labels and title
plt.xlabel("SH.DYN.MORT")
plt.ylabel("AG.LND.ARBL.ZS")
plt.title("Three Clusters")
plt.show()

# fitting the data
# x and y for the data fitting
x = hdf['SP.POP.TOTL'].to_numpy()
y = hdf['ER.H2O.FWTL.K3'].to_numpy()

# fit parameters
popt, pcorr = opt.curve_fit(linfunc, x, y)
print("Fit parameter", popt)

# extract variances and calculate sigmas
sigmas = np.sqrt(np.diag(pcorr))
y1 = linfunc(x, *popt)

# display the linear fit data
plt.figure(dpi=600)
plt.title("Linear Fit")
plt.plot(x, y, "o", markersize=3, label="data")
plt.plot(x, y1, label="fit")
plt.xlabel('Total Population (in billions)')
plt.ylabel('Annual Freshwater Withdrawals (billion cubic meters)')
plt.legend(loc="upper left")
plt.show()
