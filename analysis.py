# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 14:11:32 2023

@author: adenh
"""

import sys, os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import display

#%%

for i in np.arange(2011,2022):
    # Declare all variables as strings. Spaces must be replaced with '+', i.e., change 'John Smith' to 'John+Smith'.
    # Define the lat, long of the location and the year
    lat, lon, year = 33.2164, -97.1292, 2010
    # You must request an NSRDB api key from the link above
    api_key = 'RLfe0FuFU0F6ZWgOFjRgzIbS8iRukfVQWIeKVgqv'
    # Set the attributes to extract (e.g., dhi, ghi, etc.), separated by commas.
    attributes = 'ghi,dhi,dni,wind_speed,air_temperature,solar_zenith_angle'
    # Choose year of data
    year = i
    # Set leap year to true or false. True will return leap day data if present, false will not.
    leap_year = 'false'
    # Set time interval in minutes, i.e., '30' is half hour intervals. Valid intervals are 30 & 60.
    interval = '60'
    # Specify Coordinated Universal Time (UTC), 'true' will use UTC, 'false' will use the local time zone of the data.
    # NOTE: In order to use the NSRDB data in SAM, you must specify UTC as 'false'. SAM requires the data to be in the
    # local time zone.
    utc = 'false'
    # Your full name, use '+' instead of spaces.
    your_name = 'Aden+Hageman'
    # Your reason for using the NSRDB.
    reason_for_use = 'Energy+project'
    # Your affiliation
    your_affiliation = 'UIowa'
    # Your email address
    your_email = 'aden-hageman@uiowa.edu'
    # Please join our mailing list so we can keep you up-to-date on new developments.
    mailing_list = 'false'
    # Declare url string
    url = 'https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=api_key, attr=attributes)
    # Return just the first 2 lines to get metadata:
    info = pd.read_csv(url, nrows=1)
    # See metadata for specified properties, e.g., timezone and elevation
    timezone, elevation = info['Local Time Zone'], info['Elevation']
    # Return all but first 2 lines of csv to get data:
    df = pd.read_csv('https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=api_key, attr=attributes), skiprows=2)
    # Set the time index in the pandas dataframe:
    df = df.set_index(pd.date_range('1/1/{yr}'.format(yr=year), freq=interval+'Min', periods=8760))
    # take a look
    print('shape:', df.shape, i)
    AOIEV = df['Solar Zenith Angle'] + 30
    AOICAM = df['Solar Zenith Angle'] + 9.46
    SVFEV = (1 + np.cos(30))/2
    SVFCAM = (1 + np.cos(9.46))/2
    albedo = 0.1
    df['GTI_EV'] = ((df['DHI'] * SVFEV) + (df['GHI'] * albedo * (1-SVFEV)) + (df['DNI'] * np.sin(AOIEV)))
    df['GTI_CAM'] = ((df['DHI'] * SVFCAM) + (df['GHI'] * albedo * (1-SVFCAM)) + (df['DNI'] * np.sin(AOICAM)))
    df.to_csv(f'NSRDBpsm3{year}',',')

#%%

df = pd.read_csv('')

plt.style.use('ggplot')

def nsrdb_plot(df, i):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    df['90 Degree Zenith'] = 90
    df[['GHI', 'DNI', 'DHI', 'Solar Zenith Angle', '90 Degree Zenith']][i:i+int(interval)].plot(ax=ax, figsize=(15,8), yticks=(np.arange(0,900,100)), style={'90 Degree Zenith': '--','Solar Zenith Angle': '-o', 'DNI': '-o', 'DHI': '-o', 'GHI': '-o'}, legend=False)
    df[b'generation'][i:i+30].plot(ax=ax2, yticks=(np.arange(0,4.5,0.5)), style={'generation': 'y-o'})
    ax.grid()
    ax.set_ylabel('W/m2')
    ax2.set_ylabel('kW')
    ax.legend(loc=2, ncol=5, frameon=False)
    ax2.legend(loc=1, frameon=False)

nsrdb_plot(df,5000)

#%%

df = pd.read_csv('NSRDBpsm32011')

for i in np.arange(2013,2021):
    df2 = pd.read_csv(f'NSRDBpsm3{i}')
    df = df.append(df2)
    
df.to_csv('NSRDBData.csv')

#%%

df = pd.read_csv('NSRDBData.csv')
df = df.iloc[3192:,:]
df2 = pd.read_excel('Bus Barn and Car Charging Solar Data.xlsx',sheet_name=0,usecols = [0,1,3,4],skiprows = 1,header = 0)
df3 = pd.read_excel('Bus Barn and Car Charging Solar Data.xlsx',sheet_name=1,usecols = [0,1,3,4],skiprows = 1,header = 0)
df2['EL_Solar_BusBarn_Total_KW'] = pd.to_numeric(df2['EL_Solar_BusBarn_Total_KW'], errors='coerce')
df2['EL_Solar_BusBarn_KWH_Dtot'] = pd.to_numeric(df2['EL_Solar_BusBarn_KWH_Dtot'], errors='coerce')
df2['EL_Solar_BusBarn_Total_KW'].fillna(0,inplace=True)
df2['EL_Solar_BusBarn_KWH_Dtot'].fillna(0,inplace=True)
df2.loc[df2['EL_Solar_BusBarn_Total_KW'] > 10000, 'EL_Solar_BusBarn_Total_KW'] = 0
df2.loc[df2['EL_Solar_BusBarn_KWH_Dtot'] > 10000, 'EL_Solar_BusBarn_KWH_Dtot'] = 0
df2 = df2.iloc[2:len(df)]
df3['EL_Solar_CarCharging_total_KW'] = pd.to_numeric(df3['EL_Solar_CarCharging_total_KW'], errors='coerce')
df3['EL_Solar_CarCharging_KWH_Dtot'] = pd.to_numeric(df3['EL_Solar_CarCharging_KWH_Dtot'], errors='coerce')
df3['EL_Solar_CarCharging_total_KW'].fillna(0,inplace=True)
df3['EL_Solar_CarCharging_KWH_Dtot'].fillna(0,inplace=True)
df3.loc[df3['EL_Solar_CarCharging_total_KW'] > 10000, 'EL_Solar_CarCharging_total_KW'] = 0
df3.loc[df3['EL_Solar_CarCharging_KWH_Dtot'] > 10000, 'EL_Solar_CarCharging_KWH_Dtot'] = 0
df3 = df3.iloc[2:len(df)]
df['DailyGTI_EV'] = df['GTI_EV'].rolling(window=24).sum()
df['DailyGTI_CAM'] = df['GTI_CAM'].rolling(window=24).sum()
df['Eff_EV'] = np.abs((df2['EL_Solar_BusBarn_Total_KW'] / df['GTI_EV'])) 
df['DailyEff_EV'] = np.abs((df2['EL_Solar_BusBarn_KWH_Dtot'] / df['DailyGTI_EV']))
df['DailyEff_EV'].replace(np.inf, 0, inplace=True)
df['Eff_EV'].replace(np.inf, 0, inplace=True)
df['Eff_CAM'] = np.abs((df3['EL_Solar_CarCharging_total_KW'] / df['GTI_CAM'])) 
df['DailyEff_CAM'] = np.abs((df3['EL_Solar_CarCharging_KWH_Dtot'] / df['DailyGTI_CAM']))
df['DailyEff_CAM'].replace(np.inf, 0, inplace=True)
df['Eff_CAM'].replace(np.inf, 0, inplace=True)
df.to_csv('FinalData.csv')

#%%

def line(x,a,b):
    return a*x+b

#%%

df = pd.read_csv('FinalData.csv')
df.loc[df['DailyEff_CAM'] > 1000, 'DailyEff_CAM'] = 0
plt.plot(np.arange(1305),df['DailyEff_CAM'].iloc[1:1306])
plt.figure()
plt.plot(np.arange(1322),df['DailyEff_EV'].iloc[1:1323])
plt.figure()
plt.plot(np.arange())
plt.figure()
plt.plot(np.arange())

#%%

#plt.plot(np.arange(5000),df2['EL_Solar_BusBarn_KWH_Dtot'].iloc[2:5002])
#plt.figure()
plt.plot(np.arange(len(df['DailyGTI_CAM'])),df['DailyGTI_CAM'])

plt.xlabel('Time (Hrs)')
plt.ylabel('Power (kW)')
plt.title('Cambus Array Solar Irradiation')
plt.figure()

#plt.plot(np.arange(5000),df3['EL_Solar_CarCharging_KWH_Dtot'].iloc[2:5002])
plt.plot(np.arange(len(df['DailyGTI_EV'])),df['DailyGTI_EV'])

plt.xlabel('Time (Hrs)')
plt.ylabel('Power (kW)')
plt.title('EV Array Solar Irradiation')

#%%

plt.plot(np.arange(50000),df2['EL_Solar_BusBarn_Total_KW'].iloc[2:50002])
#plt.plot(np.arange(len(df['DailyGTI_CAM'])),df['DailyGTI_CAM'])

plt.xlabel('Time (Hrs)')
plt.ylabel('Power (kW)')
plt.title('Cambus Array Generated Power')
plt.figure()

plt.plot(np.arange(50000),df3['EL_Solar_CarCharging_total_KW'].iloc[2:50002])
#plt.plot(np.arange(len(df['DailyGTI_EV'])),df['DailyGTI_EV'])

plt.xlabel('Time (Hrs)')
plt.ylabel('Power (kW)')
plt.title('EV Array Generated Power')

#%%

df = pd.read_csv('FinalData.csv')

#plt.plot(np.arange(70000),df2['EL_Solar_BusBarn_Total_KW'].iloc[2:70002])
#plt.plot(np.arange(len(df['DailyGTI_CAM'])),df['DailyGTI_CAM'])

plt.scatter(np.arange(45000),np.abs(df2['EL_Solar_BusBarn_Total_KW'].iloc[2:45000]/df['DailyGTI_CAM'].iloc[2:45000])*100)
plt.xlabel('Time (Days)')
plt.ylabel('Efficiency')
plt.title('Cambus Array Efficiency')

plt.figure()

plt.scatter(np.arange(45000),np.abs(df3['EL_Solar_CarCharging_total_KW'].iloc[2:45000]/df['DailyGTI_EV'].iloc[2:45000])*100)
plt.xlabel('Time (Days)')
plt.ylabel('Efficiency')
plt.title('EV Array Efficiency')

