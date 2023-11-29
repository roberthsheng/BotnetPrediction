# Description: This file is used to analyze the data and generate the features for the model.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
import glob

df = pd.read_csv('data/cs448b_ipasn.csv')
df['date']= pd.to_datetime(df['date'])
df = df.groupby(['date','l_ipn'],as_index=False).sum()
df['yday'] = df['date'].dt.dayofyear
df['wday'] = df['date'].dt.dayofweek

#creating all IPs

ip0 = df[df['l_ipn']==0]
max0 = np.max(ip0['f'])
ip1 = df[df['l_ipn']==1]
max1 = np.max(ip1['f'])
ip2 = df[df['l_ipn']==2]
max2 = np.max(ip2['f'])
ip3 = df[df['l_ipn']==3]
max3 = np.max(ip3['f'])
ip4 = df[df['l_ipn']==4]
max4 = np.max(ip4['f'])
ip5 = df[df['l_ipn']==5]
max5 = np.max(ip5['f'])
ip6 = df[df['l_ipn']==6]
max6 = np.max(ip6['f'])
ip7 = df[df['l_ipn']==7]
max7 = np.max(ip7['f'])
ip8 = df[df['l_ipn']==8]
max8 = np.max(ip8['f'])
ip9 = df[df['l_ipn']==9]
max9 = np.max(ip9['f'])

f,axarray = plt.subplots(5,2,figsize=(15,20))

axarray[0,0].plot(ip0['yday'],ip0['f'])
axarray[0,0].set_yticklabels([])
axarray[0,0].set_title("Daily server 0 Traffic")


axarray[0,1].plot(ip1['yday'], ip1['f'])
axarray[0,1].set_yticklabels([])
axarray[0,1].set_title("Daily server 1 Traffic")                                


axarray[1,0].plot(ip2['yday'], ip2['f'])
axarray[1,0].set_yticklabels([])
axarray[1,0].set_title("Daily server 2 Traffic")

axarray[1,1].plot(ip3['yday'], ip3['f'])
axarray[1,1].set_yticklabels([])
axarray[1,1].set_title("Daily server 3 Traffic")
                                
axarray[2,0].plot(ip4['yday'], ip4['f'])
axarray[2,0].set_yticklabels([])
axarray[2,0].set_title("Daily server 4 Traffic")

axarray[2,1].plot(ip5['yday'], ip5['f'])
axarray[2,1].set_yticklabels([])
axarray[2,1].set_title("Daily server 5 Traffic")

axarray[3,0].plot(ip6['yday'], ip6['f'])
axarray[3,0].set_yticklabels([])
axarray[3,0].set_title("Daily server 6 Traffic")

axarray[3,1].plot(ip7['yday'], ip7['f'])
axarray[3,1].set_yticklabels([])
axarray[3,1].set_title("Daily server 7 Traffic")

axarray[4,0].plot(ip8['yday'], ip8['f'])
axarray[4,0].set_yticklabels([])
axarray[4,0].set_title("Daily server 8 Traffic")

axarray[4,1].plot(ip9['yday'], ip9['f'])
axarray[4,1].set_yticklabels([])
axarray[4,1].set_title("Daily server 9 Traffic")

plt.savefig('images/daily_traffic.png')