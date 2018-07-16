
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os as os
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import datetime as dt
import math
plt.rcParams['figure.figsize'] = (12, 12)
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
plt.rcParams['axes.titlesize'] = 1.5*plt.rcParams['font.size']
plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
#plt.rcParams['savefig.dpi'] = 3*plt.rcParams['savefig.dpi']
plt.rcParams['xtick.major.size'] = 3
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.major.size'] = 3
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.loc'] = 'center left'
plt.rcParams['axes.linewidth'] = 1

plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.gca().xaxis.set_ticks_position('bottom')
plt.gca().yaxis.set_ticks_position('left')
sns.set_style('whitegrid')
plt.close()

import datetime as datetime
from dateutil.relativedelta import relativedelta
import glob as glob


# ** read_tag_files**
# 
# this function will 
# 
#     1.read all text files from a folder (just keep tag files in this folder no other txt files)
#     2.remove unwanted tag reads: fishtags, other removes which were some mistakes done during installing readers
#     3.If the analysis needs to be restricted to specific location/readers this function will filter the data accordingly

# In[ ]:


def read_tag_files(data_path, select_readers = None, verbose = False, remove_rufous = False):
    """reading all TEXT files"""
    txtfiles = glob.glob(data_path + '\*.txt')
    txtfile = {}
    files=[]
    for txt in txtfiles:
        name  = str(txt).replace('.txt','')
        name  = str(name).replace(data_path, '')
        txtfile[name] = pd.read_csv(txt, sep=' ',header= None, usecols=[0, 1, 2, 3, 4])
        if verbose:
            print (name)
        files.append(txtfile[name])
    data = pd.concat(files, axis=0)
    data.columns = ['Date', 'Time', 'Reader', 'Tag', 'ID']
    data['DateTime'] = pd.to_datetime(data['Date'] + ' ' +data['Time'])
    data['DateTime'] = (data['DateTime']- pd.Timedelta(hours=1)).where((data['Reader'] =='A1'), data['DateTime'])
    data.set_index(pd.DatetimeIndex(data['DateTime']), inplace= True, drop= False)
    data['Time'] = pd.to_datetime(data['Time'], format= '%H:%M:%S' ).dt.time
    data['Date'] = pd.to_datetime(data['Date']).dt.date
    data['ID'] = data['ID'].map(lambda x: str(x)[1:])
    data['ID'] = data['ID'].astype(str).str[0:15]
    data['ID'] = data['ID'].str.strip()
    
    fishtags = ['3DD.003BE90A2C', '3DD.003BE90981', '3DD.003BE8FAA4', '3DD.003BE90953','3DD.003BE90A46', '3DD.003BE9001A']
    rufous = ['3D6.1D593D4615', '3D6.1D593D45D5', '3D6.00184967F6', '3D6.1D593D4602', '3D6.00184967FA', '3D6.1D593D45E3', 
              '3D6.1D593D7833', '3D6.1D593D7846']
    OtherRemoves = ['3D6.00184967EB', '3D6.00184967DE']
    if remove_rufous:
        data = data[~data['ID'].isin(rufous)]
    data = data[~data['ID'].isin(fishtags)]
    data = data[data.ID.str.contains("3DD.") == False]
    data = data[~data['ID'].isin(OtherRemoves)]
    data = data.drop(data[(data['Date'] <datetime.date(2017,7,14)) & (data['ID'] == '3D6.00184967B6')].index)
    data.replace('B3', 'A5', inplace= True)
    data.replace('982000407463844', '3D6.00184967A4', inplace = True)
    data.replace('982000407463855', '3D6.00184967AF', inplace = True)
    data.replace('982000407463853', '3D6.00184967AD', inplace = True)
    data.replace('982000407463871', '3D6.00184967BF', inplace = True)
    if select_readers:
        data = data[data['Reader'].isin(select_readers)]
    
        
    return data
    


# **collapse_reads_10_seconds**
# 
# this function will
# 
#     1. for each individual bird, it will calculate different between each reads
#     2. if the difference is more that 0.11 seconds it will identify it as a new visit.
#     3. for each new visit, it will identify the starting time, ending time
#     

# In[ ]:


def collapse_reads_10_seconds(data):
    birds = data.ID.unique()
    bird_wise_data = []
    for bird in birds.tolist():
        temp_bird = data[data.ID == bird]
        for r in temp_bird.Reader.unique().tolist():
            tempdf = temp_bird[temp_bird.Reader == r]
            tempdf = tempdf.sort_values(by='DateTime',ascending=True)
            tempdf['timediff'] = tempdf.index.to_series().diff().fillna(0)
            bird_wise_data.append(tempdf)
    data = pd.concat(bird_wise_data, axis=0)
    count_visits = []
    for bird in birds.tolist():
        temp_df = data[data.ID == bird].reset_index(0, drop = True).sort_values(by='DateTime',ascending=True)
        for r in temp_df.Reader.unique().tolist():
            temp_reader = temp_df[temp_df.Reader == r].reset_index(0, drop = True).sort_values(by='DateTime',ascending=True)
            #temp_df['visit'] = (temp_df.DateTime.diff() != '0 days 00:00:11').cumsum()
            temp_reader['visit'] = (temp_reader.DateTime.diff() > '0 days 00:00:11').cumsum()
            count_visits.append(temp_reader)
    data = pd.concat(count_visits, axis=0)
    data['visit_ID'] = data["ID"].map(str) +'_'+data.visit.astype(str) +'_'+data["Reader"]
    """function to get start and end time of groupby object"""
    def ohlc(df):
        return (df.DateTime.min(), 
                df.DateTime.max(),
                df.ID.unique()[0],
                df.Reader.unique()[0]) 
    data = pd.DataFrame(zip(*data.groupby(['ID', 'visit_ID']).apply(ohlc))).T
    data.columns = ['visit_start', 'visit_end', 'ID', 'Tag']
    data['visit_duration'] = data.visit_end - data.visit_start
    data.visit_duration.astype('timedelta64[h]')
    return data


# **read metadata file**
# 
# this function will
# 
#     1. read the master metadata file for all the birds PIT tagged until now
#     2. filter the data according to the study location 

# In[ ]:


def read_metadata (data_path, filename, restrict = False):
    p = str(data_path)+"\\" + filename
    metadata = pd.read_excel(p)
    metadata['Tagged Date'] = pd.to_datetime(metadata['Tagged Date']).dt.date
    metadata['Tag Hex'] = metadata['Tag Hex'].astype(str).str[0:15]
    metadata['Tag Hex'] = metadata['Tag Hex'].str.strip()
    rufous = ['3D6.1D593D4615', '3D6.1D593D45D5', '3D6.00184967F6', '3D6.1D593D4602', '3D6.00184967FA', '3D6.1D593D45E3', 
              '3D6.1D593D7833', '3D6.1D593D7846']
    if restrict:
        metadata = metadata[~metadata['Tag Hex'].isin(rufous)]
        metadata = metadata[metadata['Tagged Date'] <= pd.datetime(2018, 2, 28).date()]    
    metadata.replace('M ', 'M', inplace = True)
    metadata['Sex'].fillna('unknown', inplace = True)
    metadata.replace('M ', 'M', inplace = True)
    metadata.replace('HY?', 'HY', inplace = True)
    metadata.replace('SY', 'HY', inplace = True)
    metadata['Sex'].fillna('unknown', inplace = True)
    metadata['Age'].fillna('UNK', inplace = True)
    return metadata


# **merge TAG data and Metadata**
# 
# this function will
#     1. merge the tag read data with the metadata using the 'Tag Hex' column
#     2. Merge the data only after running **collapse_reads_10_seconds** function 
#     3. identifies birds which were tagged and found on the same day in 2016

# In[ ]:


def merge_metadata(data, metadata, for_paper = False):
    data = pd.merge(data, metadata, left_on='ID', right_on='Tag Hex', how='left')
    data['Date'] = pd.to_datetime(data['visit_start']).dt.date
    bird_wise_data = []
    for bird in data.ID.unique().tolist():
        bd = data[data.ID == bird]
        if (bd.visit_start.dt.date.min() == bd['Tagged Date'].unique().tolist()[0]) and (int(bd.visit_start.dt.date.min().year) == 2016):
            print str(bird)+' '+'date same, first recorded ' + str(bd.visit_start.min())+' '+ ' tagged on ' +str(bd['Tagged Date'].unique().tolist()[0])
            bd = bd.loc[bd['visit_start']!=bd['visit_start'].min()]
            bird_wise_data.append(bd)
        else: 
            bird_wise_data.append(bd)
    data = pd.concat(bird_wise_data, axis=0)
    data.set_index('visit_start', drop= False, inplace= True)
    if for_paper:
        data = data[data['visit_start'] <= pd.Timestamp('20180331')]
        data = data[data['Tagged Date'] <= pd.datetime(2018, 2, 28).date()]
        data.drop(data[data.Species == 'HYBRID'].index, inplace=True)
        data.drop(data[(data['Location Id'] == 'GR') | (data.Tag == 'A7')].index,inplace=True)
    return data


# **bird_summaries**
# 
# This function:
# 
# 1. calculates summaries for each tagged bird
# 2. save a csv file with summary output
# 3. columns are 
# 
# Tag Hex,Species,Sex, Age, tagged_date, first_obs_aft_tag, date_min, date_max, obser_period, date_u, Location

# In[ ]:


def bird_summaries(data, output_path, metadata):
    f = {'Date':['nunique','min', 'max'],
     'duration_s':['sum','min','max', 'mean', 'median'],
     'Tag':'unique'
    }
    data['duration_s'] = data.visit_duration.dt.seconds
    Bird = data.groupby('ID').agg(f).reset_index()
    Bird.columns = Bird.columns.get_level_values(1)
    Bird.columns = ['ID', 'date_u', 'date_min', 'date_max', 'readers_visiting',  
                'duration_total', 'shortest_visit', 'longest_visit','duration_mean', 'duration_median',]
    Bird = pd.merge(Bird, metadata, left_on='ID', right_on = 'Tag Hex',how='left')
    Bird['tagged_date'] = pd.to_datetime(Bird['Tagged Date']).dt.date#.str[0]
    Bird['obser_period'] = Bird['date_max'] - Bird['tagged_date']
    Bird['first_obs_aft_tag'] =  Bird['date_min']-Bird['tagged_date']
    reader_prediliction = pd.pivot_table(pd.DataFrame(data.groupby('ID').Tag.value_counts()).rename(columns= {'Tag':'Count',}).reset_index(), 
                   columns = 'Tag', index = 'ID', values = 'Count', fill_value = 0).reset_index()
    Bird = pd.merge(Bird, reader_prediliction, on='ID', how='left')
    reader_prediliction.set_index('ID',drop= True, inplace= True)
    Bird.to_csv(output_path+'\Birds_summary.csv')
    return Bird, reader_prediliction


# In[ ]:


def activity_overview_plot(data, output_path):
    data.Sex.fillna('unknown', inplace= True)
    plt.rcParams['font.size'] = 13
    #plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.5*plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']+1
    plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
    
    def plotvisits(timeunit, data, ax, c = ['#e41a1c', '#377eb8', '#4daf4a']):
        """timeunite: 10Min, D =  day, W = week, M = month"""
        if timeunit == '10Min':
            t = '10 minutes'
        elif timeunit == 'D':
            t = 'day'
        elif timeunit == 'W':
            t = 'week'
        if timeunit == 'M':
            t = 'month'
        #data.groupby['Sex'].resample(timeunit)['ID'].count().unstack('Sex').plot(kind="line", rot =45, ax= ax1,stacked=False, color = c, 
        #                                           title ='number of hummingbird visits per '+t)
        data.groupby([ pd.Grouper(freq=timeunit), 'Sex'])['ID'].count().unstack('Sex').plot(kind="line", rot =45, ax= ax,stacked=False, color = c,)

        ax.set_title('visits per '+t)
        ax.set_xlabel('Time')
        ax.set_ylabel('Number of visits')
    
    f, ((ax1, ax2),(ax3, ax4))  = plt.subplots(2, 2, figsize=(16,12), dpi=400)
    plotvisits(timeunit= '10Min', data= data, ax= ax1)
    plotvisits(timeunit= 'D', data= data, ax= ax2)
    plotvisits(timeunit= 'W', data= data, ax= ax3)
    plotvisits(timeunit= 'M', data= data, ax= ax4)
    f.suptitle('Hummingbird visits to feeders', fontsize = 24)
    f.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path + '\overview_activity_plot.png', dpi = 300)
    plt.show()


# In[ ]:


def cumulative_plot (metadata, output_path):
    fig, ax1 = plt.subplots(1, figsize=(6,4))
    idx = pd.date_range('06.01.2016', '02.28.2018')
    a = metadata.groupby('Tagged Date')['Tag Hex'].nunique()
    a.index = pd.DatetimeIndex(a.index)
    a = a.reindex(idx, fill_value=0)
    tl = a.cumsum()
    tl.columns = ['sampling']
    tl.plot.line(drawstyle = 'steps', label='Animals', ax = ax1)
    ax1.set_ylabel('Individuals tagged', fontsize = 11)
    ax1.set_xlabel('Time', fontsize = 11)
    ax1.set_title('cumulative tagging of birds')
    plt.savefig(output_path+'/'+'cumulative_tagging.png', dpi = 300)
    plt.show()


# In[ ]:
def plotBird (data, bird_ID, output_path, Bird_summary_data):
    df = data[data.ID == bird_ID]
    start_date = Bird_summary_data.loc[Bird_summary_data['Tag Hex'] == bird_ID, 'tagged_date'].iloc[0]
    idx = pd.date_range(start_date, dt.datetime.now(),freq='D')
    ten = df.resample('D')['ID'].count().reindex(idx, fill_value=0).reset_index()
    ten.columns= ['Time','visits']
    ten['Time'] = pd.to_datetime(ten['Time'])
    ten = ten.set_index(pd.DatetimeIndex(ten['Time']))

    fig, ax = plt.subplots(nrows = 1, ncols = 1, sharey= True, sharex=True)
    b1 = ten
    #b1 = getplotBird(bird_IDs[i],)
    b1.drop(b1.columns[0], axis=1, inplace= True)
    b1.plot(ax=ax,  color = '#2ca25f', linewidth=1)
    gender = Bird_summary_data.loc[Bird_summary_data['Tag Hex'] == bird_ID, 'Sex'].iloc[0]
    age = Bird_summary_data.loc[Bird_summary_data['Tag Hex'] == bird_ID, 'Age'].iloc[0]
    ax.set_ylabel('number of feeder visits per day',  fontsize = 14)
    ax.set_title(bird_ID+', '+age+', '+gender, fontsize = 14)

    fig.set_size_inches([8.27 ,5.845])
    plt.tight_layout()
    plt.show()
    fig.savefig(output_path+'/'+ bird_ID +'_Bird_plot.png',dpi=300 )
    plt.tight_layout()
    plt.show()

def plot_individual_birds (data, bird_IDs, output_path, Bird_summary_data):
    def getplotBird (bird_ID):
        df = data[data.ID == bird_ID]
        start_date = Bird_summary_data.loc[Bird_summary_data['Tag Hex'] == bird_ID, 'tagged_date'].iloc[0]
        idx = pd.date_range(start_date, dt.datetime.now(),freq='D')
        ten = df.resample('D')['ID'].count().reindex(idx, fill_value=0).reset_index()
        ten.columns= ['Time','visits']
        ten['Time'] = pd.to_datetime(ten['Time'])
        ten = ten.set_index(pd.DatetimeIndex(ten['Time']))
        return ten
    
    sns.set()
    nrows = int(math.ceil(float(len(bird_IDs))/float(3)))
    ncols = 3
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, sharey= True, sharex=True)
    for i, ax in enumerate(fig.axes):
        if i <= len(bird_IDs)-1:
            b1 = getplotBird(bird_IDs[i],)
            b1.drop(b1.columns[0], axis=1, inplace= True)
            b1.plot(ax=ax,  color = '#2ca25f', linewidth=1)
            gender = Bird_summary_data.loc[Bird_summary_data['Tag Hex'] == bird_IDs[i], 'Sex'].iloc[0]
            age = Bird_summary_data.loc[Bird_summary_data['Tag Hex'] == bird_IDs[i], 'Age'].iloc[0]
            ax.set_ylabel('number of feeder visits per day')
            ax.set_title(bird_IDs[i]+' '+age+' '+gender, fontsize = 14)

    fig.set_size_inches([4.845*ncols, 8.27 ])
    plt.tight_layout()
    plt.show()
    fig.savefig(output_path+'/'+ 'bird_activities.png',dpi=300 )


def plotvisits(timeunit, data, ax, c = ['#e41a1c', '#377eb8', '#4daf4a']):
           
        """timeunite: 10Min, D =  day, W = week, M = month"""
        if timeunit == '10Min':
            t = '10 minutes'
        elif timeunit == 'D':
            t = 'day'
        elif timeunit == 'W':
            t = 'week'
        if timeunit == 'M':
            t = 'month'
        data.groupby([ pd.Grouper(freq=timeunit)])['ID'].count().plot(kind="line", rot =45, ax= ax)

        #ax.set_title('visits per '+t)
        ax.set_xlabel('Time')
        ax.set_ylabel('Number of daily visits')


def plotBirdnight (data, bird_ID, Bird_summary_data, start, end, ax):
    b1 = data.ix[start:end]
    df = b1[b1.ID == bird_ID]
    start_date = start
    idx = pd.date_range(start_date, end,freq='1h')
    ten = df.resample('1h')['ID'].count().reindex(idx, fill_value=0).reset_index()
    ten.columns= ['Time','visits']
    ten['Time'] = pd.to_datetime(ten['Time'])
    ten = ten.set_index(pd.DatetimeIndex(ten['Time']))

    indices=[]
    for i in range(len(ten.index.hour)):
        if (ten.index.hour[i]>=22)|(ten.index.hour[i]<=4):
            indices.append(i)
    del indices[-1]
    
    b1 = ten
    #b1 = getplotBird(bird_IDs[i],)
    b1.drop(b1.columns[0], axis=1, inplace= True)
    n_plot = b1.plot(ax=ax,  color = '#2ca25f', linewidth=1, xticks=b1.index)
    gender = Bird_summary_data.loc[Bird_summary_data['Tag Hex'] == bird_ID, 'Sex'].iloc[0]
    age = Bird_summary_data.loc[Bird_summary_data['Tag Hex'] == bird_ID, 'Age'].iloc[0]
    ax.set_ylabel('hourly feeder visits')
    ax.set_title(bird_ID+', '+age+', '+gender)
    #n_plot.set_xticklabels(labels=["{0:.2f}".format(round(a,2)) for a in ten.index.hour], rotation = 90)
    label = []
    for h in ten.index.hour:
        label.append(datetime.time(h))
    #n_plot.set_xticklabels(labels=label, rotation = 90)

    #ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.grid(True, which="major")
    ax.yaxis.grid(True, which="major")
    for i in indices:
        ax.axvspan(b1.index[i], b1.index[i+1], facecolor='green', edgecolor='none', alpha=.2)
        
        
def bar_plot_report(data, output_path, bin_unit, species = None, month_name= False):
    """
    options for bin unit
    week = 'W'
    15 days = '15D'
    month = 'M'
    3 months = '3M'
    year = '1Y'
    """
    data.Sex.fillna('unknown', inplace= True)
    data.Sex.replace('M ', 'M', inplace= True)
    data.replace('ALHU', "Allen's Hummingbird", inplace= True)
    data.replace('ANHU', "Anna's Hummingbird", inplace= True)
    data.replace("RUHU", "Rufous Hummingbird", inplace = True)
    data.replace('F', "Female", inplace= True)
    data.replace('M', "Male", inplace= True)
    
    import matplotlib.gridspec as mgrid
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    
    fig = plt.figure(figsize=(9, 9))
    grid = mgrid.GridSpec(nrows=2, ncols=1, height_ratios=[1, 3])
    ax1 = fig.add_subplot(grid[1])
    ax2 = fig.add_subplot(grid[0])
    #c = ['#f2595b', '#377eb8']
    c = ['#f2595b', '#377eb8', '#4daf4a']

    if species:
        sp_data = data[data.Species == species]
    else:
        sp_data = data
    minimum_date = min(data.index)- pd.Timedelta(weeks=1)
    a = sp_data.groupby([pd.Grouper(freq= bin_unit), 'Sex'])['ID'].count()
    multi_index = pd.MultiIndex.from_product([pd.date_range(minimum_date.date(), dt.datetime.now(), freq=bin_unit), data['Sex'].unique()], names=['Date', 'Sex'])
    a = a.reindex(multi_index, fill_value=0)
    a.unstack('Sex').plot(kind="bar", rot =90, stacked=False, ax = ax1, color = c)

    if month_name:
        ax1.set_xticklabels(a.index.levels[0].strftime('%Y %B'))  
        for tick in ax1.get_xticklabels():
            tick.set_rotation(0)
    else:
        ax1.set_xticklabels(a.index.levels[0].strftime('%m/%d\n%Y'))
        
        
    for p in ax1.patches:
        ax1.annotate(str(p.get_height()), (p.get_x() * 1.00, p.get_height() * 1.015))
    #for index, label in enumerate(ax1.xaxis.get_ticklabels()):
    #    if index % 15 != 0:
    #        label.set_visible(False)

    ax1.set_xlabel('Time')
    ax1.set_title('B')
    ax1.set_ylabel('Number of feeder visits\nby tagged birds',  fontsize = 16)
    
    a = sp_data.groupby([pd.Grouper(freq= bin_unit), 'Sex'])['ID'].nunique()
    multi_index = pd.MultiIndex.from_product([pd.date_range(minimum_date.date(), dt.datetime.now(), freq=bin_unit), data['Sex'].unique()], names=['Date', 'Sex'])
    a = a.reindex(multi_index, fill_value=0)
    a.unstack('Sex').plot(kind="bar", rot =90, stacked=False, ax = ax2, color = c)
    for p in ax2.patches:
        ax2.annotate(str(p.get_height()), (p.get_x() * 1.00, p.get_height() * 1.015))

    if month_name:
        ax2.set_xticklabels(a.index.levels[0].strftime('%Y %B'))  
        for tick in ax2.get_xticklabels():
            tick.set_rotation(0)
    else:
        ax2.set_xticklabels(a.index.levels[0].strftime('%m/%d\n%Y'))

    ax2.set_xlabel('Time')
    ax2.set_title('A')
    ax2.set_ylabel('Number of tagged birds\nvisitng feeders', fontsize = 16)


    fig.tight_layout()
    plt.savefig(output_path+'/'+'barplot_birds.png')
    plt.show()


# functions for getting summaries for tagged birds

# In[ ]:


def report_Table1A (metadata, location = None):
    if location:
        metadata2 = metadata[metadata['Location Id'].isin(location)]
    else:
        metadata2 = metadata
    metadata2.replace("ALHU", "Allen's Hummingbird", inplace = True)
    metadata2.replace("ANHU", "Anna's Hummingbird", inplace = True)
    metadata2.replace("RUHU", "Rufous Hummingbird", inplace = True)
    metadata2.replace("AHY", "after hatch year", inplace = True)
    metadata2.replace("HY", "hatch year", inplace = True)
    metadata2.replace("UNK", "unknown", inplace = True)
    metadata2.replace("M", "male", inplace = True)
    metadata2.replace("F", "female", inplace = True)
    Tagged_Gender = metadata2.pivot_table(index='Sex', columns='Species', aggfunc= 'count', values='Tag Hex',
                                 fill_value=0, margins=True).astype(int)
    return Tagged_Gender

def report_Table2Paper (metadata, location = None):
    if location:
        metadata2 = metadata[metadata['Location Id'].isin(location)]
    else:
        metadata2 = metadata
    metadata2.replace("ALHU", "Allen's Hummingbird", inplace = True)
    metadata2.replace("ANHU", "Anna's Hummingbird", inplace = True)
    metadata2.replace("RUHU", "Rufous Hummingbird", inplace = True)
    metadata2.replace("AHY", "after hatch year", inplace = True)
    metadata2.replace("HY", "hatch year", inplace = True)
    metadata2.replace("UNK", "unknown", inplace = True)
    metadata2.replace("M", "male", inplace = True)
    metadata2.replace("F", "female", inplace = True)
    Table_2 = metadata2.pivot_table(index='Location Id', columns='Species', aggfunc= 'count', values='Tag Hex',
                                 fill_value=0, margins=True).astype(int)
    return Table_2

def report_Table1B (metadata, location = None):
    if location:
        metadata2 = metadata[metadata['Location Id'].isin(location)]
    else:
        metadata2 = metadata
    metadata2.replace("ALHU", "Allen's Hummingbird", inplace = True)
    metadata2.replace("ANHU", "Anna's Hummingbird", inplace = True)
    metadata2.replace("RUHU", "Rufous Hummingbird", inplace = True)
    metadata2.replace("AHY", "after hatch year", inplace = True)
    metadata2.replace("HY", "hatch year", inplace = True)
    metadata2.replace("UNK", "unknown", inplace = True)
    metadata2.replace("M", "male", inplace = True)
    metadata2.replace("F", "female", inplace = True)
    Tagged_Age = metadata2.pivot_table(index='Age', columns='Species', aggfunc= 'count', values='Tag Hex',
                                 fill_value=0, margins=True).astype(int)
    return Tagged_Age

    


# functions for getting summaries for birds detected birds

def paper_Table1 (metadata, location = None):
    if location:
        metadata2 = metadata[metadata['Location Id'].isin(location)]
    else:
        metadata2 = metadata
    metadata2.replace("ALHU", "Allen's Hummingbird", inplace = True)
    metadata2.replace("ANHU", "Anna's Hummingbird", inplace = True)
    metadata2.replace("RUHU", "Rufous Hummingbird", inplace = True)
    metadata2.replace("AHY", "after hatch year", inplace = True)
    metadata2.replace("HY", "hatch year", inplace = True)
    metadata2.replace("UNK", "unknown", inplace = True)
    metadata2.replace("M", "male", inplace = True)
    metadata2.replace("F", "female", inplace = True)
    Tagged_Age = metadata2.pivot_table(columns=['Sex', 'Age'], index='Species', aggfunc= 'count', values='Tag Hex',
                                 fill_value=0, margins=True).astype(int)
    return Tagged_Age

# In[ ]:

def paper_Table2 (data, metadata, location = None):
    if location:
        metadata2 = metadata[metadata['Location Id'].isin(location)]
    else:
        metadata2 = metadata
    metadata2.replace("ALHU", "Allen's Hummingbird", inplace = True)
    metadata2.replace("ANHU", "Anna's Hummingbird", inplace = True)
    metadata2.replace("RUHU", "Rufous Hummingbird", inplace = True)
    metadata2.replace("AHY", "after hatch year", inplace = True)
    metadata2.replace("HY", "hatch year", inplace = True)
    metadata2.replace("UNK", "unknown", inplace = True)
    metadata2.replace("M", "male", inplace = True)
    metadata2.replace("F", "female", inplace = True)
    visit_metadata = metadata2[metadata2['Tag Hex'].isin(data['Tag Hex'].unique())]
    Visit_birds = visit_metadata.pivot_table(columns=['Sex', 'Age'], index='Species', aggfunc= 'count', 
                                values='Tag Hex',fill_value=0, margins= True).astype(int)
    return Visit_birds



def report_Table2A (data, metadata, location = None):
    if location:
        metadata2 = metadata[metadata['Location Id'].isin(location)]
    else:
        metadata2 = metadata
    metadata2.replace("ALHU", "Allen's Hummingbird", inplace = True)
    metadata2.replace("ANHU", "Anna's Hummingbird", inplace = True)
    metadata2.replace("RUHU", "Rufous Hummingbird", inplace = True)
    metadata2.replace("AHY", "after hatch year", inplace = True)
    metadata2.replace("HY", "hatch year", inplace = True)
    metadata2.replace("UNK", "unknown", inplace = True)
    metadata2.replace("M", "male", inplace = True)
    metadata2.replace("F", "female", inplace = True)
    visit_metadata = metadata2[metadata2['Tag Hex'].isin(data['Tag Hex'].unique())]
    Visit_gender = visit_metadata.pivot_table(index='Sex', columns='Species', aggfunc= 'count', 
                                values='Tag Hex',fill_value=0, margins= True).astype(int)
    return Visit_gender

def report_Table2B (data, metadata, location= None):
    if location:
        metadata2 = metadata[metadata['Location Id'].isin(location)]
    else:
        metadata2 = metadata
    metadata2.replace("ALHU", "Allen's Hummingbird", inplace = True)
    metadata2.replace("ANHU", "Anna's Hummingbird", inplace = True)
    metadata2.replace("AHY", "after hatch year", inplace = True)
    metadata2.replace("RUHU", "Rufous Hummingbird", inplace = True)
    metadata2.replace("HY", "hatch year", inplace = True)
    metadata2.replace("UNK", "unknown", inplace = True)
    metadata2.replace("M", "male", inplace = True)
    metadata2.replace("F", "female", inplace = True)
    visit_metadata = metadata2[metadata2['Tag Hex'].isin(data['Tag Hex'].unique())]
    Visit_age = visit_metadata.pivot_table(index='Age', columns='Species', aggfunc= 'count', values='Tag Hex',  fill_value=0,
                                      margins= True).astype(int)
    return Visit_age



def diurnal_variation(dataframe, output_path, circular = True, normalize = False, get_df = False):
    
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.5*plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']+1
    plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
    
    agg_funcs = {'visits':[np.mean, np.sum, np.std, 'count', 'sem']}
    idx = pd.date_range('2016-9-22', dataframe.visit_start.max().date(),freq='1H')
    if normalize:
        agg_funcs_s = {'ID':['count', 'nunique']}
        hour = dataframe.resample('1H').agg(agg_funcs_s)#.reindex(idx, fill_value=0).reset_index()
        hour.columns = hour.columns.get_level_values(1)
        hour['c'] =hour['count']/hour['nunique'] 
        hour['c'].fillna(0, inplace = True)
        hour.drop(['count', 'nunique'],axis=1,  inplace= True)
        hour = hour.reindex(idx, fill_value=0).reindex(idx, fill_value=0).reset_index()
    else:
        hour = dataframe.resample('1H')['ID'].count().reindex(idx, fill_value=0).reset_index()
        
    hour.columns= ['Time','visits']
    hour['Time'] = pd.to_datetime(hour['Time'])
    hour = hour.set_index(pd.DatetimeIndex(hour['Time']))
    hour['h'] = pd.to_datetime(hour['Time'], format= '%H:%M:%S' ).dt.time
    
    dataframe['Time'] = pd.to_datetime(dataframe['visit_start'], format= '%H:%M:%S' ).dt.time
    f1 = dataframe.ix['2016-9-23':'2016-12-20']
    f2 = dataframe.ix['2017-9-22':'2017-12-21']
    
    f = pd.concat([f1, f2], axis=0)
    print 'earliest visit in fall ' +str (f['Time'].min())
    print 'latest visit in fall ' +str (f['Time'].max())
    w1 = dataframe.ix['2016-12-21':'2017-03-19']
    w2 = dataframe.ix['2017-12-22':'2018-03-19']
    w = pd.concat([w1, w2], axis=0)
    print 'earliest visit in winter ' +str (w['Time'].min())
    print 'latest visit in winter ' +str (w['Time'].max())
    
    sp1 = dataframe.ix['2017-03-20':'2017-06-19']
    sp2 = dataframe.ix['2018-03-20':'2018-04-30']
    sp = pd.concat([sp1, sp2], axis=0)
    print 'earliest visit in spring ' +str (sp['Time'].min())
    print 'latest visit in spring ' +str (sp['Time'].max())
    s1 = dataframe.ix['2017-06-20':'2017-9-22']
    s2 = dataframe.ix['2018-06-22':'2018-9-22']
    s = pd.concat([s1, s2], axis=0)
    print 'earliest visit in summer ' +str (s['Time'].min())
    print 'latest visit in summer ' +str (s['Time'].max())
    
    year = hour.groupby(hour.h).agg(agg_funcs)
    
    fall1 = hour.ix['2016-9-23':'2016-12-20']
    fall2 = hour.ix['2017-9-23':'2017-12-21']
    falld = pd.concat([fall1, fall2], axis=0)
    fall = falld.groupby(falld.h).agg(agg_funcs)
    fall=fall.rename(columns = {'visits':'fall'})

    winter1 = hour.ix['2016-12-21':'2017-03-19']
    winter2 = hour.ix['2017-12-22':'2018-03-19']
    winterd = pd.concat([winter1, winter2], axis=0)
    winter = winterd.groupby(winterd.h).agg(agg_funcs)
    winter=winter.rename(columns = {'visits':'winter'})

    spring1 = hour.ix['2017-03-20':'2017-06-19']
    spring2 = hour.ix['2018-03-20':'2018-04-30']
    springd = pd.concat([spring1, spring2], axis=0)
    
    spring = springd.groupby(springd.h).agg(agg_funcs)
    spring=spring.rename(columns = {'visits':'spring'})

    summer1 = hour.ix['2017-06-20':'2017-9-22']
    summer2 = hour.ix['2018-06-22':'2018-9-22']
    summerd = pd.concat([summer1, summer2], axis=0)
    summer = summerd.groupby(summerd.h).agg(agg_funcs)
    summer=summer.rename(columns = {'visits':'summer'})

    hourly_summary = pd.concat([fall, winter, spring, summer, year], axis=1)
    hourly_summary = hourly_summary.reset_index()
    hourly_summary['h'] = hourly_summary['h'].apply(lambda v: str(v))

    #hourly_summary.to_excel('hourly_summary.xlsx')
    fall.columns = fall.columns.get_level_values(1)
    summer.columns = summer.columns.get_level_values(1)
    winter.columns = winter.columns.get_level_values(1)
    spring.columns = spring.columns.get_level_values(1)
    def plot_circular(df, ax):
        # make a polar plot
        N = 24
        bottom = 0
        width = (2*np.pi) / N
        # create theta for 24 hours
        theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
        bars = ax.bar(theta, df['mean'].values, width=width, bottom=bottom, yerr = df['sem'].values)
        # set the lable go clockwise and start from the top
        ax.set_theta_zero_location("S")
        # clockwise
        ax.set_theta_direction(-1)
        # set the label
        ticks = ['0:00', '3:00', '6:00', '9:00', '12:00', '15:00', '18:00', '21:00']
        ax.set_xticklabels(ticks)

    if circular:
        rlim = 3.5
        f, ((ax1, ax2),(ax3, ax4))  = plt.subplots(2, 2, figsize=(12,9), dpi=400, subplot_kw=dict(projection='polar'))
        plot_circular(df = fall, ax = ax1)
        plot_circular(df = winter, ax = ax2)
        plot_circular(df = spring, ax = ax3)
        plot_circular(df = summer, ax = ax4)
        if normalize:
            ax1.set_rlim(0, rlim)
            ax2.set_rlim(0, rlim)
            ax3.set_rlim(0, rlim)
            ax4.set_rlim(0, rlim)
        from astropy.stats import rayleightest
        ax1.set_title('fall: p = '+"%.3f" % rayleightest(fall['mean']), fontsize = 12)
        ax2.set_title('winter: p = '+ "%.3f" % rayleightest(winter['mean']), fontsize = 12)
        ax3.set_title('spring: p = '+"%.3f" % rayleightest(spring['mean']), fontsize = 12)
        ax4.set_title('summer: p = '+"%.3f" % rayleightest(summer['mean']), fontsize = 12)
        #if normalize:
        #    f.suptitle('Diel variation of tagged hummingbirds at feeders\n(average hourly visits per bird)', fontsize = 24)
        #else:
        #    f.suptitle('Diel variation of tagged hummingbirds at feeders', fontsize = 24)
        f.tight_layout(rect=[0, 0.03, 1, 0.95])
    else:

        f, ((ax1, ax2),(ax3, ax4))  = plt.subplots(2, 2, figsize=(12,9), dpi=400, sharey=True, )
        fall.plot(y='mean', kind = 'bar', yerr='sem', color='#3182bd',ax= ax1 )
        winter.plot(y='mean', kind = 'bar', yerr='sem', color='#3182bd',ax= ax2)
        spring.plot(y='mean', kind = 'bar', yerr='sem', color='#3182bd',ax= ax3)
        summer.plot(y='mean', kind = 'bar', yerr='sem',color='#3182bd', ax= ax4)
        if normalize:
            yaxis_text = 'average hourly visits per bird'
        else:
            yaxis_text = 'average hourly visits'

        ax1.set_ylabel(yaxis_text)
        ax2.set_ylabel(yaxis_text)
        ax3.set_ylabel(yaxis_text)
        ax4.set_ylabel(yaxis_text)
        ax1.set_xlabel('time')
        ax2.set_xlabel('time')
        ax3.set_xlabel('time')
        ax4.set_xlabel('time')

        ax1.set_title('fall')
        ax2.set_title('winter')
        ax3.set_title('spring')
        ax4.set_title('summer')
        ticks = [ x.strftime('%H:%M') for x in fall.index.values ]
        ax1.set_xticklabels(ticks)
        ax2.set_xticklabels(ticks)
        ax3.set_xticklabels(ticks)
        ax4.set_xticklabels(ticks)

        #f.suptitle('Diel variation of tagged hummingbirds at feeders', fontsize = 24)
        f.tight_layout(rect=[0, 0.03, 1, 0.95])
    if normalize:
        plt.savefig(output_path+'\diel_plot_adjusted.png')
    else:
        plt.savefig(output_path+'\diel_plot.png')

    plt.show()
    from astropy.stats import rayleightest
    print ('Rayleigh test identify a non-uniform distribution, i.e. it is\ndesigned for detecting an unimodal deviation from uniformity')
    print 'winter: p = ', "%.3f" % rayleightest(winter['mean'])
    print 'spring: p = ',"%.3f" % rayleightest(spring['mean'])
    print 'summer: p = ',"%.3f" % rayleightest(summer['mean'])
    print 'fall: p = ',"%.3f" % rayleightest(fall['mean'])
    if get_df:
        return winter, spring, summer, fall
    



def plot_predilection(reader_predilection, output_path):
    plt.rcParams['ytick.labelsize'] = 8
    fig, ax1 = plt.subplots(1, figsize=(15,24))
    ax1.yaxis.label.set_size(10)
    sns.heatmap(reader_predilection.replace(0, np.nan), cmap='Blues', annot= True, fmt='.0f', cbar=False, ax = ax1)
    ax1.set_ylabel('tagged hummingbirds', fontsize = 16)
    ax1.set_xlabel('readers', fontsize = 16)
    ax1.set_title('Reader predilection of tagged hummingbirds')
    plt.savefig(output_path+'/'+'reader_predilection.png', dpi = 300)
    plt.show()
    plt.rcParams['ytick.labelsize'] = 11
    






















