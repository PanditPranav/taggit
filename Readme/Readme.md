
# Use of RFID technology to characterize hummingbird feeder visitations and social network 

### Ruta R. Bandivadekar, Pranav S. Pandit, Rahel Sollmann, Michael J. Thomas, Scott Logan, Jennifer Brown, A.P. Klimley, and Lisa A. Tell

The notebook demonstrates analysis presented in the manuscript 

We developed a Python package *'taggit'* for reading the tag detection data collapsing it into visits and visualization of PIT tagged hummigbird activity at the novel feeding stations.  


```python
## setting up local path and importing python packages
data_path = 'C:/Users/Falco/Desktop/directory/HHCP/RFID'
path = 'C:/Users/Falco/Desktop/directory/taggit'
output_path = 'C:/Users/Falco/Desktop/directory/taggit/outputs'

## importing python packages
import os as os
import pandas as pd
%matplotlib inline 
from bokeh.io import show, output_file , output_notebook, save
output_notebook()
from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, TapTool, BoxSelectTool, BoxZoomTool, ResetTool, PanTool, WheelZoomTool, graphs
from bokeh.palettes import Spectral4
from matplotlib import pyplot as plt
os.chdir(path)
from scipy import stats 
import numpy as np
from scipy import stats
import joypy
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm

# Importing the taggit package
from taggit import functions as HT
from taggit import interactions as Hxnet

import matplotlib.style as style
style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1
dpi = 1000
```



    <div class="bk-root">
        <a href="https://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
        <span id="b9fd9bf8-bf54-4282-857f-d76fec14a397">Loading BokehJS ...</span>
    </div>






    <div class="bk-root">
        <a href="https://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
        <span id="6bcfcb51-7bd2-499a-aa7f-5013b0a77901">Loading BokehJS ...</span>
    </div>




## importing data
function: ** read_tag_files**

this function will 

    1.read all text files from a folder (just keep tag files in this folder no other txt files)
    2.remove unwanted tag reads: fishtags, other removes which were some mistakes done during installing readers
    3.ability to restrict analysis to to specific location/readers this function will filter the data accordingly
    4.for the current analysis presented in the paper we are not inlcuding rufous humminbirds


```python
tag_data = HT.read_tag_files(data_path = data_path+'\Paper', select_readers = ['A1', 'A4', 'A5', 'A8', 'A9', 'B1', 'B2', "B3"], 
                            remove_rufous= True)
```

Beacuse of battery and power problems A4 reader was not function for some time. We use data from A2 reader which for a top reader from the DAFS for that duration 


```python
A2 = HT.read_tag_files(data_path = data_path+'\Paper', select_readers = ['A2'], 
                            remove_rufous= True)
A2 = A2.ix['2018-01-26':'2018-04-30']
A2.Reader.replace('A2', 'A4', inplace= True)
tag_data = pd.concat([tag_data, A2])
```

    C:\Users\Falco\Anaconda2\lib\site-packages\ipykernel_launcher.py:3: DeprecationWarning: 
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing
    
    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated
      This is separate from the ipykernel package so we can avoid doing imports until
    

see first 5 lines in the dataframe


```python
tag_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Time</th>
      <th>Reader</th>
      <th>Tag</th>
      <th>ID</th>
      <th>DateTime</th>
    </tr>
    <tr>
      <th>DateTime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-09-09 09:55:46</th>
      <td>2016-09-09</td>
      <td>10:55:46</td>
      <td>A1</td>
      <td>TAG</td>
      <td>3D6.00184967C6</td>
      <td>2016-09-09 09:55:46</td>
    </tr>
    <tr>
      <th>2016-09-09 10:23:19</th>
      <td>2016-09-09</td>
      <td>11:23:19</td>
      <td>A1</td>
      <td>TAG</td>
      <td>3D6.00184967AE</td>
      <td>2016-09-09 10:23:19</td>
    </tr>
    <tr>
      <th>2016-09-09 10:29:56</th>
      <td>2016-09-09</td>
      <td>11:29:56</td>
      <td>A1</td>
      <td>TAG</td>
      <td>3D6.00184967C6</td>
      <td>2016-09-09 10:29:56</td>
    </tr>
    <tr>
      <th>2016-09-09 10:58:03</th>
      <td>2016-09-09</td>
      <td>11:58:03</td>
      <td>A1</td>
      <td>TAG</td>
      <td>3D6.00184967C6</td>
      <td>2016-09-09 10:58:03</td>
    </tr>
    <tr>
      <th>2016-09-09 11:29:04</th>
      <td>2016-09-09</td>
      <td>12:29:04</td>
      <td>A1</td>
      <td>TAG</td>
      <td>3D6.00184967C6</td>
      <td>2016-09-09 11:29:04</td>
    </tr>
  </tbody>
</table>
</div>




```python
print ('number of tag reads during study period= '+str(tag_data.shape[0]))
```

    number of tag reads during study period= 118234
    

in the manuscript we are analysing data only until March 2018


```python
tag_data = tag_data.ix[:'2018-03-31']
```

    C:\Users\Falco\Anaconda2\lib\site-packages\ipykernel_launcher.py:1: DeprecationWarning: 
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing
    
    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated
      """Entry point for launching an IPython kernel.
    

### list of bird ids visiting the feeder


```python
tag_data.ID.unique()
```




    array(['3D6.00184967C6', '3D6.00184967AE', '3D6.00184967B9',
           '3D6.00184967D6', '3D6.00184967D3', '3D6.00184967D7',
           '3D6.00184967B5', '3D6.00184967A3', '3D6.00184967A1',
           '3D6.00184967C3', '3D6.00184967A9', '3D6.00184967B0',
           '3D6.00184967AB', '3D6.00184967DB', '3D6.00184967C4',
           '3D6.00184967A8', '3D6.00184967D0', '3D6.00184967F4',
           '3D6.00184967BE', '3D6.00184967B6', '3D6.00184967D2',
           '3D6.00184967A4', '3D6.00184967EF', '3D6.00184967CB',
           '3D6.00184967F5', '3D6.0018496801', '3D6.00184967DA',
           '3D6.00184967CC', '3D6.00184967C5', '3D6.00184967E4',
           '3D6.00184967FE', '3D6.00184967E1', '3D6.00184967DD',
           '3D6.00184967D9', '3D6.0018496802', '3D6.00184967E2',
           '3D6.1D593D7826', '3D6.1D593D785E', '3D6.1D593D783E',
           '3D6.1D593D7828', '3D6.1D593D7858', '3D6.1D593D7868',
           '3D6.1D593D7877', '3D6.1D593D783F', '3D6.1D593D7831',
           '3D6.1D593D7872', '3D6.1D593D786F', '3D6.1D593D7853',
           '3D6.1D593D782C', '3D6.1D593D787D', '3D6.00184967AD',
           '3D6.00184967AF', '3D6.00184967A7', '3D6.00184967BC',
           '3D6.00184967E6', '3D6.00184967E7', '3D6.00184967BA',
           '3D6.00184967A2', '3D6.00184967A6', '3D6.00184967F1',
           '3D6.00184967BF', '3D6.00184967E0', '3D6.00184967B8',
           '3D6.00184967ED', '3D6.00184967D1', '3D6.00184967F7',
           '3D6.00184967B7', '3D6.00184967D5', '3D6.00184967AC',
           '3D6.00184967C8', '3D6.00184967E3', '3D6.1D593D7848',
           '3D6.00184967C1', '3D6.1D593D782B', '3D6.00184967F8',
           '3D6.1D593D787F', '3D6.00184967E9', '3D6.00184967D4',
           '3D6.00184967CE', '3D6.1D593D45E9', '3D6.001881F761',
           '3D6.1D593D45D7', '3D6.1D593D45FA', '3D6.1D593D45CB',
           '3D6.001881F743', '3D6.1D593D45FF', '3D6.001881F780',
           '3D6.001881F777', '3D6.1D593D45D0', '3D6.1D593D460B',
           '3D6.1D593D45DA', '3D6.001881F776', '3D6.1D593D45F7',
           '3D6.001881F736', '3D6.1D593D45B6', '3D6.1D593D4611',
           '3D6.001881F733', '3D6.1D593D45F1', '3D6.001881F782',
           '3D6.1D593D4603', '3D6.1D593D45C9', '3D6.1D593D45F2',
           '3D6.1D593D7827', '3D6.1D593D7840', '3D6.1D593D45CF',
           '3D6.1D593D7843', '3D6.1D593D4612', '3D6.1D593D45D4',
           '3D6.1D593D45E4', '3D6.1D593D4609', '3D6.001881F758',
           '3D6.001881F76C', '3D6.001881F783', '3D6.001881F784',
           '3D6.1D593D7835', '3D6.1D593D45F9', '3D6.1D593D45EB',
           '3D6.1D593D45E6', '3D6.1D593D4614', '3D6.1D593D7839',
           '3D6.1D593D45EF', '3D6.001881F78A', '3D6.1D593D45E5',
           '3D6.1D593D45DF', '3D6.1D593D4604', '3D6.1D593D4607',
           '3D6.1D593D45B3', '3D6.1D593D45E1', '3D6.1D593D4606',
           '3D6.1D593D460E', '3D6.1D593D45DB', '3D6.1D593D4610',
           '3D6.1D593D45D1', '3D6.1D593D45D3', '3D6.1D593D4601',
           '3D6.001881F730', '3D6.1D593D460F', '3D6.001881F73A',
           '3D6.001881F788', '3D6.1D593D45FB', '3D6.1D593D460A',
           '3D6.001881F775', '3D6.1D593D45CE', '3D6.1D593D45C7',
           '3D6.1D593D7838', '3D6.1D593D7880', '3D6.1D593D460D',
           '3D6.1D593D45F3', '3D6.1D593D7845', '3D6.1D593D45E7',
           '3D6.1D593D787A', '3D6.1D593D45E2', '3D6.1D593D45D2',
           '3D6.1D593D786B', '3D6.1D593D45B9'], dtype=object)




```python
len(tag_data.ID.unique())
```




    155



### collapse_reads_10_seconds

this function will

    1. for each individual bird, it will calculate different between each reads
    2. if the difference is more that 0.11 seconds it will identify it as a new visit.
    3. for each new visit, it will identify the starting time, ending time
    


```python
visit_data = HT.collapse_reads_10_seconds(data= tag_data)
```


```python
visit_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>visit_start</th>
      <th>visit_end</th>
      <th>ID</th>
      <th>Tag</th>
      <th>visit_duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-09-25 05:53:00</td>
      <td>2016-09-25 05:53:00</td>
      <td>3D6.00184967A1</td>
      <td>A1</td>
      <td>0 days</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-10-26 13:32:31</td>
      <td>2016-10-26 13:32:31</td>
      <td>3D6.00184967A1</td>
      <td>A1</td>
      <td>0 days</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-10-26 13:33:34</td>
      <td>2016-10-26 13:33:34</td>
      <td>3D6.00184967A1</td>
      <td>A1</td>
      <td>0 days</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-10-26 14:38:35</td>
      <td>2016-10-26 14:38:35</td>
      <td>3D6.00184967A1</td>
      <td>A1</td>
      <td>0 days</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-10-26 15:10:45</td>
      <td>2016-10-26 15:10:45</td>
      <td>3D6.00184967A1</td>
      <td>A1</td>
      <td>0 days</td>
    </tr>
  </tbody>
</table>
</div>




```python
#visit_data.to_csv(output_path +'\Collapsed_visits.csv')
#visit_data.to_csv(team_path +'\Collapsed_visits.csv')
```

number of visits recorded


```python
print ('number of visits recorded during study period= '+str(visit_data.shape[0]))
```

    number of visits recorded during study period= 65476
    

### read metadata file

this function will

    1. read the master metadata file for all the birds PIT tagged until now
    2. filter the data according to the study location 


```python
meta = HT.read_metadata(data_path = data_path+"/Paper" , filename = 'PIT tagged Birds Info_HHCP_For manuscript_04_16_2018.xlsx', 
                       restrict = True)
meta = meta.drop_duplicates('Tag Hex', keep= 'first')
meta['Location Id'].unique()
## removing data for birds tagged in another site not in this study
remove_glide = list(set(meta[meta['Location Id'] == 'GR']['Tag Hex'].unique().tolist()).difference(set(visit_data.ID.unique().tolist())))
meta = meta[~meta['Tag Hex'].isin(remove_glide)]
meta.Species.unique()
meta.shape
```




    (230, 12)




```python
meta.Species.unique()
```




    array([u'ANHU', u'ALHU'], dtype=object)




```python
meta.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SrNo.</th>
      <th>Tagged Date</th>
      <th>Location Id</th>
      <th>Tag Hex</th>
      <th>Tag Dec</th>
      <th>Species</th>
      <th>Age</th>
      <th>Sex</th>
      <th>Band Number</th>
      <th>Location</th>
      <th>Notes</th>
      <th>True Unknowns</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2016-09-02</td>
      <td>SB</td>
      <td>3D6.00184967FF</td>
      <td>982.000407</td>
      <td>ANHU</td>
      <td>HY</td>
      <td>F</td>
      <td>NaN</td>
      <td>Manfreds</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2016-09-02</td>
      <td>SB</td>
      <td>3D6.00184967FC</td>
      <td>982.000407</td>
      <td>ANHU</td>
      <td>HY</td>
      <td>F</td>
      <td>NaN</td>
      <td>Manfreds</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2016-09-02</td>
      <td>SB</td>
      <td>3D6.00184967B0</td>
      <td>982.000407</td>
      <td>ANHU</td>
      <td>AHY</td>
      <td>M</td>
      <td>NaN</td>
      <td>Manfreds</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2016-09-02</td>
      <td>SB</td>
      <td>3D6.00184967F3</td>
      <td>982.000407</td>
      <td>ANHU</td>
      <td>HY</td>
      <td>M</td>
      <td>NaN</td>
      <td>Manfreds</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2016-09-02</td>
      <td>SB</td>
      <td>3D6.00184967DF</td>
      <td>982.000407</td>
      <td>ANHU</td>
      <td>AHY</td>
      <td>F</td>
      <td>NaN</td>
      <td>Manfreds</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### merge TAG data and Metadata

this function will
    1. merge the tag read data with the metadata using the 'Tag Hex' column
    2. Merge the data only after running **collapse_reads_10_seconds** function 
    3. identifies birds which were tagged and found on the same day in 2016


```python
data = HT.merge_metadata(data = visit_data, metadata = meta, for_paper = False)
```

    3D6.00184967A3 date same, first recorded 2016-09-23 08:56:09  tagged on 2016-09-23
    3D6.00184967AE date same, first recorded 2016-09-09 10:23:19  tagged on 2016-09-09
    3D6.00184967B5 date same, first recorded 2016-09-23 08:54:49  tagged on 2016-09-23
    3D6.00184967C6 date same, first recorded 2016-09-09 09:55:46  tagged on 2016-09-09
    3D6.00184967D3 date same, first recorded 2016-09-23 08:52:58  tagged on 2016-09-23
    3D6.00184967D6 date same, first recorded 2016-09-23 08:05:57  tagged on 2016-09-23
    3D6.00184967D7 date same, first recorded 2016-09-23 08:54:00  tagged on 2016-09-23
    3D6.00184967DB date same, first recorded 2016-10-21 08:20:40  tagged on 2016-10-21
    


```python
a = meta['Tag Hex'].unique().tolist()
data = data[data['ID'].isin(a)]
```


```python
meta.Location.unique()
```




    array([u'Manfreds', u'Arboretum', u'Glide Ranch', u"Susan's"],
          dtype=object)




```python
data['Location'].value_counts()
```




    Arboretum      25210
    Susan's        18478
    Manfreds       14609
    Glide Ranch      753
    Name: Location, dtype: int64




```python
data.Species.unique()
```




    array([u'ANHU', u'ALHU'], dtype=object)




```python
data.Species.value_counts(normalize=True)
```




    ANHU    0.908535
    ALHU    0.091465
    Name: Species, dtype: float64




```python
Bird_summary, reader_predilection = HT.bird_summaries(data = data,
                                 output_path = output_path,
                                 metadata = meta)
```


```python
Bird_summary[['obser_period', 'first_obs_aft_tag', 'date_u']].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>obser_period</th>
      <th>first_obs_aft_tag</th>
      <th>date_u</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>141</td>
      <td>141</td>
      <td>141.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>88 days 19:44:40.851063</td>
      <td>31 days 10:33:11.489361</td>
      <td>24.581560</td>
    </tr>
    <tr>
      <th>std</th>
      <td>110 days 01:02:39.808943</td>
      <td>58 days 15:34:02.081666</td>
      <td>44.338174</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0 days 00:00:00</td>
      <td>0 days 00:00:00</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>16 days 00:00:00</td>
      <td>0 days 00:00:00</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>34 days 00:00:00</td>
      <td>5 days 00:00:00</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>122 days 00:00:00</td>
      <td>32 days 00:00:00</td>
      <td>28.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>554 days 00:00:00</td>
      <td>444 days 00:00:00</td>
      <td>289.000000</td>
    </tr>
  </tbody>
</table>
</div>



# Table 1 


```python
pd.pivot_table(meta, index = 'Species', columns=['Sex', 'Age'], values='Tag Hex', aggfunc='count', margins= True, fill_value=0)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>Sex</th>
      <th colspan="3" halign="left">F</th>
      <th colspan="3" halign="left">M</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Age</th>
      <th>AHY</th>
      <th>HY</th>
      <th>UNK</th>
      <th>AHY</th>
      <th>HY</th>
      <th>UNK</th>
      <th></th>
    </tr>
    <tr>
      <th>Species</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ALHU</th>
      <td>6</td>
      <td>0</td>
      <td>8</td>
      <td>26</td>
      <td>2</td>
      <td>21</td>
      <td>63</td>
    </tr>
    <tr>
      <th>ANHU</th>
      <td>27</td>
      <td>23</td>
      <td>18</td>
      <td>39</td>
      <td>40</td>
      <td>20</td>
      <td>167</td>
    </tr>
    <tr>
      <th>All</th>
      <td>33</td>
      <td>23</td>
      <td>26</td>
      <td>65</td>
      <td>42</td>
      <td>41</td>
      <td>230</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.pivot_table(meta, index = 'Species', columns='Age', values='Tag Hex', aggfunc='count', margins= True, fill_value=0)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Age</th>
      <th>AHY</th>
      <th>HY</th>
      <th>UNK</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Species</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ALHU</th>
      <td>32</td>
      <td>2</td>
      <td>29</td>
      <td>63</td>
    </tr>
    <tr>
      <th>ANHU</th>
      <td>66</td>
      <td>63</td>
      <td>38</td>
      <td>167</td>
    </tr>
    <tr>
      <th>All</th>
      <td>98</td>
      <td>65</td>
      <td>67</td>
      <td>230</td>
    </tr>
  </tbody>
</table>
</div>




```python
table1 = HT.report_Table1A(metadata = meta, location= ['SB', 'Arbo -1 ','BH', 'GR'])
table1.to_csv(output_path+'/table1.csv')
table1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Species</th>
      <th>Allen's Hummingbird</th>
      <th>Anna's Hummingbird</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Sex</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>female</th>
      <td>14</td>
      <td>68</td>
      <td>82</td>
    </tr>
    <tr>
      <th>male</th>
      <td>49</td>
      <td>99</td>
      <td>148</td>
    </tr>
    <tr>
      <th>All</th>
      <td>63</td>
      <td>167</td>
      <td>230</td>
    </tr>
  </tbody>
</table>
</div>



# Table 2


```python
table2 = HT.report_Table2Paper(metadata = meta, location= ['SB', 'Arbo -1 ','BH', 'GR'] )
table2.to_csv(output_path+'/table2.csv')
table2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Species</th>
      <th>Allen's Hummingbird</th>
      <th>Anna's Hummingbird</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Location Id</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Arbo -1</th>
      <td>0</td>
      <td>36</td>
      <td>36</td>
    </tr>
    <tr>
      <th>BH</th>
      <td>63</td>
      <td>78</td>
      <td>141</td>
    </tr>
    <tr>
      <th>GR</th>
      <td>0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>SB</th>
      <td>0</td>
      <td>50</td>
      <td>50</td>
    </tr>
    <tr>
      <th>All</th>
      <td>63</td>
      <td>167</td>
      <td>230</td>
    </tr>
  </tbody>
</table>
</div>



# Table 4


```python
table4 = HT.paper_Table2(data = data, metadata = meta, location= ['SB', 'Arbo -1 ','BH', 'GR'])
table4.to_csv(output_path+'/table4.csv')
table4
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>Sex</th>
      <th colspan="3" halign="left">female</th>
      <th colspan="3" halign="left">male</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Age</th>
      <th>after hatch year</th>
      <th>hatch year</th>
      <th>unknown</th>
      <th>after hatch year</th>
      <th>hatch year</th>
      <th>unknown</th>
      <th></th>
    </tr>
    <tr>
      <th>Species</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Allen's Hummingbird</th>
      <td>3</td>
      <td>0</td>
      <td>6</td>
      <td>12</td>
      <td>1</td>
      <td>12</td>
      <td>34</td>
    </tr>
    <tr>
      <th>Anna's Hummingbird</th>
      <td>18</td>
      <td>17</td>
      <td>6</td>
      <td>28</td>
      <td>25</td>
      <td>13</td>
      <td>107</td>
    </tr>
    <tr>
      <th>All</th>
      <td>21</td>
      <td>17</td>
      <td>12</td>
      <td>40</td>
      <td>26</td>
      <td>25</td>
      <td>141</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.pivot_table(data, columns='Sex', index='Species', aggfunc='nunique', values='ID')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Sex</th>
      <th>F</th>
      <th>M</th>
    </tr>
    <tr>
      <th>Species</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ALHU</th>
      <td>9</td>
      <td>25</td>
    </tr>
    <tr>
      <th>ANHU</th>
      <td>41</td>
      <td>66</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.pivot_table(data, columns='Age', index='Species', aggfunc='nunique', values='ID')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Age</th>
      <th>AHY</th>
      <th>HY</th>
      <th>UNK</th>
    </tr>
    <tr>
      <th>Species</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ALHU</th>
      <td>15</td>
      <td>1</td>
      <td>18</td>
    </tr>
    <tr>
      <th>ANHU</th>
      <td>46</td>
      <td>42</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>



# Figure 5: 

### Daily visitations of PIT- tagged male and female Anna’s Hummingbirds at the feeding stations at Sites 1 and 2 in Northern California and Site 3 in Southern California from September 2016 to March 2018. 


```python
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
data.Sex.fillna('unknown', inplace= True)
alhu_data = data[data.Species == "ALHU"]
anhu_data = data[data.Species == "ANHU"]
plt.rcParams['font.size'] = 13
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
plt.rcParams['axes.titlesize'] = 1.5*plt.rcParams['font.size']
plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']+1
plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']

def plotvisits(timeunit, data, ax, c = ['#e41a1c', '#377eb8', '#4daf4a'], style = '-'):
    """timeunite: 10Min, D =  day, W = week, M = month"""
    if timeunit == '10Min':
        t = '10 minutes'
    elif timeunit == 'D':
        t = 'day'
    elif timeunit == 'W':
        t = 'week'
    if timeunit == 'M':
        t = 'month'
    a = data.groupby([ pd.Grouper(freq=timeunit), 'Sex'])['ID'].count().unstack('Sex')
    a.plot(kind="line", style = style, rot =45, ax= ax,stacked=False, color = c,)
    
    #set ticks every month
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    #set major ticks format
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    ## hiding every second month
    for index, label in enumerate(ax.xaxis.get_ticklabels()):
        if index % 3 != 0:
            label.set_visible(False)
    
    #ax.set_title('visits per '+t)
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of visits per day')

f, (ax1)  = plt.subplots(1, 1, figsize=(8,6))
plotvisits(timeunit= 'D', data = anhu_data, ax= ax1)

f.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(output_path + '\Figure4.png', dpi = dpi)
plt.savefig(output_path + '\Figure4.eps',format = 'eps', dpi = dpi)
plt.show()
```


![png](output_46_0.png)



```python
[item.get_text() for item in ax1.get_xticklabels()]
```




    [u'Sep 2016',
     u'Oct 2016',
     u'Nov 2016',
     u'Dec 2016',
     u'Jan 2017',
     u'Feb 2017',
     u'Mar 2017',
     u'Apr 2017',
     u'May 2017',
     u'Jun 2017',
     u'Jul 2017',
     u'Aug 2017',
     u'Sep 2017',
     u'Oct 2017',
     u'Nov 2017',
     u'Dec 2017',
     u'Jan 2018',
     u'Feb 2018',
     u'Mar 2018',
     u'Apr 2018']




```python
anhu_data.index
```




    DatetimeIndex(['2016-09-25 05:53:00', '2016-10-26 13:32:31',
                   '2016-10-26 13:33:34', '2016-10-26 14:38:35',
                   '2016-10-26 15:10:45', '2016-10-26 15:46:14',
                   '2016-10-26 16:12:54', '2016-10-27 10:38:53',
                   '2016-10-28 09:12:40', '2016-10-29 10:53:49',
                   ...
                   '2018-03-06 07:31:05', '2018-03-06 07:31:20',
                   '2018-03-06 10:35:45', '2018-03-06 10:36:07',
                   '2018-02-27 16:45:14', '2018-03-06 16:03:50',
                   '2018-02-28 13:10:57', '2018-02-28 13:12:39',
                   '2018-02-28 13:13:00', '2018-02-28 13:13:14'],
                  dtype='datetime64[ns]', name=u'visit_start', length=53649, freq=None)




```python
data.shape
```




    (59050, 19)



### Tagging activity Figure 5 Inset: The cumulative tagging effort throughout the study.


```python
import matplotlib.dates as mdates
fig, ax1 = plt.subplots(1, figsize=(6,4))
idx = pd.date_range('06.01.2016', '02.28.2018')
a = meta.groupby('Tagged Date')['Tag Hex'].nunique()
a.index = pd.DatetimeIndex(a.index)
a = a.reindex(idx, fill_value=0)
#tl = a.cumsum()
#tl.columns = ['sampling']
#tl.plot.line(drawstyle = 'steps', label='Animals', rot =45,ax = ax1)
a.resample('D').cumsum().plot(kind="line", rot =45,ax = ax1)
#set ticks every month
#ax1.xaxis.set_major_locator(mdates.MonthLocator())
#set major ticks format
#ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

## hiding every second month
#for index, label in enumerate(ax1.xaxis.get_ticklabels()):
#    if index % 3 != 0:
#        label.set_visible(False)

ax1.set_ylabel('Individuals tagged', fontsize = 11)
ax1.set_xlabel('Time', fontsize = 11)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(output_path+'/'+'cumulative_tagging.png', dpi = dpi)
plt.savefig(output_path+'/'+'cumulative_tagging.eps',format = 'eps',  dpi = dpi)
plt.show()
```

    C:\Users\Falco\Anaconda2\lib\site-packages\ipykernel_launcher.py:10: FutureWarning: 
    .resample() is now a deferred operation
    You called cumsum(...) on this deferred object which materialized it into a series
    by implicitly taking the mean.  Use .resample(...).mean() instead
      # Remove the CWD from sys.path while we load stuff.
    


![png](output_51_1.png)


# Figure 6:

### Comparison of the daily detections from the two different antennas at the double antenna feeding station  transceiver at Site 2 over time. The top antenna for this feeding station was deployed in May 2017 and discontinued in January 2018 for data collection. Overall, the total number of visits detected by the side antenna exceeded the number of visits for the top antenna but not enough to warrant the addition of a second antenna. Both antennas detected the presence of the same number of individual birds.


```python
## Top reader
tag_dataA2 = HT.read_tag_files(data_path = data_path+'\Paper', select_readers = ['A2'], )
visit_dataA2 = HT.collapse_reads_10_seconds(data= tag_dataA2)
dataA2 = HT.merge_metadata(data = visit_dataA2, metadata = meta, for_paper = True)

## Side reader
tag_dataA4 = HT.read_tag_files(data_path = data_path+'\Paper', select_readers = ['A4'])
visit_dataA4 = HT.collapse_reads_10_seconds(data= tag_dataA4)
dataA4 = HT.merge_metadata(data = visit_dataA4, metadata = meta, for_paper = True)

dataA4 = dataA4.ix['2017-05-23':'2018-01-25']
dataA2 = dataA2.ix['2017-05-23':'2018-01-25']
```

    C:\Users\Falco\Anaconda2\lib\site-packages\ipykernel_launcher.py:11: DeprecationWarning: 
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing
    
    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\Falco\Anaconda2\lib\site-packages\ipykernel_launcher.py:12: DeprecationWarning: 
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing
    
    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated
      if sys.path[0] == '':
    


```python
print ('side antenna recorded ' +str(dataA4.shape[0]) + ' visits')
```

    side antenna recorded 7825 visits
    


```python
print ('top antenna recorded ' +str(dataA2.shape[0]) + ' visits')
```

    top antenna recorded 7620 visits
    


```python
print ('side antenna recorded ' +str(len(dataA4.ID.unique())) + ' PIT tagged birds')
```

    side antenna recorded 20 PIT tagged birds
    


```python
print ('top antenna recorded ' +str(len(dataA2.ID.unique())) + ' PIT tagged birds')
```

    top antenna recorded 20 PIT tagged birds
    

# Figure 6


```python
dpi = 1000
```


```python
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
        
        #set ticks every month
        #ax.xaxis.set_major_locator(mdates.MonthLocator())
        #set major ticks format
        #ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y %B'))

        ## hiding every second month
        #for index, label in enumerate(ax.xaxis.get_ticklabels()):
        #    if index % 3 != 0:
        #        label.set_visible(False)

        #ax.set_title('visits per '+t)
        ax.set_xlabel('Time')
        ax.set_ylabel('Number of visits per day')

                
f, (ax1)  = plt.subplots(1, 1, figsize=(8,6))
plotvisits(timeunit= 'D', data= dataA4, ax= ax1)
plotvisits(timeunit= 'D', data= dataA2, ax= ax1)
labels = ['Jun 2017', 'Jul 2017', 'Aug 2017','Sep 2017', 'Oct 2017', 'Nov 2017', 'Dec 2017', 'Jan 2018', 'Feb 2018']
ax1.set_xticklabels(labels)
#set ticks every month
#ax1.xaxis.set_major_locator(mdates.MonthLocator())
#set major ticks format
#ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y %B'))

## hiding every second month
#for index, label in enumerate(ax1.xaxis.get_ticklabels()):
#    if index % 3 != 0:
#        label.set_visible(False)
f.legend(['side antenna reader unit', 'top antenna reader unit'],loc = 'best')
#f.suptitle('Hummingbird visits to feeders', fontsize = 24)
f.tight_layout()#rect=[0, 0.03, 1, 0.95]
f.savefig(output_path+'/Figure 5.png', dpi = dpi)
f.savefig(output_path+'/'+'Figure 5.eps',format = 'eps',  dpi = dpi)
plt.show()
```

    C:\Users\Falco\Anaconda2\lib\site-packages\matplotlib\legend.py:649: UserWarning: Automatic legend placement (loc="best") not implemented for figure legend. Falling back on "upper right".
      warnings.warn('Automatic legend placement (loc="best") not '
    


![png](output_60_1.png)


## Gender association with activity of bird visitation metrices


```python
v = pd.merge(visit_data, meta, left_on='ID', right_on='Tag Hex', how='left')
sns.violinplot(v.Sex,np.log((v.visit_duration / np.timedelta64(1, 's')).astype(int)+0.0001))
plt.ylabel('visit duration (log)')
plt.show()
```


![png](output_62_0.png)



```python
data.visit_duration.describe()
```




    count                     59050
    mean     0 days 00:00:07.291244
    std      0 days 00:00:17.425316
    min             0 days 00:00:00
    25%             0 days 00:00:00
    50%             0 days 00:00:00
    75%             0 days 00:00:10
    max             0 days 00:10:15
    Name: visit_duration, dtype: object




```python
Bird_summary.duration_total.describe()
```




    count      141.000000
    mean      3053.531915
    std       8455.482026
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%       1331.000000
    max      59922.000000
    Name: duration_total, dtype: float64



# Short visits


```python
v_s = v[v.visit_duration =='0 seconds']
v_s.shape
```




    (48713, 17)




```python
v_s.visit_duration.sum()
```




    Timedelta('0 days 00:00:00')



## Long visits


```python
v_d = v[v.visit_duration>'0 seconds']
v_d.shape
```




    (16763, 17)



## Total time observed


```python
v_d.visit_duration.sum()
```




    Timedelta('5 days 05:01:30')




```python
sns.violinplot(v_d.Sex,np.log((v_d.visit_duration / np.timedelta64(1, 's')).astype(int)+0.0001), cut=0)
plt.ylabel('visit duration (log)')
plt.show()
```


![png](output_72_0.png)


## difference in the visit duration males vs females


```python
Bird_summary.Sex.unique()
```




    array([u'F', u'M'], dtype=object)




```python
v_d.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>visit_start</th>
      <th>visit_end</th>
      <th>ID</th>
      <th>Tag</th>
      <th>visit_duration</th>
      <th>SrNo.</th>
      <th>Tagged Date</th>
      <th>Location Id</th>
      <th>Tag Hex</th>
      <th>Tag Dec</th>
      <th>Species</th>
      <th>Age</th>
      <th>Sex</th>
      <th>Band Number</th>
      <th>Location</th>
      <th>Notes</th>
      <th>True Unknowns</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>274</th>
      <td>2017-10-10 10:06:29</td>
      <td>2017-10-10 10:06:40</td>
      <td>3D6.00184967A1</td>
      <td>A1</td>
      <td>00:00:11</td>
      <td>16.0</td>
      <td>2016-09-23</td>
      <td>SB</td>
      <td>3D6.00184967A1</td>
      <td>982.000407</td>
      <td>ANHU</td>
      <td>HY</td>
      <td>F</td>
      <td>NaN</td>
      <td>Manfreds</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>278</th>
      <td>2017-10-10 10:08:17</td>
      <td>2017-10-10 10:08:27</td>
      <td>3D6.00184967A1</td>
      <td>A1</td>
      <td>00:00:10</td>
      <td>16.0</td>
      <td>2016-09-23</td>
      <td>SB</td>
      <td>3D6.00184967A1</td>
      <td>982.000407</td>
      <td>ANHU</td>
      <td>HY</td>
      <td>F</td>
      <td>NaN</td>
      <td>Manfreds</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>280</th>
      <td>2017-10-10 11:15:46</td>
      <td>2017-10-10 11:15:56</td>
      <td>3D6.00184967A1</td>
      <td>A1</td>
      <td>00:00:10</td>
      <td>16.0</td>
      <td>2016-09-23</td>
      <td>SB</td>
      <td>3D6.00184967A1</td>
      <td>982.000407</td>
      <td>ANHU</td>
      <td>HY</td>
      <td>F</td>
      <td>NaN</td>
      <td>Manfreds</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>282</th>
      <td>2017-10-10 11:19:53</td>
      <td>2017-10-10 11:20:25</td>
      <td>3D6.00184967A1</td>
      <td>A1</td>
      <td>00:00:32</td>
      <td>16.0</td>
      <td>2016-09-23</td>
      <td>SB</td>
      <td>3D6.00184967A1</td>
      <td>982.000407</td>
      <td>ANHU</td>
      <td>HY</td>
      <td>F</td>
      <td>NaN</td>
      <td>Manfreds</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>284</th>
      <td>2017-10-10 11:46:45</td>
      <td>2017-10-10 11:46:56</td>
      <td>3D6.00184967A1</td>
      <td>A1</td>
      <td>00:00:11</td>
      <td>16.0</td>
      <td>2016-09-23</td>
      <td>SB</td>
      <td>3D6.00184967A1</td>
      <td>982.000407</td>
      <td>ANHU</td>
      <td>HY</td>
      <td>F</td>
      <td>NaN</td>
      <td>Manfreds</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
v_d['visit_duration_seconds'] = v_d['visit_duration'].dt.total_seconds()
trial = v_d.groupby(['ID', 'Sex']).visit_duration_seconds.mean()
trial = pd.DataFrame(trial)
trial.columns = ['duration_mean']
trial.reset_index(inplace= True)
trial.head()
```

    C:\Users\Falco\Anaconda2\lib\site-packages\ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """Entry point for launching an IPython kernel.
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Sex</th>
      <th>duration_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3D6.00184967A1</td>
      <td>F</td>
      <td>43.838710</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3D6.00184967A2</td>
      <td>F</td>
      <td>40.178049</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3D6.00184967A4</td>
      <td>M</td>
      <td>27.726704</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3D6.00184967A6</td>
      <td>M</td>
      <td>21.397590</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3D6.00184967A7</td>
      <td>F</td>
      <td>21.530667</td>
    </tr>
  </tbody>
</table>
</div>




```python
m_v_d1 = trial[trial.Sex== 'M']
m_v_d = m_v_d1.duration_mean #/ np.timedelta64(1, 's')
f_v_d1 = trial[trial.Sex== 'F']
f_v_d = f_v_d1.duration_mean #/ np.timedelta64(1, 's')
```


```python
print (np.mean(f_v_d), np.std(f_v_d), len(f_v_d) )
```

    (25.110059470897557, 11.969362949243743, 25)
    


```python
print (np.mean(m_v_d), np.std(m_v_d), len(m_v_d) )
```

    (23.43185166728472, 11.273722727146934, 45)
    


```python
stats.kruskal(m_v_d, f_v_d)
```




    KruskalResult(statistic=0.25264184738533463, pvalue=0.6152209840639247)



##### Comparison of median


```python
stats.median_test(f_v_d, m_v_d)
```




    (0.24888888888888888, 0.6178585337664037, 21.76533333333333, array([[14, 21],
            [11, 24]], dtype=int64))




```python
print 'male median visit time '+ str( np.median(m_v_d))
print 'female median visit time '+ str( np.median(f_v_d))
```

    male median visit time 21.233055885850177
    female median visit time 22.832512315270936
    


```python
print 'male mean visit time '+ str( np.mean(m_v_d))
print 'female means visit time '+ str( np.mean(f_v_d))
```

    male mean visit time 23.4318516673
    female means visit time 25.1100594709
    


```python
print 'male std visit time '+ str( np.std(m_v_d))
print 'female std visit time '+ str( np.std(f_v_d))
```

    male std visit time 11.2737227271
    female std visit time 11.9693629492
    


```python
print 'male std visit time '+ str( m_v_d.shape)
print 'female std visit time '+ str( f_v_d.shape)
```

    male std visit time (45L,)
    female std visit time (25L,)
    

### Proportoin of time present


```python
Bird_summary['proportion_present'] = Bird_summary.duration_total/(Bird_summary.date_u*86400)
Bird_summary['proportion_present'].describe()
sns.boxplot(Bird_summary.Sex,Bird_summary.proportion_present)
plt.show()
```


![png](output_88_0.png)



```python
m_pt = Bird_summary[Bird_summary.Sex== 'M']
f_pt = Bird_summary[Bird_summary.Sex== 'F']
stats.mannwhitneyu(m_pt.proportion_present, f_pt.proportion_present)
```




    MannwhitneyuResult(statistic=2252.5, pvalue=0.45957163430402526)




```python
m_pt.proportion_present.describe()
```




    count    91.000000
    mean      0.000651
    std       0.001232
    min       0.000000
    25%       0.000000
    50%       0.000000
    75%       0.001028
    max       0.007655
    Name: proportion_present, dtype: float64




```python
f_pt.proportion_present.describe()
```




    count    50.000000
    mean      0.000731
    std       0.001580
    min       0.000000
    25%       0.000000
    50%       0.000006
    75%       0.000628
    max       0.007378
    Name: proportion_present, dtype: float64




```python
Bird_summary.proportion_present.describe()
```




    count    141.000000
    mean       0.000680
    std        0.001361
    min        0.000000
    25%        0.000000
    50%        0.000000
    75%        0.000874
    max        0.007655
    Name: proportion_present, dtype: float64



# Diel activity of Humminbirds

# Figure 7:

### Diel plots of mean hourly hummingbird visits to the feeding stations at three sites at northern and southern California from September 2016 to March 2018. The bar denotes the mean hourly visits by hummingbirds and black lines show the standard error. All distributions were statistically non-uniform (probability values reported by season).


```python
%%time
plt.rcParams['xtick.major.pad']='8'
HT.diurnal_variation(dataframe = data, output_path = output_path)
```

    C:\Users\Falco\Anaconda2\lib\site-packages\matplotlib\projections\polar.py:58: RuntimeWarning: invalid value encountered in less
      mask = r < 0
    


![png](output_95_1.png)


    Rayleigh test identify a non-uniform distribution, i.e. it is
    designed for detecting an unimodal deviation from uniformity
    winter: p =  0.000
    spring: p =  0.000
    summer: p =  0.000
    fall: p =  0.000
    Wall time: 20.5 s
    

### bird_summaries

This function:

1. calculates summaries for each tagged bird
2. save a csv file with summary output
3. columns are 

Tag Hex,Species,Sex, Age, tagged_date, first_obs_aft_tag, date_min, date_max, obser_period, date_u, Location

# Night time activity

## Birds that showed activity between 10 PM to 4 AM 


```python
visit_data.set_index(pd.DatetimeIndex(visit_data['visit_start']), inplace= True)
import datetime
start = datetime.time(22,0,0)
end = datetime.time(4,0,0)
visit_data.between_time(start, end).ID.unique().tolist()
```




    ['3D6.00184967A4',
     '3D6.00184967A7',
     '3D6.00184967AD',
     '3D6.00184967AF',
     '3D6.00184967BC',
     '3D6.00184967BF',
     '3D6.1D593D7848']



### number of visits in night by birds


```python
import numpy as np
n = visit_data.between_time(start, end)
n['date']= n.index.date
n.groupby(['ID', 'date']).visit_start.count()
```

    C:\Users\Falco\Anaconda2\lib\site-packages\ipykernel_launcher.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      This is separate from the ipykernel package so we can avoid doing imports until
    




    ID              date      
    3D6.00184967A4  2018-01-04     3
                    2018-01-06     1
    3D6.00184967A7  2017-06-19     1
    3D6.00184967AD  2017-05-22     2
                    2017-05-23    11
                    2017-05-24     7
                    2017-05-25     4
    3D6.00184967AF  2017-05-23     2
                    2017-05-24     3
                    2017-05-25     1
                    2018-01-05     3
                    2018-01-06     1
    3D6.00184967BC  2017-06-19     1
    3D6.00184967BF  2018-01-04     1
                    2018-02-10     1
    3D6.1D593D7848  2018-02-13     1
    Name: visit_start, dtype: int64




```python
n.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>visit_duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>43</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0 days 00:00:06.488372</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0 days 00:00:14.921017</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0 days 00:00:00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0 days 00:00:00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0 days 00:00:00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0 days 00:00:00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0 days 00:01:05</td>
    </tr>
  </tbody>
</table>
</div>




```python
date_format='%H:%M:%S'
```

## Time spent at the feeder in a night


```python
n.groupby(['ID', 'date']).visit_duration.sum()
```




    ID              date      
    3D6.00184967A4  2018-01-04   00:01:27
                    2018-01-06   00:00:00
    3D6.00184967A7  2017-06-19   00:00:43
    3D6.00184967AD  2017-05-22   00:00:00
                    2017-05-23   00:00:00
                    2017-05-24   00:00:00
                    2017-05-25   00:00:00
    3D6.00184967AF  2017-05-23   00:00:00
                    2017-05-24   00:00:00
                    2017-05-25   00:00:00
                    2018-01-05   00:01:14
                    2018-01-06   00:00:00
    3D6.00184967BC  2017-06-19   00:01:05
    3D6.00184967BF  2018-01-04   00:00:00
                    2018-02-10   00:00:10
    3D6.1D593D7848  2018-02-13   00:00:00
    Name: visit_duration, dtype: timedelta64[ns]



## feeders used in night


```python
n.groupby(['ID', 'date'])['Tag'].unique()
```




    ID              date      
    3D6.00184967A4  2018-01-04    [A4]
                    2018-01-06    [A4]
    3D6.00184967A7  2017-06-19    [A4]
    3D6.00184967AD  2017-05-22    [A4]
                    2017-05-23    [A4]
                    2017-05-24    [A4]
                    2017-05-25    [A4]
    3D6.00184967AF  2017-05-23    [A4]
                    2017-05-24    [A4]
                    2017-05-25    [A4]
                    2018-01-05    [A4]
                    2018-01-06    [A4]
    3D6.00184967BC  2017-06-19    [A4]
    3D6.00184967BF  2018-01-04    [A4]
                    2018-02-10    [A5]
    3D6.1D593D7848  2018-02-13    [A5]
    Name: Tag, dtype: object




```python
from matplotlib import dates as dates 
import datetime as datetime
from matplotlib.ticker import MaxNLocator
```

# Figure 8:

### Activity of an after hatch year male Anna’s hummingbird with a subcutaneously placed PIT at Site 2 between May 22nd and May 26th 2017. This bird was detected at the feeder after sunset, evident in the peaks of hourly visits during the night (green shaded sections of plot).


```python
f, (ax1)  = plt.subplots(1, 1, figsize=(8,6))
HT.plotBirdnight(data = visit_data, bird_ID='3D6.00184967AD', 
              start = '2017-05-22', 
              end ='2017-05-26',Bird_summary_data= Bird_summary,  ax = ax1)

#f.suptitle('Hummingbird visits to feeders', fontsize = 24)
f.tight_layout()#rect=[0, 0.03, 1, 0.95]
f.savefig(output_path+'/Figure 7.png', dpi = dpi)
f.savefig(output_path+'/Figure 7.eps',format = 'eps',  dpi=dpi)
plt.show()
```


![png](output_110_0.png)


# Figure S1:

### Activity of PIT tagged Anna’s Hummingbirds (Calypte anna) detected between 10 PM to 4 AM for the study period (September 2016- March 2018).  This nocturnal activity was only seen at study site 2 in northern California. The shaded portion represents night time. Days preceding and succeeding the night 


```python
fig, ((ax1,ax2,ax3),(ax4,ax5,ax6), (ax7,ax8,ax9))  = plt.subplots(nrows = 3, ncols = 3, figsize = (11,11))

HT.plotBirdnight(data = visit_data, bird_ID='3D6.00184967A4', 
              start = '2018-01-04', 
              end ='2018-01-07',Bird_summary_data= Bird_summary,  ax = ax1, title = True)

HT.plotBirdnight(data = visit_data, bird_ID='3D6.00184967A7', 
              start = '2017-06-18', 
              end ='2017-06-20',Bird_summary_data= Bird_summary,  ax = ax2,title = True)

HT.plotBirdnight(data = visit_data, bird_ID='3D6.00184967AD', 
              start = '2017-05-22', 
              end ='2017-05-26',Bird_summary_data= Bird_summary,  ax = ax3,title = True)

HT.plotBirdnight(data = visit_data, bird_ID='3D6.00184967AF', 
              start = '2017-05-22', 
              end ='2017-05-26',Bird_summary_data= Bird_summary,  ax = ax4,title = True)

HT.plotBirdnight(data = visit_data, bird_ID='3D6.00184967AF', 
              start = '2018-01-05', 
              end ='2018-01-07',Bird_summary_data= Bird_summary,  ax = ax5,title = True)

HT.plotBirdnight(data = visit_data, bird_ID='3D6.00184967BC', 
              start = '2017-06-18', 
              end ='2017-06-20',Bird_summary_data= Bird_summary,  ax = ax6,title = True)

HT.plotBirdnight(data = visit_data, bird_ID='3D6.00184967BF', 
              start = '2018-01-04', 
              end ='2018-01-06',Bird_summary_data= Bird_summary,  ax = ax7,title = True)

HT.plotBirdnight(data = visit_data, bird_ID='3D6.00184967BF', 
              start = '2018-02-09', 
              end ='2018-02-11',Bird_summary_data= Bird_summary,  ax = ax8,title = True)

HT.plotBirdnight(data = visit_data, bird_ID='3D6.1D593D7848', 
              start = '2018-02-12', 
              end ='2018-02-14',Bird_summary_data= Bird_summary,  ax = ax9,title = True)


plt.tight_layout()
plt.savefig(output_path+'/Figure S3.png', dpi = dpi)
plt.savefig(output_path+'/Figure S3.eps',format = 'eps',  dpi=dpi)
plt.show()
```


![png](output_112_0.png)


# Feeder preference

# Figure 9:

### The preference of individual Anna’s and Allen’s Hummingbirds for the primary, secondary, and tertiary feeders in Site 2 in Northern California. Solid circles show the data points.


```python
reader_predilection.columns =reader_predilection.columns.get_level_values(0)
t = reader_predilection.drop('A1', axis=1)
#t.set_index('ID',inplace= True)
primary = []
secondary = []
tertiary = []
bird = []
for index,row in t.iterrows():
    if row.values.max() !=0:
        r = sorted(row.values, reverse= True)
        total  = float(row.values.sum())
        bird.append(index)
        primary.append(r[0]/total)
        secondary.append(r[1]/total)
        tertiary.append(r[2]/total)

r_p = pd.DataFrame({'primary':primary,'secondary':secondary,'tertiary':tertiary}, index=bird)
r_p = r_p*100
###################################################################################################
###################################################################################################
#r_p.head()
t = t[(t.T != 0).any()]
t.idxmax(axis=1).value_counts()
###################################################################################################
###################################################################################################
primary = []
secondary = []
tertiary = []
bird = []
for index,row in t.iterrows():
    if row.values.max() !=0:
        r = sorted(row.values, reverse= True)
        total  = float(row.values.sum())
        bird.append(index)
        primary.append(r[0]/total)
        secondary.append(r[1]/total)
        tertiary.append(r[2]/total)
        
###################################################################################################
###################################################################################################

import seaborn as sns
f, (ax1)  = plt.subplots(1, 1, figsize=(8,6))
sns.boxplot(x="variable", y="value", data=r_p.melt(), ax = ax1)
ax = sns.swarmplot(x="variable", y="value", data=r_p.melt(), color=".25", ax = ax1)
ax1.set_xlabel('feeders')
ax1.set_ylabel('% feeder visits')
ax1.text(1, 85, r'Kruskal-Wallis, p<$2.2e^{-16}$')
f.tight_layout()#rect=[0, 0.03, 1, 0.95]
f.savefig(output_path+'/Figure 8.png', dpi = dpi)
f.savefig(output_path+'/Figure 8.eps',format = 'eps',  dpi=dpi)
plt.show()
```


![png](output_115_0.png)



```python
r_p.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>primary</th>
      <th>secondary</th>
      <th>tertiary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>109.000000</td>
      <td>109.000000</td>
      <td>109.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>86.674542</td>
      <td>10.809845</td>
      <td>2.515613</td>
    </tr>
    <tr>
      <th>std</th>
      <td>19.231371</td>
      <td>15.635133</td>
      <td>6.478267</td>
    </tr>
    <tr>
      <th>min</th>
      <td>33.333333</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>73.317308</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>100.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>100.000000</td>
      <td>21.875000</td>
      <td>0.144509</td>
    </tr>
    <tr>
      <th>max</th>
      <td>100.000000</td>
      <td>50.000000</td>
      <td>33.333333</td>
    </tr>
  </tbody>
</table>
</div>



## Interactions contact network


```python
interactions = Hxnet.get_intreactions(data=data)
interactions.shape
```




    (1635, 28)




```python
print ('there were '+ str(interactions.shape[0])+' interactions observed during the study period')
```

    there were 1635 interactions observed during the study period
    


```python
Hnet = Hxnet.get_interaction_networks(network_name='hummingbirds', data = data, 
                                      interactions = interactions, 
                                      location = output_path)   
```


![png](output_120_0.png)



```python
import networkx as nx
nx.write_graphml(Hnet, output_path +'/'+ "hummingbirds_interaction.graphml")
```


```python
from taggit import interactions as Hxnet
```


```python
a = data.groupby(['ID', 'Sex', 'Age', 'Location', 'Species']).size().reset_index().rename(columns={0:'count'})
```


```python
from fa2 import ForceAtlas2
forceatlas2 = ForceAtlas2(
                          # Behavior alternatives
                          outboundAttractionDistribution=False,  # Dissuade hubs
                          linLogMode=False,  # NOT IMPLEMENTED
                          adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                          edgeWeightInfluence=0.5,

                          # Performance
                          jitterTolerance=1.0,  # Tolerance
                          barnesHutOptimize=True,
                          barnesHutTheta=1.2,
                          multiThreaded=False,  # NOT IMPLEMENTED

                          # Tuning
                          scalingRatio=1.0,
                          strongGravityMode=False,
                          gravity=5.0,

                          # Log
                          verbose=True)
```

# Figure 10: (Manuscript has a figure generated using Gephi)

### Contact network of tagged Anna’s (pink nodes) and Allen’s hummingbirds (green nodes). The sex of the hummingbirds indicated by M for males and F for females. The size of the node is proportional to the degree of the interaction. Edges represent time spent together at the feeding station and the width of the edge width is proportional to the time spent together.


```python
color_map = {"ANHU":'#e41a1c', "ALHU":'#377eb8'} 

plt.figure(figsize=(25,25))
options = {
    'edge_color': '#999994',
    'font_weight': 'regular',
    'label': True,
    'alpha': 0.8
}
colors = [color_map[Hnet.node[node]['Species']] for node in Hnet]
#sizes = [G.node[node]['nodesize']*10 for node in G]
w = nx.get_edge_attributes(Hnet,'weight').values()
w2 = [x -1 for x in w]
s = [1000+(10000*np.log2(x+1))+(np.log2(np.log2(x+1)+1)*10000) for x in nx.betweenness_centrality(Hnet,).values()]
l = nx.get_node_attributes(Hnet, 'Sex').values()
mapping=nx.get_node_attributes(Hnet, 'Sex')
#Hnet=nx.relabel_nodes(Hnet,mapping)
"""
Using the spring layout : 
- k controls the distance between the nodes and varies between 0 and 1
- iterations is the number of times simulated annealing is run
default k=0.1 and iterations=50
"""

positions = forceatlas2.forceatlas2_networkx_layout(Hnet, pos=None, iterations=2000)

nx.draw(Hnet, node_color=colors, node_size=s, pos=positions, width = w,  **options)
nx.draw_networkx_labels(Hnet,positions,mapping,font_size=24)
ax = plt.gca()
ax.collections[0].set_edgecolor("#555555") 
plt.savefig(output_path+'/Network_diagram.png', dpi = dpi)
plt.savefig(output_path+'/Network_diagram.eps', format = 'eps', dpi = dpi)
plt.show()
```

    100%|████████████████████████████████████████████████████████████████████████████| 2000/2000 [00:00<00:00, 2280.50it/s]
    

    ('BarnesHut Approximation', ' took ', '0.19', ' seconds')
    ('Repulsion forces', ' took ', '0.51', ' seconds')
    ('Gravitational forces', ' took ', '0.01', ' seconds')
    ('Attraction forces', ' took ', '0.06', ' seconds')
    ('AdjustSpeedAndApplyForces step', ' took ', '0.06', ' seconds')
    


![png](output_126_2.png)


## Analysis for comparing degree distrbutions

# Figure S2:

### Results of permutation-based regression analysis to understand the effect of age and sex on the degree of in the observed network. Blue lines show the distribution of coefficients after 10,000 permutations. Red lines show original coefficients.


```python
%%time
Hxnet.run_permutation_test(dependent='degree', network= Hnet, number_of_permutations = 10000, output_path = output_path)
```


![png](output_129_0.png)


    Wall time: 1min 22s
    

# Figure S3:

### Results of permutation-based regression analysis to understand the effect of age and sex on the betweenness centrality of in the observed network. Blue lines show the distribution of coefficients after 10,000 permutations. Red lines show original coefficients.


```python
%%time
Hxnet.run_permutation_test(dependent='centrality', network= Hnet, number_of_permutations = 10000, output_path = output_path)
```


![png](output_131_0.png)


    Wall time: 1min 30s
    


```python
males = zip(*filter(lambda (n, d): d['Sex'] == 'M', Hnet.nodes(data=True)))[0]
females = zip(*filter(lambda (n, d): d['Sex'] == 'F', Hnet.nodes(data=True)))[0]
HY = zip(*filter(lambda (n, d): d['Age'] == 'HY', Hnet.nodes(data=True)))[0]
AHY = zip(*filter(lambda (n, d): d['Age'] == 'AHY', Hnet.nodes(data=True)))[0]

#####################################################################################
#####################################################################################

m_d = sorted(zip(*Hnet.degree(males))[1], reverse=True)
f_d = sorted(zip(*Hnet.degree(females))[1], reverse=True)
hy_d = sorted(zip(*Hnet.degree(HY))[1], reverse=True)
ahy_d = sorted(zip(*Hnet.degree(AHY))[1], reverse=True)

#####################################################################################
#####################################################################################

bc = nx.betweenness_centrality(Hnet)
males_bc = { m: bc[m] for m in males }
females_bc = { m: bc[m] for m in females }
AHY_bc = { m: bc[m] for m in AHY }
HY_bc = { m: bc[m] for m in HY }
```


```python
degree = sorted(zip(*Hnet.degree(males))[1], reverse=True)

max(degree)
```




    16




```python
min(degree)
```




    1



#### degree distbutions between males and females statistical comparison


```python
stats.ks_2samp(m_d, f_d)
```




    Ks_2sampResult(statistic=0.1883333333333333, pvalue=0.5583719606128525)



#### degree distbutions between hatch year and after hatch years statistical comparison


```python
stats.ks_2samp(hy_d, ahy_d)
```




    Ks_2sampResult(statistic=0.21169354838709678, pvalue=0.433553562913418)



#### betweenness centrality males and females years statistical comparison


```python
stats.ks_2samp(males_bc.values(), females_bc.values())
```




    Ks_2sampResult(statistic=0.16416666666666674, pvalue=0.7282154106780466)



#### betweenness centrality hatch year and after hatch years statistical comparison


```python
stats.ks_2samp(AHY_bc.values(), HY_bc.values())
```




    Ks_2sampResult(statistic=0.09173387096774188, pvalue=0.9988423204076042)



# Figure 10:
### Contact network of tagged Anna’s (pink nodes) and Allen’s hummingbirds (green nodes). The sex of the hummingbirds indicated by M for males and F for females. The size of the node is proportional to the degree of the interaction. Edges represent time spent together at the feeding station and the width of the edge width is proportional to the time spent together.


```python
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (8,8))
ax1.loglog(m_d, 'b-', marker='o', label = 'males')
ax1.loglog(f_d, 'r-', marker='o', label = 'females')
ax1.legend()
ax1.set_ylabel("degree")
ax1.set_xlabel("rank")

ax2.loglog(hy_d, 'b-', marker='o', label = 'hatch year')
ax2.loglog(ahy_d, 'r-', marker='o', label = 'after hatch years')
ax2.legend()
ax2.set_ylabel("degree")
ax2.set_xlabel("rank")

ax3.loglog(sorted (males_bc.values(), reverse= True), 'b-', marker='o', label = 'males')
ax3.loglog(sorted (females_bc.values(), reverse= True), 'r-', marker='o', label = 'females')
ax3.legend()
ax3.set_ylabel("betweenness centrality")
ax3.set_xlabel("rank")

ax4.loglog(sorted (HY_bc.values(), reverse= True), 'b-', marker='o', label = 'hatch year')
ax4.loglog(sorted (AHY_bc.values(), reverse= True), 'r-', marker='o', label = 'after hatch years')
ax4.legend()
ax4.set_ylabel("betweenness centrality")
ax4.set_xlabel("rank")
plt.tight_layout()
plt.savefig(output_path+'/Figure 10.png', dpi= dpi)
plt.savefig(output_path+'/Figure 10.eps',format= 'eps', dpi= dpi)
plt.show()
```


![png](output_144_0.png)


## Table 6


```python
i = list(pd.unique(interactions[['ID', 'second_bird']].values.ravel('K')))
interacted_birds = meta[meta['Tag Hex'].isin(i)]
inter = pd.pivot_table(data=interacted_birds, index='Species', columns=['Sex', 'Age'] , aggfunc= 'count', values='Tag Hex', fill_value=0, margins= True)
inter
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>Sex</th>
      <th colspan="3" halign="left">F</th>
      <th colspan="3" halign="left">M</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Age</th>
      <th>AHY</th>
      <th>HY</th>
      <th>UNK</th>
      <th>AHY</th>
      <th>HY</th>
      <th>UNK</th>
      <th></th>
    </tr>
    <tr>
      <th>Species</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ALHU</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>9</td>
    </tr>
    <tr>
      <th>ANHU</th>
      <td>10</td>
      <td>14</td>
      <td>0</td>
      <td>18</td>
      <td>17</td>
      <td>5</td>
      <td>64</td>
    </tr>
    <tr>
      <th>All</th>
      <td>10</td>
      <td>14</td>
      <td>1</td>
      <td>21</td>
      <td>18</td>
      <td>9</td>
      <td>73</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.pivot_table(data=interacted_birds, index='Species', columns='Sex' , aggfunc= 'count', values='Tag Hex', fill_value=0, margins= True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Sex</th>
      <th>F</th>
      <th>M</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Species</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ALHU</th>
      <td>1</td>
      <td>8</td>
      <td>9</td>
    </tr>
    <tr>
      <th>ANHU</th>
      <td>24</td>
      <td>40</td>
      <td>64</td>
    </tr>
    <tr>
      <th>All</th>
      <td>25</td>
      <td>48</td>
      <td>73</td>
    </tr>
  </tbody>
</table>
</div>



### Types of interactions


```python
m2 = meta[['Tag Hex', 'Species', 'Age', 'Sex']]
m2.columns = ['second_bird', 'Species2', 'Age2', 'Sex2']
i2 = pd.merge(interactions, m2, on='second_bird', how='left')
def returnTypeInt (c):
    if [(c.Sex == 'M') & (c.Sex2 == 'M')]:
        return 'MM'
    elif [(c.Sex == 'M') & (c.Sex2 == 'F')]:
        return 'MF'
    elif [(c.Sex == 'F') & (c.Sex2 == 'M')]:
        return 'MF'
    elif [(c.Sex == 'F') & (c.Sex2 == 'F')]:
        return 'FF'
    else:
        return 'Problem'
```


```python
conditions = [
    (i2['Sex'] == 'M') & (i2['Sex2'] == 'M'),
    (i2['Sex'] == 'M') & (i2['Sex2'] == 'F'),
    (i2['Sex'] == 'F') & (i2['Sex2'] == 'M'),
    (i2['Sex'] == 'F') & (i2['Sex2'] == 'F')
]
choices = ['MM', 'MF', 'MF', 'FF']
i2['Type'] = np.select(conditions, choices, default='MM')


conditions2 = [
    (i2['overlap'] <= 0),
    (i2['overlap'] > 0)
]#(i2['overlap'] == 0)
choices = ['short_interactions',  'long_interactions'] #'medium_interactions',
i2['DurationType'] = np.select(conditions2, choices, default='short_interaction')

#i2['Type'] = i2.apply(returnTypeInt, axis=1)
i2['Type'].value_counts()
```




    MM    1020
    MF     491
    FF     124
    Name: Type, dtype: int64




```python
#i2['Type'] = i2.apply(returnTypeInt, axis=1)
i2['Type'].value_counts()
```




    MM    1020
    MF     491
    FF     124
    Name: Type, dtype: int64




```python
i2['DurationType'].value_counts()
```




    short_interactions    1608
    long_interactions       27
    Name: DurationType, dtype: int64




```python
inter_pv = pd.pivot_table(columns='DurationType', index='Type',data=i2, fill_value=0, aggfunc='count', values='ID')
inter_pv
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>DurationType</th>
      <th>long_interactions</th>
      <th>short_interactions</th>
    </tr>
    <tr>
      <th>Type</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>FF</th>
      <td>7</td>
      <td>117</td>
    </tr>
    <tr>
      <th>MF</th>
      <td>4</td>
      <td>487</td>
    </tr>
    <tr>
      <th>MM</th>
      <td>16</td>
      <td>1004</td>
    </tr>
  </tbody>
</table>
</div>




```python
i2.Age2.unique()
```




    array([u'HY', u'AHY', u'UNK'], dtype=object)




```python
conditions3 = [
    (i2['Age'] == 'HY') & (i2['Age2'] == 'HY'),
    (i2['Age'] == 'HY') & (i2['Age2'] == 'AHY'),
    (i2['Age'] == 'HY') & (i2['Age2'] == 'UNK'),
    
    (i2['Age'] == 'AHY') & (i2['Age2'] == 'HY'),
    (i2['Age'] == 'AHY') & (i2['Age2'] == 'AHY'),
    (i2['Age'] == 'AHY') & (i2['Age2'] == 'UNK'),
    
    (i2['Age'] == 'UNK') & (i2['Age2'] == 'HY'),
    (i2['Age'] == 'UNK') & (i2['Age2'] == 'AHY'),
    (i2['Age'] == 'UNK') & (i2['Age2'] == 'UNK'),
    ]
choices = ['HyHy', 'HyAHy', 'HyUNK', 'HyAHy', 'AHyAHy', 'AHyUNK', 'HyUNK', 'AHyUNK', 'UNKUNK']
i2['TypeAge'] = np.select(conditions3, choices, default='UNKUNK')
```


```python
i2['TypeAge'].value_counts()
```




    HyHy      494
    HyAHy     430
    AHyAHy    303
    AHyUNK    240
    HyUNK     126
    UNKUNK     42
    Name: TypeAge, dtype: int64




```python
age_pv = pd.pivot_table(columns='DurationType', index='TypeAge',data=i2, fill_value=0, aggfunc='count', values='ID')
age_pv
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>DurationType</th>
      <th>long_interactions</th>
      <th>short_interactions</th>
    </tr>
    <tr>
      <th>TypeAge</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AHyAHy</th>
      <td>4</td>
      <td>299</td>
    </tr>
    <tr>
      <th>AHyUNK</th>
      <td>2</td>
      <td>238</td>
    </tr>
    <tr>
      <th>HyAHy</th>
      <td>8</td>
      <td>422</td>
    </tr>
    <tr>
      <th>HyHy</th>
      <td>13</td>
      <td>481</td>
    </tr>
    <tr>
      <th>HyUNK</th>
      <td>0</td>
      <td>126</td>
    </tr>
    <tr>
      <th>UNKUNK</th>
      <td>0</td>
      <td>42</td>
    </tr>
  </tbody>
</table>
</div>




```python
i2.set_index(pd.DatetimeIndex(i2['visit_start']), inplace= True)
```


```python
i2.groupby([ pd.Grouper(freq='M'), 'DurationType'])['ID'].count().unstack('DurationType').plot(kind="line", rot =45, stacked=False)
plt.show()
```


![png](output_159_0.png)



```python
i2.groupby([ pd.Grouper(freq='M'), 'DurationType'])['ID'].count().unstack('DurationType')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>DurationType</th>
      <th>long_interactions</th>
      <th>short_interactions</th>
    </tr>
    <tr>
      <th>visit_start</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-10-31</th>
      <td>NaN</td>
      <td>49.0</td>
    </tr>
    <tr>
      <th>2016-11-30</th>
      <td>NaN</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2016-12-31</th>
      <td>NaN</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2017-01-31</th>
      <td>NaN</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2017-04-30</th>
      <td>NaN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2017-05-31</th>
      <td>NaN</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>2017-06-30</th>
      <td>4.0</td>
      <td>83.0</td>
    </tr>
    <tr>
      <th>2017-07-31</th>
      <td>15.0</td>
      <td>350.0</td>
    </tr>
    <tr>
      <th>2017-08-31</th>
      <td>3.0</td>
      <td>160.0</td>
    </tr>
    <tr>
      <th>2017-09-30</th>
      <td>NaN</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>2017-10-31</th>
      <td>NaN</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>2017-11-30</th>
      <td>NaN</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>2017-12-31</th>
      <td>NaN</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>2018-01-31</th>
      <td>NaN</td>
      <td>38.0</td>
    </tr>
    <tr>
      <th>2018-02-28</th>
      <td>NaN</td>
      <td>180.0</td>
    </tr>
    <tr>
      <th>2018-03-31</th>
      <td>5.0</td>
      <td>647.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
i2['visit_start'] = pd.to_datetime(i2['visit_start'])
i2.set_index(pd.DatetimeIndex(i2['visit_start']), inplace= True)
multi_index = pd.MultiIndex.from_product([pd.date_range('2016-9-22', i2.visit_start.max().date(),freq='1H'), i2['DurationType'].unique()], names=['Date', 'DurationType'])
a = i2.groupby([ pd.Grouper(freq='1H'), 'DurationType'])['ID'].count().reindex(multi_index, fill_value=0).unstack('DurationType').reset_index()#.plot(kind="bar", rot =0, stacked=True).res
a['h'] = pd.to_datetime(a['Date'], format= '%H:%M:%S' ).dt.time
agg_funcs = {'long_interactions':[np.mean, 'sem'], 'short_interactions':[np.mean, 'sem']}
a = a.groupby(a.h).agg(agg_funcs).reset_index()
a.columns = a.columns.droplevel(0)
a.columns = ['hour', 'mean_long', 'sem_long', 'mean_short', 'sem_short',]
#a['hour'] = pd.to_datetime(a['hour'])
a = a.set_index(a['hour'])
a
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hour</th>
      <th>mean_long</th>
      <th>sem_long</th>
      <th>mean_short</th>
      <th>sem_short</th>
    </tr>
    <tr>
      <th>hour</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>00:00:00</th>
      <td>00:00:00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>01:00:00</th>
      <td>01:00:00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>02:00:00</th>
      <td>02:00:00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>03:00:00</th>
      <td>03:00:00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>04:00:00</th>
      <td>04:00:00</td>
      <td>0.003604</td>
      <td>0.002546</td>
      <td>0.030631</td>
      <td>0.009624</td>
    </tr>
    <tr>
      <th>05:00:00</th>
      <td>05:00:00</td>
      <td>0.003604</td>
      <td>0.002546</td>
      <td>0.172973</td>
      <td>0.032830</td>
    </tr>
    <tr>
      <th>06:00:00</th>
      <td>06:00:00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.214414</td>
      <td>0.036315</td>
    </tr>
    <tr>
      <th>07:00:00</th>
      <td>07:00:00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.194595</td>
      <td>0.036293</td>
    </tr>
    <tr>
      <th>08:00:00</th>
      <td>08:00:00</td>
      <td>0.007207</td>
      <td>0.004407</td>
      <td>0.227027</td>
      <td>0.039977</td>
    </tr>
    <tr>
      <th>09:00:00</th>
      <td>09:00:00</td>
      <td>0.005405</td>
      <td>0.003115</td>
      <td>0.108108</td>
      <td>0.017625</td>
    </tr>
    <tr>
      <th>10:00:00</th>
      <td>10:00:00</td>
      <td>0.001802</td>
      <td>0.001802</td>
      <td>0.138739</td>
      <td>0.024082</td>
    </tr>
    <tr>
      <th>11:00:00</th>
      <td>11:00:00</td>
      <td>0.001802</td>
      <td>0.001802</td>
      <td>0.149550</td>
      <td>0.023829</td>
    </tr>
    <tr>
      <th>12:00:00</th>
      <td>12:00:00</td>
      <td>0.001802</td>
      <td>0.001802</td>
      <td>0.131532</td>
      <td>0.021293</td>
    </tr>
    <tr>
      <th>13:00:00</th>
      <td>13:00:00</td>
      <td>0.001802</td>
      <td>0.001802</td>
      <td>0.142342</td>
      <td>0.022507</td>
    </tr>
    <tr>
      <th>14:00:00</th>
      <td>14:00:00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.133333</td>
      <td>0.022244</td>
    </tr>
    <tr>
      <th>15:00:00</th>
      <td>15:00:00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.136937</td>
      <td>0.021000</td>
    </tr>
    <tr>
      <th>16:00:00</th>
      <td>16:00:00</td>
      <td>0.001802</td>
      <td>0.001802</td>
      <td>0.218018</td>
      <td>0.029557</td>
    </tr>
    <tr>
      <th>17:00:00</th>
      <td>17:00:00</td>
      <td>0.012613</td>
      <td>0.005384</td>
      <td>0.473874</td>
      <td>0.083434</td>
    </tr>
    <tr>
      <th>18:00:00</th>
      <td>18:00:00</td>
      <td>0.003604</td>
      <td>0.002546</td>
      <td>0.275676</td>
      <td>0.056249</td>
    </tr>
    <tr>
      <th>19:00:00</th>
      <td>19:00:00</td>
      <td>0.003604</td>
      <td>0.002546</td>
      <td>0.113514</td>
      <td>0.023778</td>
    </tr>
    <tr>
      <th>20:00:00</th>
      <td>20:00:00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>21:00:00</th>
      <td>21:00:00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>22:00:00</th>
      <td>22:00:00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>23:00:00</th>
      <td>23:00:00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.rcParams['font.family'] = 'Times New Roman'
f, (ax1, ax2)  = plt.subplots(1, 2, figsize=(12,4), dpi=dpi, sharey=True, )
a.plot(y='mean_long', kind = 'bar', yerr='sem_long', color='#3182bd',ax= ax1, legend=False)
a.plot(y='sem_short', kind = 'bar', yerr='sem_short', color='#3182bd',ax= ax2, legend=False)
yaxis_text = 'average interactions'
ax1.set_ylabel(yaxis_text)
ax2.set_ylabel(yaxis_text)

ax1.set_title('long interactions')
ax2.set_title('short interactions')

ticks = [ x.strftime('%H:%M') for x in a.index.values ]
ax1.set_xticklabels(ticks)
ax2.set_xticklabels(ticks)

ax1.set_xlabel('time')
ax2.set_xlabel('time')
plt.tight_layout()
#plt.savefig(output_path+'/Interactions_daily_variation.png', dpi=600, figsize = (6,4))
plt.show()
```


![png](output_162_0.png)

