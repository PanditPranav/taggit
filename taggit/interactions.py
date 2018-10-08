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

import networkx as nx
from bokeh.io import show, output_file , output_notebook, save
output_notebook()
from bokeh.io import show, output_file
from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, TapTool, BoxSelectTool, BoxZoomTool, ResetTool, PanTool, WheelZoomTool
import  bokeh.models.graphs as graphs
#from bokeh.model.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.palettes import Spectral4


def get_intreactions(data):
    
    data['visit_start'] =pd.to_datetime(data.visit_start)
    data['visit_end'] =pd.to_datetime(data.visit_end)
    data['Date'] =pd.to_datetime(data.Date)
    data.sort_values('visit_start', inplace= True)
    #df = data[['visit_start','visit_end','ID','Tag','visit_duration']]
    
    data['v1_start'] = data['visit_start']
    data['v2_start'] = data['visit_start'].shift(-1)
    data['v1_end'] = data['visit_end']
    data['v2_end'] = data['visit_end'].shift(-1)
    data['second_bird'] = data['Tag Hex'].shift(-1)
    data['second_tag'] = data['Tag'].shift(-1)
    
    latest_start = []
    for index, row in data.iterrows():
        m = max(row['v2_start'], row['v1_start'])
        latest_start.append(m)

    earliest_end = []
    for index, row in data.iterrows():
        z = min(row['v2_end'], row['v1_end'])
        earliest_end.append(z)


    data['earliest_end'] = earliest_end
    data['latest_start'] = latest_start
    data['overlap'] = data.earliest_end-data.latest_start
    
    def convert_seconds(c):
        return c.overlap.total_seconds()
    
    data['overlap'] = data.apply(convert_seconds, axis=1)

    #b = data[(data.ID != data.second_bird) & (data.Tag == data.second_tag) & (data.overlap >= 0)]
    #b = data[(data.ID != data.second_bird) & (data.Tag == data.second_tag) & (data.overlap > 0)]
    b = data[(data.ID != data.second_bird) & (data.Tag == data.second_tag) & (data.overlap >=-10)]
    
    return b
           

##############################################################################################
##############################################################################################

def get_interaction_networks(network_name, data, interactions, location):
    
    DG=nx.Graph()
    """Initiating host nodes"""
    birds = data.groupby(['ID', 'Sex', 'Age', 'Location', 'Species']).size().reset_index().rename(columns={0:'count'})
    birds_n = pd.unique(interactions[['ID', 'second_bird']].values.ravel('K'))
    birds = birds[birds['ID'].isin(birds_n)]
    for index, row in birds.iterrows():
        DG.add_node(row['ID'], type="host", 
                    Sex = row['Sex'],Age = row['Age'], Location= row['Location'], Species= row['Species'])

    """Iterating through the raw data to add Edges if a virus is found in a host"""
    for index, row in interactions.iterrows():
        DG.add_edge(row['ID'], row['second_bird'], weight = row['overlap'] + 12)

    """Creating positions of the nodes"""
    #layout = nx.spring_layout(DG, k = 0.05, scale=2) #
    layout = nx.fruchterman_reingold_layout(DG, k = 0.05, iterations=50)
    """graph ready"""
    nx.write_graphml(DG, location +'/'+ network_name + "hummingbirds_interaction.graphml")
    nx.draw(DG)  # networkx draw()
    plt.draw()
    return DG
        

##############################################################################################
##############################################################################################

def interactive_plot(network, network_name, layout_func = 'fruchterman_reingold'):
    plot = Plot(plot_width=800, plot_height=800,
                x_range=Range1d(-1.1,1.1), y_range=Range1d(-1.1,1.1))
    
    plot.title.text = network_name
    
    plot.add_tools(HoverTool(tooltips=[('bird','@index'),("age","@Age"), 
                                       ("sex","@Sex"),("location",'@Location')]),
                   TapTool(),
                   BoxSelectTool(),
                   BoxZoomTool(), 
                   ResetTool(),
                   PanTool(),
                   WheelZoomTool())
    if layout_func == 'fruchterman_reingold':
        graph_renderer = graphs.from_networkx(network, nx.fruchterman_reingold_layout, scale=1, center=(0,0))
        
    elif layout_func =='spring':
        graph_renderer = graphs.from_networkx(network, nx.spring_layout, scale=1, center=(0,0))
        
    elif layout_func =='circular':
        graph_renderer = graphs.from_networkx(network, nx.circular_layout, scale=1, center=(0,0))
        
    elif layout_func == 'kamada':
        graph_renderer = graphs.from_networkx(network, nx.kamada_kawai_layout, scale=1, center=(0,0))
        
    elif layout_func == 'spectral':
        graph_renderer = graphs.from_networkx(network, nx.spectral_layout, scale=1, center=(0,0))
        
    else:
        graph_renderer = graphs.from_networkx(network, nx.fruchterman_reingold_layout, scale=1, center=(0,0))
    
    centrality = nx.algorithms.centrality.betweenness_centrality(network)
    """ first element are nodes again """
    _, nodes_centrality = zip(*centrality.items())
    max_centraliy = max(nodes_centrality)
    c_centrality = [7 + 15 * t / max_centraliy
                      for t in nodes_centrality]
    
    import community #python-louvain
    partition = community.best_partition(network)
    p_, nodes_community = zip(*partition.items())
    
    community_colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628',
                        '#b3cde3','#ccebc5','#decbe4','#fed9a6','#ffffcc','#e5d8bd','#fddaec',
                        '#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d',
                        '#666666']
    colors = [community_colors[t % len(community_colors)] for t in nodes_community]
    
    

    
    
    _, Sex = zip(*nx.get_node_attributes(network, 'Sex').items())
    _, Age = zip(*nx.get_node_attributes(network, 'Age').items())
    _, Location = zip(*nx.get_node_attributes(network, 'Location').items())
    graph_renderer.node_renderer.data_source.add(c_centrality, 'centrality')
    graph_renderer.node_renderer.data_source.add(Sex, 'Sex')
    graph_renderer.node_renderer.data_source.add(Age, 'Age')
    graph_renderer.node_renderer.data_source.add(Location, 'Location')
    
    graph_renderer.node_renderer.data_source.add(colors, 'colors')
    graph_renderer.node_renderer.glyph = Circle(size='centrality', fill_color='colors')
    graph_renderer.node_renderer.selection_glyph = Circle(size='centrality', fill_color=Spectral4[2])
    graph_renderer.node_renderer.hover_glyph = Circle(size=15, fill_color=Spectral4[1])
    
    _, edge_weights = zip(*nx.get_edge_attributes(network,'weight').items())
    max_weights = max(edge_weights)
    c_weights = [7 + 2 * (t / max_weights)
                      for t in edge_weights]
    
    graph_renderer.edge_renderer.data_source.add(c_weights, 'weight')
    graph_renderer.edge_renderer.glyph = MultiLine(line_color="#757474", line_alpha=0.2, line_width='weight')
    graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width='weight')
    graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width='weight')

    graph_renderer.selection_policy = graphs.NodesAndLinkedEdges()
    graph_inspection_policy = graphs.NodesOnly() 
    #graph_renderer.inspection_policy = graphs.EdgesAndLinkedNodes()

    plot.renderers.append(graph_renderer)

    #output_file("interactive_graphs.html")
    return plot

##############################################################################################
##############################################################################################


def run_permutation_test(dependent, network, number_of_permutations, output_path):
    nodes = pd.DataFrame.from_dict(dict(network.nodes(data=True)), orient='index')
    degree = pd.DataFrame.from_dict(dict(network.degree()), orient='index')
    centrality = pd.DataFrame.from_dict(dict(nx.betweenness_centrality(network)), orient='index')
    h1 = pd.concat([nodes, degree, centrality], axis=1).reset_index(0)
    h1.columns = ['ID', 'Age', 'Species', 'type', 'Location', 'Sex', 'degree', 'centrality']
    h1['degree_dist'] = h1.degree/float(h1.degree.max())

    equation = dependent + "~ Age + Sex"
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.cov_struct import (Exchangeable,
        Independence,Autoregressive)
    from statsmodels.genmod.families import Poisson
    fam = Poisson()
    ind = Independence()

    model = GEE.from_formula(equation, "Location", h1, cov_struct=ind, family=fam)
    main_model_result = model.fit()
    main_result  = pd.DataFrame(main_model_result.params).T

    degree_random_coeff = []
    for i in range (number_of_permutations):
        rand_h1= h1.copy()
        rand_h1[dependent] = np.random.permutation(h1[dependent])
        fam = Poisson()
        ind = Independence()
        model = GEE.from_formula(equation, "Location", rand_h1, cov_struct=ind, family=fam)
        result = model.fit()
        degree_random_coeff.append(result.params)



    d = pd.DataFrame.from_records(degree_random_coeff)
    import seaborn as sns
    f, (ax1,ax2, ax3) = plt.subplots(1, 3, sharey=True)
    ax1.hist(d['Age[T.HY]'], bins = 100)
    ax1.axvline(x=main_result['Age[T.HY]'].values[0], color = '#fc9272')
    p = (d['Age[T.HY]']>main_result['Age[T.HY]'].values[0]).sum()/float(number_of_permutations)
    if p>0.5:
        p = 1-p
    else:
        p = p
    ax1.set_xlabel('Coefficient Age: Hatch Year\n(ref: After Hatch Year)\np= '+'{0:.2f}'.format(p))
    ax1.set_ylabel('Frequency')

    ax2.hist(d['Age[T.UNK]'], bins = 100)
    ax2.axvline(x=main_result['Age[T.UNK]'].values[0], color = '#fc9272')
    p = (d['Age[T.UNK]']>main_result['Age[T.UNK]'].values[0]).sum()/float(number_of_permutations)
    if p>0.5:
        p = 1-p
    else:
        p = p

    ax2.set_xlabel('Coefficient Age: Unknown\n(ref: After Hatch Year)\np= '+'{0:.2f}'.format(p))

    ax3.hist(d['Sex[T.M]'], bins = 100)
    ax3.axvline(x=main_result['Sex[T.M]'].values[0], color = '#fc9272')
    p = (d['Sex[T.M]']>main_result['Sex[T.M]'].values[0]).sum()/float(number_of_permutations)
    if p>0.5:
        p = 1-p
    else:
        p = p

    ax3.set_xlabel('Coefficient Sex: Male\n (ref: Female)\np= '+'{0:.2f}'.format(p))
    title = 'permutation test for '+ dependent  
    f.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path+'/'+dependent+'_Permutation_test.png', dpi = 300)
    plt.show()