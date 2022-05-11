

import numpy as np
import networkx as nx
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import time,os
from wordfreq import word_frequency



def graph_func(p0, pN, s: nx.Graph, out:dict, fitness_func):
    '''
    This is the code taken from a previous project... and modified slightly
    in order to match this problem.

    recursive function to get x and y points of the disconnectivity graph
    4/28/22 -- I'm so sorry coding gods, I took this from dg , but
    I don't want to mess up dg b/c its working solid, but I need to change this
    function...

    start with p0=0 and pN=1
    :param p0: start x
    :param pN: end x
    :param s:  a new nx.Graph instance, where nodes are integers
    :param out: semantic_fluency_dict so it's the node int to node value mapping
    :return:
    '''
    N = s.number_of_nodes()

    if N == 1:
        y = fitness_func(list(s.nodes)[0])
        x = p0 + (pN - p0) / 2
        return [(x, y, out[list(s.nodes)[0]])]

    #
    # if fitness_func==None:
    #     # we need to define the sort function artificially if no sort function is defined.
    #     fitness_func=lambda x:x

    node2remove = sorted(s.nodes,key=fitness_func)[0]
    print(f"removing node {out[node2remove]} with fitness {fitness_func(node2remove)}")
    s.remove_node(node2remove) # remove the smallest node... true.
    m = []
    nodes = list(s.nodes)
    for nds in nodes:
        if s.degree[nds] == 0:
            m.append(nds)
            s.remove_node(nds)

    components = [c for c in nx.connected_components(s)]
    length_list = [len(c) for c in components]

    # todo: test this function with an ideal test set to make sure its working

    # randomly put the new basin on either side if the graph, we don't want new ,
    # basins always on same side of the graph.
    if np.random.randint(0, 2) and len(m) > 0:
        m_i = min(m)
        length_list += [len(m)]
        components += [{m_i}]
        s.add_edge(m_i, m_i)
    elif len(m) > 0:
        m_i = min(m)
        length_list = [len(m)] + length_list
        components = [{m_i}] + components
        s.add_edge(m_i, m_i)

    # length of the given section
    xypoints = []
    length_of_rod = (pN - p0) - (pN - p0) / N
    for l, c, i in zip(length_list, components, np.arange(len(components))):
        # the first p0 has to have this specific value. Then after it is additive by a different amount
        if i == 0:
            # get proper starting point for p0
            p0 = p0 + (pN - p0) / (2 * N)
        else:
            p0 = pN_tilde
        ratio = l / (N - 1)
        pN_tilde = p0 + ratio * length_of_rod

        # something here is messed up with my p0 and pN_tilde's
        xypoints += [(p0, fitness_func(node2remove))]
        xypoints += [(pN_tilde, fitness_func(node2remove))]
        H = nx.Graph()
        H.add_edges_from(list(s.edges(c)))
        xypoint = graph_func(p0, pN_tilde, H, out,fitness_func)
        # add all the points that were found!!!!
        xypoints += xypoint

    return xypoints

def get_xypoints(G,fluency_dict,fitness_func):
    xypoints = graph_func(0, 1, G, fluency_dict,fitness_func)
    # your not even including the intial pionts boy..
    # xypoints += [(1, 0), (0, 0)]
    xypoints = sorted(xypoints, key=lambda xy: xy[0])
    return xypoints

def check_semantic_func_word_fitness(fitness_func,fluency_list,fluency_dict):
    fluency_dict_values=list(fluency_dict.values())

    total_words=[]
    '''
    need to make sure that i'm mapping fluency list to the value in the fluency dict
    '''
    for l in fluency_list:
        for word in l:
            node_nb= fluency_dict_values.index(word)
            total_words.append((word,fitness_func(node_nb)))
            # print(f"{word}---fitness:{fitness_func(node_nb)}")

    sorted(total_words,key=lambda x: x[1])
    print('=====================')
    for tw in total_words:
        print(f"word: {tw[0]} --- fitness {tw[1]}")

    assert (np.array([tw[1]!=0 for tw in total_words])).all()  , 'all fitness values have to be non-zero'
def semantic_func_word_fitness(node_nb,fitness_value:str,semantic_fluecy_list,semantic_fluency_dict):
    '''
    semantic_fluecy_list : must be a list of shape 2D
    this function uses the earliest word as the fitness value, having a value happen earlier in the list
    corresponds to better fitness.
    '''
    node_word=semantic_fluency_dict[node_nb]
    total=[]
    second=[]
    for single_fluency_list in semantic_fluecy_list:
        indices=[]
        for i,x in enumerate(single_fluency_list):
            if x.lower()==node_word.lower():
                indices.append(i)
        if len(indices)>0:
            total.append(eval(fitness_value))
    return sum(total)


def closest(node_words:list,word:str):
    '''
    sees what word in a given graph is closest to a word.
    :param node_words: list of values of all words in the graph
    :param word: specific word to compare what it is closest too.
    :return: the closest word
    '''
    distances= np.array([nltk.edit_distance(nw,word) for nw in node_words])
    return node_words[np.argmin(distances)]


def make_fluency_list_for_a_sub(df,sub,diagnosis,fluency_dict):
    ''''
    df is the data frame of the ucsd_fluency_for_snafu_20180518.csv
    and sub is a string of the subject unique id number
    '''
    fluency_dict_values=list(fluency_dict.values())
    temp_df=df[df['id']==int(sub)].copy()
    var = [temp_df['item'][(temp_df['listnum'] == i) & (temp_df['group']==diagnosis)].tolist() for i in temp_df['listnum'].unique()]
    var =[v for v in var if len(v)>0]

    # this code is terrible but strings are annoying
    var2=[]

    for v in var:
        #gross double for-loop
        v2=[]
        for word in v:
            v2.append(closest(fluency_dict_values,word))
        var2.append(v2)


    # print("var",var)
    # print('after edits',var2)


    assert np.array([len(v2) for v2 in var2]).shape== np.array([len(v) for v in var]).shape,\
        'shapes after conversion are different'

    return var2


### make different variations of fitness, where like one variation is like
#### hey how can we like combine word frequency with the amount they choose it,
#### so its kinda like a prior... oooo this is getting cool .

#### also I could compare the importance of using different lists, that would be pretty cool.

def make_graph_and_dg(xypoints,input):
    fig,ax=plt.subplots(1,1)
    x = [xy[0] for xy in xypoints]
    y = [xy[1] for xy in xypoints]
    ax.plot(x, y, markersize=5, marker='o')
    ax.set_xticks([])
    ax.set_ylabel('fitness')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_title(f"Patient {input['subject_nb']} \n "
                 f"Diagnosis {input['diagnosis']}\n")

    for xy in xypoints:
        if len(xy)>2:
            ax.text(xy[0],xy[1],xy[2],horizontalalignment='center')

    # fig.show()
    outdir=f"dataCS841/results/{input['subject_nb']}"
    if not os.path.exists(outdir):
        os.mkdir(f"{outdir}")
    fig.savefig(f"{outdir}/{input['fitness_value']}_{input['diagnosis']}.png")


def make_semantic_dg(adj_matrix,fluency_dict,diagnosis,subject_nb,myfitness_func,**kwargs):
    '''
    this makes a semantic disconnectivity graph for one subject for one diagnosis
    '''
    # print('adj_matrix',adj_matrix,'\n'
    #       'fluency_dict',fluency_dict,'\n'
    #       'diagnosis',diagnosis,'\n'
    #       'subject_nb',subject_nb,'\n'
    #       'myfitness_func',myfitness_func )

    G=nx.from_numpy_array(adj_matrix)
    xypoints=get_xypoints(G,fluency_dict,myfitness_func)
    return xypoints


labels=['prior','prior*exp(end)','prior*exp(-beg)','beg','end','end*prior','beg*prior']
functions=["word_frequency(single_fluency_list[indices[0]], 'en', wordlist='best', minimum=0.0)",
           "np.exp(len(single_fluency_list) - indices[0]) * word_frequency(single_fluency_list[indices[0]], 'en', wordlist='best', minimum=0.0)",
           "np.exp(-indices[0]) * word_frequency(single_fluency_list[indices[0]], 'en', wordlist='best', minimum=0.0)",
           "indices[0]",
           'len(single_fluency_list) - indices[0]',
           "(len(single_fluency_list)-indices[0])*word_frequency(single_fluency_list[indices[0]], 'en', wordlist='best', minimum=0.0)",
           "(indices[0])*word_frequency(single_fluency_list[indices[0]], 'en', wordlist='best', minimum=0.0)"]

fitness_func_library=dict(zip(labels,functions))


def normalize(fluency_dict,fluency_list,fitness_func):
    Z=0
    for word in np.unique(np.array(sum(fluency_list, []))):
        node_nb=list(fluency_dict.values()).index(word)
        Z+=fitness_func(node_nb)
    return Z

def fluency_lists_playing_around(fluency_list,input):
    what2plot=['prior','end*prior']
    list_flatten = sum(fluency_list, [])
    df = pd.DataFrame()
    df['words'] = list_flatten
    df = df.set_index('words')
    list_flatten = [list(input['fluency_dict'].values()).index(lf) for lf in list_flatten]
    for f,label in zip(fitness_func_library.values(),fitness_func_library.keys()):
        fitness_func=lambda x: semantic_func_word_fitness(x,
                         semantic_fluecy_list=fluency_list,
                         semantic_fluency_dict=input['fluency_dict'],
                         fitness_value=f)
        Z=normalize(input['fluency_dict'],fluency_list,fitness_func)
        if label in what2plot:
            df[label] = [fitness_func(lf)/Z  for lf in list_flatten]

    df.plot.scatter(x=what2plot[0],y=what2plot[1])
    plt.title(f'Correlation between {what2plot[0]} vs {what2plot[1]} (Normalized)')
    # plt.ylabel('Fitness')
    # plt.gcf().subplots_adjust(bottom=0.3)
    plt.semilogy()
    plt.semilogx()
    # plt.yticks([1e-4,1e-3,1e-2,1e-1])
    # plt.xticks([1e-4, 1e-3, 1e-2, 1e-1])
    plt.savefig(f"dataCS841/results/{input['subject_nb']}/{'_'.join(what2plot)}_normalize_correlation.png")

    # for lf,wf in zip(list_flatten,wf):
    #     print(f"{lf}--{wf}")


if __name__ == '__main__':

    ### need a much stronger input format here,,, my goodness.
    # then i also have to consider what I want to save.
    #len(single_fluency_list)
    # np.exp(-indices[0])
    # possible_fitness_funcs=['len(single_fluency_list)-indices[0]','indices[0]']
    file = open('dataCS841/ucsd_ad_graphs_usf_persev.pickle', 'rb')
    data = pickle.load(file, encoding='latin1')
    sub='14003'
    graph_df=pd.DataFrame(data).set_index('subs')
    UCSD_FLUENCY_FOR_SNAFU = pd.read_csv('dataCS841/ucsd_fluency_for_snafu_20180518.csv')
    input = {'adj_matrix': graph_df.loc[sub]['subgraphs'],
             'fluency_dict': graph_df.loc[sub]['items'],
             'diagnosis': 'ProbAD',
             'subject_nb': sub,
             'fitness_value':"np.exp(-indices[0]) * word_frequency(single_fluency_list[indices[0]], 'en', wordlist='best', minimum=0.0)"}

    print(f"running \n {input['subject_nb']} \n {input['diagnosis']}"
          f"\n {input['fitness_value']} ")

    print("====================starting run ==================")

             # 'fitness_value':'len(single_fluency_list)-indices[0]'}
    print('starting to make fluency list')
    start=time.time()
    fluency_list = make_fluency_list_for_a_sub(UCSD_FLUENCY_FOR_SNAFU, input['subject_nb'], input['diagnosis'],input['fluency_dict'])

    print(f"total time elapsed for making fluency list {time.time()-start}")
    input['myfitness_func']= lambda x : semantic_func_word_fitness(x,
                                                                   semantic_fluecy_list=fluency_list,
                                                                   semantic_fluency_dict=input['fluency_dict'],
                                                                   fitness_value=input['fitness_value'])

    #check_semantic_func_word_fitness(input['myfitness_func'],fluency_list,input['fluency_dict'])
    xypoints=make_semantic_dg(**input)
    print("fluency list\n",fluency_list)
    make_graph_and_dg(xypoints,input)

    fluency_lists_playing_around(fluency_list,input)







