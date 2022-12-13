from FLAI import data
import bnlearn as bn
import networkx as nx
import matplotlib.pyplot as plt
from netgraph import Graph,InteractiveGraph
from pgmpy.factors.discrete import TabularCPD
import numpy as np
import itertools
import math
class CausalGraph():

    def __init__(self, flai_dataset = None, node_edge = None, CPD = None, indepence_test = True,
                 root_node = None, verbose = 0):
        if flai_dataset is None and node_edge is None:
            raise Exception("Data or edges should be provided")
        if node_edge is None:
            if not isinstance(flai_dataset,data.Data):
                raise Exception("Data should be a FLAI.data.Data class")
            self.flai_dataset = flai_dataset
            if root_node is None:
                self.graph = bn.structure_learning.fit(flai_dataset.data,verbose= verbose)
            else:
                self.graph = bn.structure_learning.fit(flai_dataset.data,root_node= root_node,verbose= verbose)
            
            if indepence_test:
                self.independence_test(flai_dataset, test='chi_square', prune=True,verbose= verbose)
        else:
            if CPD is None:
                self.flai_dataset = flai_dataset
                self.graph = bn.make_DAG(node_edge, verbose= verbose)
            else:
                self.flai_dataset = flai_dataset
                self.graph = bn.make_DAG(node_edge, CPD=CPD,verbose= verbose)


    def independence_test(self, flai_dataset, test='chi_square', prune=True,verbose= 0):
        self.graph = bn.independence_test(self.graph, flai_dataset.data, test, prune,verbose= verbose)

    def learn_cpd(self, flai_dataset = None, methodtype= None,verbose = 0):
        if methodtype is None:
            methodtype = 'bayes'
        if flai_dataset is None and self.flai_dataset is None:
            raise Exception("Data should be provided")
        if flai_dataset is None: 
            self.graph = bn.parameter_learning.fit(self.graph,  self.flai_dataset.data,methodtype = methodtype,verbose= verbose) 
        else:
            if not isinstance(flai_dataset,data.Data):
                raise Exception("Data should be a FLAI.data.Data class")
            self.flai_dataset = flai_dataset
            self.graph = bn.parameter_learning.fit(self.graph, self.flai_dataset.data,methodtype = methodtype,verbose= verbose) 

    def calculate_cpd(self,verbose = 0):
        list_cpd = []
        for node in list(self.graph['model'].nodes()):
            node_value = list(np.sort(self.flai_dataset.data[node].unique()))
            evidence = [edge[0] for edge in self.graph['model_edges'] if edge[1] == node]
            evidence_value = [list(np.sort(self.flai_dataset.data[e].unique())) for e in evidence]
                
            evidence_combination = evidence_value
            evidence_combination = list(itertools.product(*evidence_combination))
            list_probas = []
            for ec in evidence_combination:
                filters = [[filter[0],filter[1]]  for filter in zip(evidence,ec)]
                data_aux =  self.flai_dataset.data
                for f in filters:
                    data_aux = data_aux[data_aux[f[0]] == f[1]]
                list_probas = list_probas + [[0.5 if math.isnan((data_aux[node] == nv).sum() / len(data_aux[node] == nv)) else (data_aux[node] == nv).sum() / len(data_aux[node] == nv) for nv in node_value]]
            list_cpd = list_cpd + [TabularCPD(node, len(node_value), np.array(list_probas).T.tolist(),
                        evidence= evidence,
                        evidence_card= [len(ev) for ev in evidence_value])]
        self.graph = bn.make_DAG(list(self.graph['model_edges']), CPD=list_cpd,verbose= verbose)
    def plot(self, directed = True):
        if directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        G.add_edges_from(self.graph['model_edges'])
        plt.figure(figsize=(5,5)) 
        nx.draw_circular(
            G, with_labels=True, arrowsize=30, node_size=800, 
            alpha=0.3, font_weight="bold"
        )

        #Graph(G, node_layout='radial', node_shape= 'o',
        #node_labels=True ,node_size=10, arrows = directed
        #    )
        
    
    def get_CPDs(self):
        CPDs = bn.print_CPD(self.graph,verbose = 0)
        return CPDs

    def generate_dataset(self, n_samples = 1000, methodtype = 'bayes', verbose = 0):
        df = bn.sampling(self.graph, n= n_samples, methodtype = methodtype, verbose = verbose)
        return data.Data(df, transform=False)

    def mitigate_edge_relation(self, sensible_feature = []):
        ### improve for more than one sensible feature 
        if len(sensible_feature) == 0:
            raise Exception("Sensible features should be provided")
        impacted = [e for e in self.graph['model_edges'] if e[0] == 'Sex' ]
        fair_edges = []
        for sf in sensible_feature:
            for e in self.graph['model_edges']:
                if e[1] == sf:
                    for e_impacted in impacted:
                        fair_edges = fair_edges + [(e[0],e_impacted[1])]
                else:
                    fair_edges = fair_edges + [e]
        return fair_edges
    def mitigate_calculation_cpd(self,verbose = 0,sensible_feature = []):
        if len(sensible_feature) == 0:
            raise Exception("Sensible feature should be provided")
        list_cpd = []
        for node in list(self.graph['model'].nodes()):
            node_value = list(np.sort(self.flai_dataset.data[node].unique()))
            evidence = [edge[0] for edge in self.graph['model_edges'] if edge[1] == node]
            evidence_value = [list(np.sort(self.flai_dataset.data[e].unique())) for e in evidence]
                
            evidence_combination = evidence_value
            evidence_combination = list(itertools.product(*evidence_combination))
            list_probas = []
            for ec in evidence_combination:
                filters = [[filter[0],filter[1]]  for filter in zip(evidence,ec)]
                data_aux =  self.flai_dataset.data
                for f in filters:
                    if f[0] not in sensible_feature :
                        data_aux = data_aux[data_aux[f[0]] == f[1]]
                #if node in sensible_feature:
                #    list_probas = list_probas + [[0.5 for nv in node_value]]
                else:
                    list_probas = list_probas + [[0.5 if math.isnan((data_aux[node] == nv).sum() / len(data_aux[node] == nv)) else (data_aux[node] == nv).sum() / len(data_aux[node] == nv) for nv in node_value]]
            list_cpd = list_cpd + [TabularCPD(node, len(node_value), np.array(list_probas).T.tolist(),
                        evidence= evidence,
                        evidence_card= [len(ev) for ev in evidence_value])]
        self.graph =  bn.make_DAG(list(self.graph['model_edges']), CPD=list_cpd,verbose= verbose)
        