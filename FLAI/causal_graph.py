from FLAI import data
import bnlearn as bn
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.factors.discrete import TabularCPD
import numpy as np
import itertools
import math
class CausalGraph():
    """
    The CausalGraph class represents a causal Bayesian network and provides methods for its manipulation, 
    inference, mitigation, and learning.
    """
    def __init__(self, flai_dataset = None, node_edge = None, CPD = None, indepence_test = True,
                 root_node = None, target = None,  verbose = 0):
        
        """
        Initialize a CausalGraph object.
        
        Args:
        flai_dataset (flai.data.Data, optional): Dataset to be used for the network. Default is None.
        node_edge (list of tuples, optional): List of tuples where each tuple represents an edge of the network. Default is None.
        CPD (list of TabularCPD objects, optional): List of conditional probability distributions for the network. Default is None.
        indepence_test (bool, optional): If true, performs an independence test on the network. Default is True.
        root_node (str, optional): The root node of the network. Default is None.
        target (str, optional): The target node of the network. Default is None.
        verbose (int, optional): Verbosity level. Default is 0.
        """

        if flai_dataset is None and node_edge is None:
            raise Exception("Data or edges should be provided")
        if target is None:
            raise Exception("Please indicate the target feature")
        if node_edge is None:
            if not isinstance(flai_dataset,data.Data):
                raise Exception("Data should be a FLAI.data.Data class")
            self.flai_dataset = flai_dataset
            if root_node is None:
                self.graph = bn.structure_learning.fit(flai_dataset.data, methodtype='hc', scoretype='bic',verbose= verbose)
            else:
                self.graph = bn.structure_learning.fit(flai_dataset.data,root_node= root_node,verbose= verbose)
            self.graph['model_edges'] = [(edge[0],edge[1]) if edge[0] != target else (edge[1],edge[0]) for edge in self.graph['model_edges']]
            list_edges = self.graph['model_edges']
            ends = list((np.array(self.graph['model_edges'])[:,0]))
            for n in list(self.graph['model'].nodes()):
                if (n not in ends) & (n != target):
                    print(n)
                    list_edges = list_edges + [(n,target)]
            self.graph['model_edges'] = list_edges
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
        """
        Performs an independence test on the network.
        
        Args:
        flai_dataset (flai.data.Data): Dataset to be used for the independence test.
        test (str, optional): Type of test to be performed. Default is 'chi_square'.
        prune (bool, optional): If true, prunes the network after the test. Default is True.
        verbose (int, optional): Verbosity level. Default is 0.
        """

        self.graph = bn.independence_test(self.graph, flai_dataset.data, test, prune,verbose= verbose)

    def inference(self,variables = [],evidence = {},verbose=0):
        """
        Performs inference on the network given evidence.

        Args:
        variables (list, optional): List of variables to be included in the inference. Default is an empty list.
        evidence (dict, optional): Dictionary of evidence in the form {variable: value}. Default is an empty dictionary.
        """

        if len(variables) == 0:
            raise Exception("Variables should be provided") 
        return bn.inference.fit(self.graph, variables=variables, evidence=evidence,verbose = verbose).df          

    def predict(self, data = None, variables = [],verbose=0):
        """
        Makes predictions based on the network.

        Args:
        data (DataFrame, optional): Data used for making predictions. Default is None.
        variables (list, optional): List of variables to be predicted. Default is an empty list.
        """
        if len(variables) == 0:
            raise Exception("Variables should be provided") 
        if data is None:
            raise Exception("Data should be provided") 
        return bn.predict(self.graph, data, variables = variables, verbose=verbose)
    
    def learn_cpd(self, flai_dataset = None, methodtype= None,verbose = 0):
        """
        Learn the Conditional Probability Distribution (CPD) for the network.

        Args:
        flai_dataset (flai.data.Data, optional): Dataset to be used for learning the CPD. Default is None.
        methodtype (str, optional): The method type to be used for learning. Default is None.
        verbose (int, optional): Verbosity level. Default is 0.
        """

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
        """
        Calculate the Conditional Probability Distribution (CPD) for the network.

        Args:
        verbose (int, optional): Verbosity level. Default is 0.
        """

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
        """
        Plots the network.

        Args:
        directed (bool, optional): If True, plots a directed graph. Default is True.
        """

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
        """
        Returns the Conditional Probability Distributions (CPDs) of the network.
        """

        CPDs = bn.print_CPD(self.graph,verbose = 0)
        return CPDs

    def generate_dataset(self, n_samples = 1000, methodtype = 'bayes', verbose = 0):
        """
        Generates a dataset based on the network.

        Args:
        n_samples (int, optional): Number of samples to generate. Default is 1000.
        methodtype (str, optional): The method type to be used for generation. Default is 'bayes'.
        verbose (int, optional): Verbosity level. Default is 0.
        """

        df = bn.sampling(self.graph, n= n_samples, methodtype = methodtype, verbose = verbose)
        return data.Data(df, transform=False)

    def mitigate_edge_relation(self, sensible_feature = []):
        """
        Mitigates the relationship of certain edges in the network.

        Args:
        sensible_feature (list, optional): List of features for which to mitigate the relationship. Default is an empty list.
        """

        ### improve for more than one sensible feature 
        if len(sensible_feature) == 0:
            raise Exception("Sensible features should be provided")
        reverse_edge = []
        ### first apply transitivity and reverse
        maintain_edge = self.graph['model_edges']
        longitud = -1
        while(longitud!=0):
            model_edges = maintain_edge
            maintain_edge = [e for e in model_edges if ((e[1] not in  sensible_feature) & (e[0] in sensible_feature)) | ((e[1] not in sensible_feature) & (e[0] not in sensible_feature))] 
            #print('maintain',maintain_edge)
            longitud = 0

            for sf in sensible_feature:
                impact_sensible = [e for e in model_edges if e[1] == sf ]
                longitud = longitud + len(impact_sensible)
                #print(impact_sensible)
                for i_s in impact_sensible:
                    sensible_impact = [e for e in model_edges if e[0] == sf ]
                    #print(sensible_impact)
                    transitivity_edge = [(i_s[0],e[1]) for e in model_edges if (i_s[0] != e[1]) & (e[0] == sf) & ((i_s[0],e[1]) not in reverse_edge) ]
                    #print('Transitivity',transitivity_edge)
                    maintain_edge = maintain_edge + transitivity_edge
                    if ((i_s[0], 'label') in maintain_edge) & ((sf, i_s[0]) not in reverse_edge):
                        maintain_edge = maintain_edge + [(sf, i_s[0])]
                        reverse_edge = reverse_edge + [(i_s[0],sf)]
            maintain_edge_aux = maintain_edge
            maintain_edge = []
            for mea in maintain_edge_aux:
                if mea not in maintain_edge:
                    maintain_edge = maintain_edge + [mea]
            #print(longitud)
            #print('maintain',maintain_edge)
        self.graph['model_edges'] = maintain_edge
        return maintain_edge
    def mitigate_calculation_cpd(self,verbose = 0,sensible_feature = [],fix_proportion = True):
        """
        Mitigates the calculation of the Conditional Probability Distribution (CPD) for certain features.

        Args:
        verbose (int, optional): Verbosity level. Default is 0.
        sensible_feature (list, optional): List of sensible features for which to mitigate the CPD calculation. Default is an empty list.
        """
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
                if (node in sensible_feature) & fix_proportion:
                    list_probas = list_probas + [[1/len(node_value) for nv in node_value]]
                else:
                    list_probas = list_probas + [[0.5 if math.isnan((data_aux[node] == nv).sum() / len(data_aux[node] == nv)) else (data_aux[node] == nv).sum() / len(data_aux[node] == nv) for nv in node_value]]
            list_cpd = list_cpd + [TabularCPD(node, len(node_value), np.array(list_probas).T.tolist(),
                        evidence= evidence,
                        evidence_card= [len(ev) for ev in evidence_value])]
        self.graph =  bn.make_DAG(list(self.graph['model_edges']), CPD=list_cpd,verbose= verbose)
        