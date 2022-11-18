from FLAI import data
import bnlearn as bn
import networkx as nx
import matplotlib.pyplot as plt
class CausalGraph():
    def __init__(self, flai_dataset = None, node_edge = None, CPD = None, indepence_test = True,
                 verbose = 0):
        if flai_dataset is None and node_edge is None:
            raise Exception("Data or edges should be provided")
        if node_edge is None:
            if not isinstance(flai_dataset,data.Data):
                raise Exception("Data should be a FLAI.data.Data class")
            self.flai_dataset = flai_dataset
            self.graph = bn.structure_learning.fit(flai_dataset.data,verbose= verbose)
            if indepence_test:
                self.graph = self.independence_test(self.graph, flai_dataset, test='chi_square', prune=True,verbose= verbose)
        else:
            if CPD is None:
                self.flai_dataset = flai_dataset
                self.graph = bn.make_DAG(node_edge, verbose= verbose)
            else:
                self.flai_dataset = flai_dataset
                self.graph = bn.make_DAG(node_edge, CPD=CPD,verbose= verbose)


    def independence_test(self, graph, flai_dataset, test='chi_square', prune=True,verbose= 0):
        graph = bn.independence_test(graph, flai_dataset.data, test, prune,verbose= verbose)
        return graph

    def learn_cpd(self, flai_dataset = None, verbose = 0):
        if flai_dataset is None and self.flai_dataset is None:
            raise Exception("Data should be provided")
        if flai_dataset is None: 
            self.graph = bn.parameter_learning.fit(self.graph, self.flai_dataset.data,verbose= verbose) 
        else:
            if not isinstance(flai_dataset,data.Data):
                raise Exception("Data should be a FLAI.data.Data class")
            self.flai_dataset = flai_dataset
            self.graph = bn.parameter_learning.fit(self.graph, self.flai_dataset.data,verbose= verbose) 


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
    
    def get_CPDs(self):
        CPDs = bn.print_CPD(self.graph,verbose = 0)
        return CPDs