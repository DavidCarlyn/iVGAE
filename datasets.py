import numpy as np
import torch
from torch_geometric.data import Data

def create_adjacency_matrix(nodes, edges, interventions):
    edge_index = []
    for src, dest in edges:
        if dest in interventions.keys(): continue
        edge_index.append([nodes.index(src), nodes.index(dest)])
    
    return np.array(edge_index).T

def run_probabilities(num_samples, prob):
    return (np.random.rand(num_samples) < prob).astype(np.uint8)

class ASIA:
    """
    See paper: 
    Title: Local Computations with Probabilities on Graphical Structures and Their Application to Expert Systems
    Authors: Lauritzen S, Spiegelhalter D
    Year: 1988

    A -> T, S -> L, S -> B, T -> E, L -> E, B -> D, E -> X, E -> D

    p(A) = 0.01                 p(E | L, T) = 1.0
                                p(E | L, ~T) = 1.0
    p(T | A) = 0.05             p(E | ~L, T) = 1.0
    p(T | ~A) = 0.01            p(E | ~L, ~T) = 0.0

    p(S) = 0.50                 p(X | E) = 0.98
                                p(X | ~E) = 0.05
    p(L | S) = 0.10
    p(L | ~S) = 0.01            p(D | E, B) = 0.90
                                p(D | E, ~B) = 0.70
    p(B | S) = 0.60             p(D | ~E, B) = 0.80
    p(B | ~S) = 0.30            p(D | ~E, ~B) = 0.10
    """
    def __init__(self, num_samples=10000, interventions={}, transform=None):
        super().__init__()
        self.nodes = ['A', 'S', 'T', 'L', 'B', 'E', 'X', 'D']
        self.edges = [
            ['A', 'T'],
            ['S', 'L'],
            ['S', 'B'],
            ['T', 'E'],
            ['L', 'E'],
            ['B', 'D'],
            ['E', 'X'],
            ['E', 'D']
        ]
        self.edge_index = create_adjacency_matrix(self.nodes, self.edges, interventions)
        self.graphs = []

        # Visit to Asia
        if 'A' in interventions.keys():
            A = run_probabilities(num_samples, interventions['A'])
        else:
            A = run_probabilities(num_samples, 0.01)
        
        # Is a smoker
        if 'S' in interventions.keys():
            S = run_probabilities(num_samples, interventions['S'])
        else:
            S = run_probabilities(num_samples, 0.50)
        
        # Has Tuberculosis
        if 'T' in interventions.keys():
            T = run_probabilities(num_samples, interventions['T'])
        else:
            A_mask = A == 1
            T = np.zeros(num_samples)
            T[A_mask] = run_probabilities(A_mask.sum(), 0.05)
            T[~A_mask] = run_probabilities((~A_mask).sum(), 0.01)

        # Has Lung Cancer
        if 'L' in interventions.keys():
            L = run_probabilities(num_samples, interventions['L'])
        else:
            S_mask = S == 1
            L = np.zeros(num_samples)
            L[S_mask] = run_probabilities(S_mask.sum(), 0.10)
            L[~S_mask] = run_probabilities((~S_mask).sum(), 0.01)
        
        # Has Bronchitis
        if 'B' in interventions.keys():
            B = run_probabilities(num_samples, interventions['B'])
        else:
            S_mask = S == 1
            B = np.zeros(num_samples)
            B[S_mask] = run_probabilities(S_mask.sum(), 0.60)
            B[~S_mask] = run_probabilities((~S_mask).sum(), 0.30)

        # Has either Tuberculosis or Lung Cancer
        if 'E' in interventions.keys():
            E = run_probabilities(num_samples, interventions['E'])
        else:
            T_mask = T == 1
            L_mask = L == 1
            E = np.zeros(num_samples)
            mask = T_mask & L_mask
            E[mask] = run_probabilities(mask.sum(), 1.0)
            mask = T_mask & ~L_mask
            E[mask] = run_probabilities(mask.sum(), 1.0)
            mask = ~T_mask & L_mask
            E[mask] = run_probabilities(mask.sum(), 1.0)
            mask = ~T_mask & ~L_mask
            E[mask] = run_probabilities(mask.sum(), 0.0)

        # Has a Positive X-ray
        if 'X' in interventions.keys():
            X = run_probabilities(num_samples, interventions['X'])
        else:
            E_mask = E == 1
            X = np.zeros(num_samples)
            X[E_mask] = run_probabilities(E_mask.sum(), 0.98)
            X[~E_mask] = run_probabilities((~E_mask).sum(), 0.05)

        # Has Dyspnoea
        if 'D' in interventions.keys():
            D = run_probabilities(num_samples, interventions['E'])
        else:
            E_mask = E == 1
            B_mask = B == 1
            D = np.zeros(num_samples)
            mask = E_mask & B_mask
            D[mask] = run_probabilities(mask.sum(), 0.90)
            mask = E_mask & ~B_mask
            D[mask] = run_probabilities(mask.sum(), 0.70)
            mask = ~E_mask & B_mask
            D[mask] = run_probabilities(mask.sum(), 0.80)
            mask = ~E_mask & ~B_mask
            D[mask] = run_probabilities(mask.sum(), 0.10)

        # ['A', 'S', 'T', 'L', 'B', 'E', 'X', 'D']
        self.nodes_features = torch.tensor(np.stack((A, S, T, L, B, E, X, D)).T, dtype=torch.float)
        self.edge_index = torch.tensor(self.edge_index, dtype=torch.long)
        def process(data):
            if transform is not None:
                return transform(data)
            return data

        self.data_list = [process(Data(x=x.unsqueeze(1), edge_index=self.edge_index)) for x in self.nodes_features]


class Earthquake:
    """
    See book:
    Title: Bayesian artificial intelligence
    Authors: Korb K, Nicholson A
    Year: 2010

    B -> A, E -> A, A -> J, A -> M

    p(B) = 0.001            p(A | B, E) = 0.95
                            p(A | B, ~E) = 0.94
    p(E) = 0.002            p(A | ~B, E) = 0.29
                            p(A | ~B, ~E) = 0.001

    p(J | A) = 0.90         p(M | A) = 0.07
    p(J | ~A) = 0.05        p(M | ~A) = 0.01
    
    """
    def __init__(self, num_samples=10000, interventions={}, transform=None):
        super().__init__()
        self.nodes = ['B', 'E', 'A', 'J', 'M']
        self.edges = [
            ['B', 'A'],
            ['E', 'A'],
            ['A', 'J'],
            ['A', 'M']
        ]
        self.edge_index = create_adjacency_matrix(self.nodes, self.edges, interventions)
        self.graphs = []

        # Burglary occurs
        if 'B' in interventions.keys():
            B = run_probabilities(num_samples, interventions['B'])
        else:
            B = run_probabilities(num_samples, 0.001)
        
        # Earthquake occurs
        if 'E' in interventions.keys():
            E = run_probabilities(num_samples, interventions['E'])
        else:
            E = run_probabilities(num_samples, 0.002)
        
        # Alarm sounds
        if 'A' in interventions.keys():
            A = run_probabilities(num_samples, interventions['A'])
        else:
            B_mask = B == 1
            E_mask = E == 1
            A = np.zeros(num_samples)
            mask = B_mask & E_mask
            A[mask] = run_probabilities(mask.sum(), 0.95)
            mask = B_mask & ~E_mask
            A[mask] = run_probabilities(mask.sum(), 0.94)
            mask = ~B_mask & E_mask
            A[mask] = run_probabilities(mask.sum(), 0.29)
            mask = ~B_mask & ~E_mask
            A[mask] = run_probabilities(mask.sum(), 0.001)

        # John calls
        if 'J' in interventions.keys():
            J = run_probabilities(num_samples, interventions['J'])
        else:
            A_mask = A == 1
            J = np.zeros(num_samples)
            J[A_mask] = run_probabilities(A_mask.sum(), 0.90)
            J[~A_mask] = run_probabilities((~A_mask).sum(), 0.05)
        
        # Mary calls
        if 'M' in interventions.keys():
            M = run_probabilities(num_samples, interventions['M'])
        else:
            A_mask = A == 1
            M = np.zeros(num_samples)
            M[A_mask] = run_probabilities(A_mask.sum(), 0.70)
            M[~A_mask] = run_probabilities((~A_mask).sum(), 0.01)
        
        # ['B', 'E', 'A', 'J', 'M']
        self.nodes_features = torch.tensor(np.stack((B, E, A, J, M)).T, dtype=torch.float)
        self.edge_index = torch.tensor(self.edge_index, dtype=torch.long)
        def process(data):
            if transform is not None:
                return transform(data)
            return data

        self.data_list = [process(Data(x=x.unsqueeze(1), edge_index=self.edge_index)) for x in self.nodes_features]
