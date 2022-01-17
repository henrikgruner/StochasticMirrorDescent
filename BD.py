from collections import OrderedDict
import numpy as np
import torch
from models import ResNet18
import matplotlib.pyplot as plt
from tabulate import tabulate
from texttable import Texttable
from collections import defaultdict
import latextable


norm = np.linalg.norm


def psi(x, p):
    """
    p-norm to the power of p:
    1/p * ||x||_p ^p
    """
    return 1/p * norm(x, ord=p)**p


def grad_psi(x, p):
    """
    Gradient of p norm to the power of p:
    |x|^(p-1)*sgn(x)
    """
    return np.abs(x)**(p-1)*np.sign(x)


def bregman_divergence(p, x1, x2):
    """
    Bregman divergence with the potential p norm

    returns:
        psi(x_1) - psi(x_2)-gradient(psi)^T(x1-x2)
    where:
        psi = 1/p * ||x||_p ^p

    """
    return psi(x1, p) - psi(x2, p) - grad_psi(x2, p)@(x1-x2)


# Weights of network to a flat vector

def unpack_dict(PATH, network="ResNet18"):
    """
    Turns torch_state_dict into flat numpy array

    Parameters
    ----------
    PATH : string 
        Path to the state_dict

    Model: Pytorch NN module
        Either ResNet18 or Classifier from models.py 
    ----------

    Output:
    ----------
    Weights: np.array() 
        The flattened weights in numpy array
    ----------
    """

    # Classifier for MNIST
    if(network == "Classifier"):
        network = Classifier()
    # ResNet18 for CIFAR10
    if(network == "ResNet18"):
        network = ResNet18()

    init = torch.load(PATH, map_location=torch.device('cpu'))

    # When training with GPU, the returned dicts keys all start with module. which has to be removed
    new_state_dict = OrderedDict()
    for k, v in init.items():

        if(k[:7] == "module."):
            name = k[7:]  # remove `module.` from data.paralell.nn
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v

    network.load_state_dict(new_state_dict)
    layers = []
    for name, weight in network.named_parameters():
        layers.extend(weight.cpu().detach().numpy().flatten())

    return np.array(layers)


def find_all_BDs_from_intials_norm_q(q):
    """
    Fixed SMD with q norm.

    Parameters
    ----------
    q : int 
        for the q norm

    ----------

    Output:
    ----------
    for i initializations with j trained networks (SMD with q norm), the function
    finds the bregman divergence from the initialization i to each j trained netowork.

    The output is a dictionary with indexes as 1,2,3,4
    Every row has index name corresponding to the initialziation j

    ----------
    """

    weights = {}
    init = {}

    # For every initialization
    for i in [1, 2, 3, 4]:

        init[str(i)] = unpack_dict(
            'iteration-'+str(i)+'/Initialization-'+str(i))

        weights[i] = unpack_dict('iteration-'+str(i) +
                                 '/final_trained_network_q='+str(q))

    BDs = {"1": ["Init 1"], "2": ["Init 2"], "3": ["Init 3"], "4": ["Init 4"]}
    for i in [1, 2, 3, 4]:
        BDs["1"].append(bregman_divergence(q, weights[1], init[str(i)]))
        BDs["2"].append(bregman_divergence(q, weights[2], init[str(i)]))
        BDs["3"].append(bregman_divergence(q, weights[3], init[str(i)]))
        BDs["4"].append(bregman_divergence(q, weights[4], init[str(i)]))
    return BDs


def find_all_BDs_from_intials_norm_tables():
    """
    Converts the dictionaries obtained by
    find_all_BDs_from_initials_norm_q(q) for every q. 
    The result is turned into tables in latex form and printed. 
    """
    header = ["Initial", "Final 1", "Final 2", "Final 3", "Final 3"]
    for i in [1, 2, 3, 10]:
        BD = list(find_all_BDs_from_intials_norm_q(i).values())
        BD.insert(0, header)
        table = Texttable()
        table.set_cols_dtype(['t',  # int
                              'e',
                              'e',
                              'e',
                              'e'])
        table.set_cols_align(["c"] * len(BD[0]))
        table.set_deco(Texttable.HEADER | Texttable.VLINES)
        table.add_rows(BD)

        print('\nTexttable Latex:')
        print(latextable.draw_latex(
            table, caption="The Bregman Divergence for initalizatin " + str(i)))


def find_BD_for_all_norms(i):
    """
    Fixed initialization i.

    Parameters
    ----------
    i : int 
        The initialization i 
    ----------

    Output:
    ----------
    The function find the bregman divergences from the initialization point
    to each final poit obtained by SMD with q norm where q = 1,2,3,10. This is
    done for the bregman divergence with the potential function q norm where q = 1,2,3,4

    The output is on dictionary form, where:
    Lq is the key to each row
    each row has index in regards to the q norm

    """

    BDs = {"L1": ["L1", ], "L2": ["L2", ], "L3": ["L3", ], "L10": ["L10", ]}
    init = unpack_dict('iteration-'+str(i)+'/Initialization-'+str(i))

    for q in [1, 2, 3, 10]:
        weights = unpack_dict('iteration-4/final_trained_network_q='+str(q))

        BDs["L1"].append(bregman_divergence(1, weights, init))
        BDs["L2"].append(bregman_divergence(2, weights, init))
        BDs["L3"].append(bregman_divergence(3, weights, init))
        BDs["L10"].append(bregman_divergence(10, weights, init))
    return BDs


def find_all_BD_into_tables():
    """
    Converts the dictionaries obtained by
    find_BD_for_all_norms(i) for every intialization.

    The result is printed to latex form 
    """
    header = ["BD norm", "SMD-L1", "SMD-L2", "SMD-L3", "SMD-L10"]

    for i in [1, 2, 3, 4]:
        BD = list(find_BD_for_all_norms(i).values())
        BD.insert(0, header)
        table = Texttable()
        table.set_cols_dtype(['t',  # int
                              'e',
                              'e',
                              'e',
                              'e'])
        table.set_cols_align(["c"] * len(BD[0]))
        # table.set_precision(10)
        table.set_deco(Texttable.HEADER | Texttable.VLINES)
        table.add_rows(BD)
        print('\nTexttable Latex:')
        print(latextable.draw_latex(
            table, caption="The Bregman Divergence for initalizatin " + str(i)))


# These functions assume four initialization with four fully trained networks with SMD with q =1,2,3,10.
# Each initialization is in each own folder by the name 'Initialization-i' and the four finally trained networks on
# The given initialization is in the same folder by the name 'iteration-4/final_trained_network_q=q'.
# find_all_BD_into_tables()
# find_all_BDs_from_intials_norm_tables()
