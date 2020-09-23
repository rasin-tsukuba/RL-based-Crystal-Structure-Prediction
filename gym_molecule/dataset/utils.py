# %%
import json
import os
from glob import glob
from itertools import combinations
from math import sqrt

import networkx as nx
import pandas as pd
import plotly.graph_objs as go
from openbabel import openbabel as ob
from rdkit import Chem

cpk_colors = dict(Ar='cyan', B='salmon', Ba='darkgreen', Be='darkgreen', Br='darkred', C='black', Ca='darkgreen',
                  Cl='green', Cs='violet', F='green', Fe='darkorange', Fr='violet', H='white', He='cyan',
                  I='darkviolet', K='violet', Kr='cyan', Li='violet', Mg='darkgreen', N='blue', Na='violet', Ne='cyan',
                  O='red', P='orange', Ra='darkgreen', Rb='violet', S='yellow', Sr='darkgreen', Ti='gray', Xe='cyan')

cpk_color_rest = 'pink'

atomic_radii = dict(Ac=1.88, Ag=1.59, Al=1.35, Am=1.51, As=1.21, Au=1.50, B=0.83, Ba=1.34, Be=0.35, Bi=1.54, Br=1.21,
                    C=0.68, Ca=0.99, Cd=1.69, Ce=1.83, Cl=0.99, Co=1.33, Cr=1.35, Cs=1.67, Cu=1.52, D=0.23, Dy=1.75,
                    Er=1.73, Eu=1.99, F=0.64, Fe=1.34, Ga=1.22, Gd=1.79, Ge=1.17, H=0.23, Hf=1.57, Hg=1.70, Ho=1.74,
                    I=1.40, In=1.63, Ir=1.32, K=1.33, La=1.87, Li=0.68, Lu=1.72, Mg=1.10, Mn=1.35, Mo=1.47, N=0.68,
                    Na=0.97, Nb=1.48, Nd=1.81, Ni=1.50, Np=1.55, O=0.68, Os=1.37, P=1.05, Pa=1.61, Pb=1.54, Pd=1.50,
                    Pm=1.80, Po=1.68, Pr=1.82, Pt=1.50, Pu=1.53, Ra=1.90, Rb=1.47, Re=1.35, Rh=1.45, Ru=1.40, S=1.02,
                    Sb=1.46, Sc=1.44, Se=1.22, Si=1.20, Sm=1.80, Sn=1.46, Sr=1.12, Ta=1.43, Tb=1.76, Tc=1.35, Te=1.47,
                    Th=1.79, Ti=1.47, Tl=1.55, Tm=1.72, U=1.58, V=1.33, W=1.37, Y=1.78, Yb=1.94, Zn=1.45, Zr=1.56)

def letter(inp):
    return ''.join(filter(str.isalpha, inp))


def cif_parser(cif_dir, json_dir):
    path = sorted(glob(cif_dir))
    for i in path:
        data = {}
        name = i.split('/')[-1].split('\\')[-1].split('.')[0]
        print(name)
        data['name'] = name
        data['info'] = {}
        data['cell'] = {}
        data['atoms'] = []
        data['bonds'] = []
        loop = 0
        for line in open(i, 'r'):
            line = ' '.join(line.split())
            # print(line)
            if line == 'loop_':
                loop += 1
            if loop == 0:
                if line.startswith('_audit_creation_date'):
                    data['info']['creation_date'] = line.split(' ')[-1]
            if loop == 1:
                if line.startswith('_cell_length_a'):
                    data['cell']['a'] = float(line.split(' ')[-1])
                if line.startswith('_cell_length_b'):
                    data['cell']['b'] = float(line.split(' ')[-1])
                if line.startswith('_cell_length_c'):
                    data['cell']['c'] = float(line.split(' ')[-1])
                if line.startswith('_cell_angle_alpha'):
                    data['cell']['alpha'] = float(line.split(' ')[-1])
                if line.startswith('_cell_angle_beta'):
                    data['cell']['beta'] = float(line.split(' ')[-1])
                if line.startswith('_cell_angle_gamma'):
                    data['cell']['gamma'] = float(line.split(' ')[-1])
            if loop == 2:
                if len(line.split(' ')) == 8:
                    # print(line, len(line.split(' ')))
                    atom_info = {'label': line.split(' ')[0],
                                 'type_symbol': line.split(' ')[1],
                                 'x': float(line.split(' ')[2]),
                                 'y': float(line.split(' ')[3]),
                                 'z': float(line.split(' ')[4])
                                 }
                    data['atoms'].append(atom_info)

            if loop == 3:
                if len(line.split(' ')) == 5:
                    bond_info = {'first_atom': line.split(' ')[0],
                                 'second_atom': line.split(' ')[1],
                                 'length': line.split(' ')[2]
                                 }
                    data['bonds'].append(bond_info)

        data2 = json.dumps(data)
        file = open(json_dir + name.strip() + '.json', 'w')
        file.write(data2)


def convert_to_cartesian(src_dir, out_dir):
    path = sorted(glob(src_dir))
    for jfile in path:
        filename = jfile.split('/')[-1].split('\\')[-1].split('.')[0].strip()
        # print(filename)
        # break
        f = open(jfile, 'r')
        data = json.load(f)
        x, y, z = 0, 0, 0
        for i in data['atoms']:
            i['x'] = data['cell']['a'] * i['x']
            i['y'] = data['cell']['b'] * i['y']
            i['z'] = data['cell']['c'] * i['z']

        for i in data['atoms']:
            if i['label'].startswith('P'):
                x, y, z = i['x'], i['y'], i['z']
                break
        for i in data['atoms']:
            i['x'] = round(i['x'] - x, 6)
            i['y'] = round(i['y'] - y, 6)
            i['z'] = round(i['z'] - z, 6)

        data2 = json.dumps(data)
        print(out_dir + filename.strip() + '.json')
        file = open(out_dir + filename.strip() + '.json', 'w')
        file.write(data2)


# %%
def mol_to_nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   # returns the atom's index (ordering in the molecule) 
                   symbol=atom.GetSymbol(),
                   # returns the atomic symbol (a string)
                   formal_charge=atom.GetFormalCharge(),
                   # returns the formal charge 
                   # a formal charge is the charge assigned to an 
                   # atom in a molecule, assuming that electrons 
                   # in all chemical bonds are shared equally 
                   # between atoms, regardless of relative
                   # electronegativity
                   implicit_valence=atom.GetImplicitValence(),
                   # Returns the number of implicit Hs on the atom
                   ring_atom=atom.IsInRing(),
                   # Returns whether or not the atom is in a ring
                   degree=atom.GetDegree(),
                   # Returns the degree of the atom in the molecule.
                   # The degree of an atom is defined to be its number 
                   # of directly-bonded neighbors. The degree is 
                   # independent of bond orders, but is dependent
                   # on whether or not Hs are explicit in the 
                   # graph.
                   hybridization=atom.GetHybridization()
                   # Returns the atom’s hybridization.
                   # Hybridization is the idea that atomic orbitals 
                   # fuse to form newly hybridized orbitals, which in 
                   # turn, influences molecular geometry and bonding 
                   # properties.
                   )
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   # Returns the index of the bond’s first atom.
                   bond.GetEndAtomIdx(),
                   # Returns the index of the bond’s first atom.
                   bond_type=bond.GetBondType()
                   # Set the type of the bond as a BondType
                   )
    return G


# %%
def nx_to_mol(G):
    mol = Chem.RWMol()
    # The RW molecule class (read/write)
    atomic_nums = nx.get_node_attributes(G, 'atomic_num')
    # The number of atom
    chiral_tags = nx.get_node_attributes(G, 'chiral_tag')
    # 
    formal_charges = nx.get_node_attributes(G, 'formal_charge')
    # The formal charges
    node_is_aromatics = nx.get_node_attributes(G, 'is_aromatic')
    # Aromatics are hydrocarbons, organic compounds that consist 
    # exclusively of the elements carbon and hydrogen 
    node_hybridizations = nx.get_node_attributes(G, 'hybridization')
    # hybridization
    num_explicit_hss = nx.get_node_attributes(G, 'num_explicit_hs')
    # the number of explicit hss
    node_to_idx = {}

    for node in G.nodes():
        a = Chem.Atom(atomic_nums[node])
        a.SetChiralTag(chiral_tags[node])
        #
        a.SetFormalCharge(formal_charges[node])
        # 
        a.SetIsAromatic(node_is_aromatics[node])
        # 
        a.SetHybridization(node_hybridizations[node])
        # Sets the hybridization of the atom.
        a.SetNumExplicitHs(num_explicit_hss[node])
        # 
        idx = mol.AddAtom(a)
        node_to_idx[node] = idx

    bond_types = nx.get_edge_attributes(G, 'bond_type')
    print(bond_types)
    for edge in G.edges():
        first, second = edge
        ifirst = node_to_idx[first]
        isecond = node_to_idx[second]
        bond_type = bond_types[first, second]
        mol.AddBond(ifirst, isecond, bond_type)

    Chem.SanitizeMol(mol)
    # Kekulize?, check valencies, set aromaticity, 
    # conjugation and hybridization
    return mol


# %%
def cif_to_mol(path, out_path):
    section = path[0].split('/')[-2]
    if not os.path.exists(out_path + section):
        os.mkdir(out_path + section)
    for i in path:
        filename = i.split('/')[-1].split('.')[0]
        obConversion = ob.OBConversion()
        obConversion.SetInAndOutFormats("cif", "mol")

        mol = ob.OBMol()
        obConversion.ReadFile(mol, i)

        print(mol.NumAtoms())
        outMDL = obConversion.WriteFile(mol, out_path + section + '/' + filename + '.mol')
        print(out_path + section + '/' + filename + '.mol')
        print(mol.AddHydrogens())
        print(mol.NumAtoms())


# %%
def load_dataset(path):
    df = pd.read_csv(path, header=None, names=['path'])
    return df


class crystal_dataset:
    def __init__(self, path):
        super().__init__()
        self.data = load_dataset(path)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        print(self.data[item])
        mol = Chem.MolFromMolFile(self.data[item])
        return mol


class CrystalGraph:
    """
    Represents a Crystal Graph.
    """

    __slots__ = [
        'name',
        'elements',
        'x',
        'y',
        'z',
        'adj_list',
        'bond_lengths'
    ]

    def __init__(self):
        self.name = ""
        self.elements = []
        self.x = []
        self.y = []
        self.z = []
        self.adj_list = {}
        self.bond_lengths = {}

    def __getitem__(self, item):
        # print(item)
        if isinstance(item, int):
            return self.elements[item], (self.x[item], self.y[item], self.z[item])
        else:
            position = self.elements.index(item)
            return self.elements[position], (self.x[position], self.y[position], self.z[position])

    def __len__(self):
        return len(self.elements)

    def add_adj_list(self, i, j, distance):
        self.adj_list.setdefault(i, set()).add(j)
        self.adj_list.setdefault(j, set()).add(i)
        self.bond_lengths[frozenset([i, j])] = round(distance, 3)

    def read_json(self, file_path: str):
        """
        Read an CIF file, searches for elements and their
        cartesian coordinates
        :param file_path:
        :return:
        """
        with open(file_path) as file:
            data = json.load(file)

            self.name = data["name"]
            for i in data["atoms"]:
                self.elements.append(i["label"])
                self.x.append(i["x"])
                self.y.append(i["y"])
                self.z.append(i["z"])

            for i in data["bonds"]:
                self.adj_list.setdefault(i["first_atom"], set()).add(i["second_atom"])
                self.adj_list.setdefault(i["second_atom"], set()).add(i["first_atom"])
                self.bond_lengths[frozenset([i["first_atom"], i["second_atom"]])] = i["length"]

        for i, j in combinations(self.elements, 2):
            # print(i, j)
            # additional H-O bond
            if i.startswith('H') and j.startswith('O') or i.startswith('O') and j.startswith('H'):
                # print(i, j)
                x_i, y_i, z_i = self.__getitem__(i)[1]
                x_j, y_j, z_j = self.__getitem__(j)[1]
                distance = sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2 + (z_i - z_j) ** 2)

                # a O-H specific rule
                if 0.9 < distance < 1.1 or 1.4 < distance < 2:
                    # print(self.adj_list, self.bond_lengths)
                    if i in self.adj_list and j in self.adj_list:
                        if j in self.adj_list[i] \
                                or i in self.adj_list[j] \
                                or frozenset([i, j]) in self.bond_lengths\
                                or frozenset([j, i]) in self.bond_lengths:
                            continue
                        else:
                            self.add_adj_list(i, j, distance)
                    else:
                        self.add_adj_list(i, j, distance)
        # print(self.adj_list, self.bond_lengths)

    def edges(self):
        """
        creates an iterator with all graph edges.
        :return:
        """

        edges = set()

        for node, neighbours in self.adj_list.items():
            for neighbour in neighbours:
                edge = frozenset([node, neighbour])
                if edge in edges:
                    continue
                edges.add(edge)

                yield node, neighbour


def to_plotly_figure(graph: CrystalGraph) -> go.Figure:
    """
    Create a Plotly Figure
    :param graph:
    :return:
    """

    def atom_trace():
        """
        Creates an atom trace for the plot
        :return:
        """

        colors = [cpk_colors.get(letter(element), cpk_color_rest) for element in graph.elements]
        sizes = []
        for element in graph.elements:
            sizes.append(atomic_radii.get(letter(element)) * 40)

        markers = dict(color=colors,
                       line=dict(color='lightgray',
                                 width=2),
                       size=sizes,
                       symbol='circle',
                       opacity=0.8)
        trace = go.Scatter3d(
            x=graph.x,
            y=graph.y,
            z=graph.z,
            mode='markers',
            marker=markers,
            text=graph.elements
        )

        return trace

    def bond_trace():
        """
        Creates a bond trace for the plot.
        :return:
        """

        trace = go.Scatter3d(
            x=[],
            y=[],
            z=[],
            hoverinfo='none',
            mode='lines',
            marker=dict(color='gray',
                        size=3,
                        opacity=0.1)
        )

        adjacent_atoms = (
            (atom, neighbour) for atom, neighbours in graph.adj_list.items()
            for neighbour in neighbours)

        for i, j in adjacent_atoms:
            x_i, y_i, z_i = graph.__getitem__(i)[1]
            x_j, y_j, z_j = graph.__getitem__(j)[1]
            trace['x'] += (x_i, x_j, None)
            trace['y'] += (y_i, y_j, None)
            trace['z'] += (z_i, z_j, None)
        return trace

    # print(element for element in graph)
    annotations_elements = [
        dict(text=element,
             x=x,
             y=y,
             z=z,
             showarrow=False,
             yshift=15)
        for element, (x, y, z) in graph
    ]

    annotations_indices = [
        dict(text=number,
             x=x,
             y=y,
             z=z,
             showarrow=False,
             yshift=15)
        for number, (_, (x, y, z)) in enumerate(graph)
    ]

    annotations_bonds = []
    for (i, j), length in graph.bond_lengths.items():
        #print(i, j, length)
        x_i, y_i, z_i = graph.__getitem__(i)[1]
        x_j, y_j, z_j = graph.__getitem__(j)[1]
        x = (x_i + x_j) / 2
        y = (y_i + y_j) / 2
        z = (z_i + z_j) / 2
        annotations_bonds.append(
            dict(
                text=length,
                x=x,
                y=y,
                z=z,
                showarrow=False,
                yshift=15,
                font=dict(color="steelblue")
            )
        )

    updatemenus = list([
        dict(buttons=list([
            dict(label='Elements',
                 method='relayout',
                 args=[{
                     'scene.annotations': annotations_elements
                 }]),
            dict(label='Element & Bond Lengths',
                 method='relayout',
                 args=[{
                     'scene.annotations': annotations_elements + annotations_bonds
                 }]),
            dict(label='Indices',
                 method='relayout',
                 args=[{
                     'scene.annotations': annotations_indices
                 }]),
            dict(label='Indices & Bond Lengths',
                 method='relayout',
                 args=[{
                     'scene.annotations': annotations_indices + annotations_bonds
                 }]),
            dict(label='Bond Lengths',
                 method='relayout',
                 args=[{
                     'scene.annotations': annotations_bonds
                 }]),
            dict(label='Hide All',
                 method='relayout',
                 args=[{
                 }])
        ]),
            direction='down',
            xanchor='left',
            yanchor='top'
        ),
    ])

    data = [atom_trace(), bond_trace()]
    axis_params = dict(
        showgrid=True,
        showbackground=True,
        showticklabels=True,
        zeroline=True,
        titlefont=dict(color='white')
    )
    layout = dict(
        scene=dict(
            xaxis=axis_params,
            yaxis=axis_params,
            zaxis=axis_params,
            annotations=annotations_elements
        ),
        margin=dict(
            r=0,
            l=0,
            b=0,
            t=0
        ),
        showlegend=False,
        updatemenus=updatemenus
    )
    figure = go.Figure(data=data, layout=layout)

    return figure


def to_networkx_graph(graph: CrystalGraph) -> nx.Graph:
    """
    Creates a NetworkX graph
    Atomic elements and coordinates are added to the graph
    as node attributes 'element' and 'xyz" respectively.
    Bond lengths are added to the graph as edge attribute 'length'
    :param graph:
    :return:
    """

    G = nx.Graph(graph.adj_list)
    node_attrs = {
        num: {
            'element': element,
            'xyz': xyz
        } for num, (element, xyz) in enumerate(graph)
    }
    nx.set_node_attributes(G, node_attrs)
    edge_attrs = {
        edge: {
            'length': length
        } for edge, length in graph.bond_lengths.items()
    }
    nx.set_edge_attributes(G, edge_attrs)
    return G


# %%
#

# if __name__ == '__main__':
#     from glob import glob
#
#     path = '../../Data/cartesian_json/pos/*.json'
#     for i in sorted(glob(path)):
#
#         # Print the file name
#         print(i)
#
#         # Create the Crystal Graph object
#         mg = CrystalGraph()
#
#         # Read the json file
#         mg.read_json(i)
#
#         # Convert the molecular graph to the NetworkX graph
#         G = to_networkx_graph(mg)
#         print(G, G.nodes, G.edges)
#
#         break


# import matplotlib.pyplot as plt
# path = sorted(glob('/home/rasin/Workspace/Project/Crystal/Data/cif/monomer/*.cif'))
# cif_to_mol(path, out_path='/home/rasin/Workspace/Project/Crystal/Data/mol/')
# print("Done")

# for i in sorted(glob('/home/rasin/Workspace/Project/Crystal/Data/mol/pos/*.mol')):
#     m = Chem.MolFromMolFile(i, removeHs=False)
#     print(i.split('/')[-1])
#     print(m.GetAtoms)
#     G = mol_to_nx(m)
#     nx.draw(G, cmap=plt.get_cmap('jet'))
#     plt.show()
#     print(list(G.degree))
# %%
# df = load_dataset('/home/rasin/Workspace/Project/Crystal/Data/mol/mol.csv')
# %%
