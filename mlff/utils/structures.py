from collections import namedtuple


System = namedtuple("System", ("R", "Z", "cell", "total_charge"))
Graph = namedtuple("Graph", ("positions", "edges", "nodes", "centers", "others", "mask", "total_charge", "partial_charges", "i_pairs", "j_pairs"))
Neighbors = namedtuple("Neighbors", ("centers", "others", "overflow", "reference_positions"))
PrimitiveNeighbors = namedtuple("PrimitiveNeighbors", ("idx_i", "idx_j", "shifts", "overflow"))
