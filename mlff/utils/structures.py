from collections import namedtuple


System = namedtuple("System", ("R", "Z", "cell", "total_charge"))
Graph = namedtuple("Graph", ("positions", "edges", "nodes", "centers", "others", "mask", "total_charge", "partial_charges", "idx_i_lr", "idx_j_lr"))
Neighbors = namedtuple("Neighbors", ("centers", "others", "overflow", "reference_positions"))
PrimitiveNeighbors = namedtuple("PrimitiveNeighbors", ("idx_i", "idx_j", "shifts", "overflow"))
