from collections import namedtuple


System = namedtuple("System", ("R", "Z", "cell"))
Graph = namedtuple('Graph', ('edges', 'nodes', 'centers', 'others', 'mask'))
Neighbors = namedtuple("Neighbors", ("centers", "others", "overflow", "reference_positions"))
PrimitiveNeighbors = namedtuple("PrimitiveNeighbors", ("idx_i", "idx_j", "shifts", "overflow"))
