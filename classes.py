import numpy as np

class Node:

    """ Representation of a Node object. """

    def __init__(self, id, coord):
        self.id = id
        self.index = id - 1 # the index in a vector
        self.coord = coord

    def __repr__(self):
        """stdout representation of a node
        
        example: 
        >>> Node(1, 3.8)
        >>> (1, 3.8) 

        """
        return "({0}, {1})".format(self.id, self.coord)
        
class Element:

    """ Representation for a Element object. """

    def __init__(self, id, node1, node2):
        self.id  = id
        self.node1 = node1
        self.node2 = node2

    def __repr__(self):
        """stdout representation of an element
        
        example:
        >>> node1 = Node(1, 0)
        >>> node2 = Node(2, 0.5)
        >>> Element(1, node1, node2)
        >>> (1, (1, 0), (2, 0.5))

        """
        return "({0}, {1}, {2})".format(self.id, self.node1, self.node2)

class Condition:

    """ Representation of a neumann or dirichlet condition. """
    def __init__(self, node, value):
        self.node = node
        self.value = value

    def __repr__(self):
        """stdout representation of a condition

        example:
        >>> node = Node(1, 0)
        >>> Condition(node, 15)
        >>> node: 1, value: 15
        
        """
        return "node: {0}, value: {1}".format(self.node.id, self.value)
        

class Mesh:

    """Representation for a Mesh object. """

    def __init__(self, parameters, nodes, elements, dirichlet_condition, neumann_condition):
        self.parameters = parameters
        self.nodes = np.array(nodes)
        self.elements = np.array(elements)
        self.dirichlet_condition = dirichlet_condition
        self.neumann_condition = neumann_condition

