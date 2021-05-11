from classes import Node, Element, Mesh, Condition
import numpy as np

def read_condition(message):
    """ Reads a condition (neumann or dirichlet) from console

    :message: message to display on the console input

    :returns: [<node_id>::int, <value>::float]

    """
    [node, value] = input(message).split(" ")
    return [int(node), float(value)]

def read_mesh():
    """ Reads data from console to fill the mesh

    :returns: Mesh

    """

    l, k, Q = map(float, input("ingrese los parametros para el modelo separado por espacios (l k Q): ").split(" "))
    cant_nodes = int(input("ingrese la cantidad de nodos para la malla: "))

    # create list of nodes
    nodes = [Node(x, round(x * l, 1)) for x in range(1, cant_nodes + 1)]

    # create list of elements
    elements = [Element(x, n1, n2) for x, n1, n2 in zip(range(1, cant_nodes), nodes, nodes[1:])]

    # create dirichlet condition
    dirichlet_node, dirichlet_value = read_condition("ingrese la condicion de dirichlet separada por espacios (node_id value): ")
    dirichlet_condition = Condition(next(x for x in nodes if x.id == int(dirichlet_node)), float(dirichlet_value))

    # create neumann condition
    neumann_node, neumann_value = read_condition("ingrese la condicion de neumann separada por espacios (node_id value): ")
    neumann_condition = Condition(next(x for x in nodes if x.id == int(neumann_node)), float(neumann_value))

    # mesh data
    return Mesh([l, k, Q], nodes, elements, dirichlet_condition, neumann_condition)

def create_local_K(l, k):
    """ Creates the matrix of local K for FEM1D

    :l: the distance between each node
    :k: the value of k

    :returns: matrix with proper values

    """
    return np.matrix([[k / l, -k / l], [-k / l, k / l]])


def create_local_b(l, Q):
    """ Creates the vector of local b for FEM1D

    :l: the distance between each node 
    :Q: the value for Q

    :returns: a vector with proper values

    """
    return np.array([(Q * l) / 2, (Q * l) / 2])

def assembly(nodes, elements, localKs, localbs):
    """ Assembly K and b

    :nodes: the list of nodes
    :elements: the list of elements
    :localKs: the array of local Ks
    :localBs: the array of local bs

    :returns: K, b

    """
    num_nodes = len(nodes)

    # initialize K and b as zeroes
    K = np.zeros((num_nodes, num_nodes))
    b = np.zeros(num_nodes)

    for index, element in enumerate(elements):
        # get nodes from element
        node1 = element.node1
        node2 = element.node2

        # fill K
        localK = localKs[index]
        K[node1.index][node1.index] += localK[0][0]  
        K[node1.index][node2.index] += localK[0][1]
        K[node2.index][node1.index] += localK[1][0]
        K[node2.index][node2.index] += localK[1][1]

        #fill b
        local_b = localbs[index]
        b[node1.index] += local_b[0]
        b[node2.index] += local_b[1]

    return K, b

def apply_conditions(neumann_condition, dirichlet_condition, K, b):
    """ Apply neumann and dirichlet conditions to K and b

    :neumann_condition: the neumann_condition read from the mesh
    :dirichlet_condition: the dirichlet_condition read from the mesh
    :K: the global matrix K
    :b: the global vector b

    :returns: K, b

    """
    # temp variables
    temp_K = K
    temp_b = b

    # applying neumann
    neumann_node = neumann_condition.node
    neumann_value = neumann_condition.value
    temp_b[neumann_node.index] += neumann_value


    # applying dirichlet
    dirichlet_node = dirichlet_condition.node
    dirichlet_value = dirichlet_condition.value
    temp_K = np.delete(temp_K, dirichlet_node.index, 0) # delete from K, the object (index) from axis 0 (row)
    temp_b = np.delete(temp_b, dirichlet_node.index) # delete from b, the object (index)

    # pass value from column in K to b converted
    for index, row in enumerate(temp_K):
        cell = row[dirichlet_node.index]
        temp_b[index] += -1 * dirichlet_value * cell
    
    # delete column
    temp_K = np.delete(temp_K, dirichlet_node.index, 1) # delete from K, the object (index) from axis 1 (column)

    return temp_K, temp_b

def calculate_fem(K, b):
    """ Calculates the FEM value

    :K: global K
    :b: global b

    :returns: T

    """
    K_inv = np.linalg.inv(K)
    return K_inv.dot(b)
