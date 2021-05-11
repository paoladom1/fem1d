from utils import read_condition, read_mesh, create_local_K, create_local_b, assembly, apply_conditions, calculate_fem
import numpy as np

def main():
    """ Backbone of FEM1D process """
    # fill the mesh data
    mesh = read_mesh()

    # get info read from mesh
    l, k, Q = mesh.parameters
    elements = mesh.elements
    nodes = mesh.nodes

    # get size of elements
    num_elements = len(elements)

    # create local systems
    local_K_array = np.array([create_local_K(l, k) for x in range(num_elements)])
    local_b_array = np.array([create_local_b(l, Q) for x in range(num_elements)])

    # build base K and b from assembly
    K, b = assembly(nodes, elements, local_K_array, local_b_array)

    # apply neumann and dirichlet conditions
    K, b = apply_conditions(mesh.neumann_condition, mesh.dirichlet_condition, K, b)

    T = calculate_fem(K, b)
    print("T: {0}".format(T))

if __name__ == "__main__":
   main() 
