# File created on: 2018-05-08 15:24:47.266919
#
# IMPORTANT:
# ----------
# - Before making a new Practical always make sure you have the latest version of the addon!
# - Delete the old versions of the files in your user folder for practicals you want to update (make a copy of your work!).
# - For the assignment description and to download the addon, see the course website at: http://www.cs.uu.nl/docs/vakken/ddm/
# - Mail any bugs and inconsistencies you see to: uuinfoddm@gmail.com
# - Do not modify any of the signatures provided.
# - You may add as many functions to the assignment as you see fit but try to avoid global variables.

import ddm
import bpy
import numpy
from numpy import array as Vector
from numpy import matrix as Matrix
from numpy import identity

    
#################################
# Place additional imports here #
#################################

# Return a list of vertices
def get_vertices(context):
    selected_obj = context.scene.objects.active
    vertices = []
    for v in selected_obj.data.vertices:
        vertices.append( v.co )
    return vertices
    
# Returns a list of triangles of vertex indices (you need to perform simple triangulation) 
def get_faces(context):
    selected_obj = context.scene.objects.active
    faces = []
    for f in selected_obj.data.polygons:
        for i in range( len(f.vertices) - 2):
            faces.append((f.vertices[i], f.vertices[i+1], f.vertices[i+2]))
    return faces

# Returns the 1-ring (a list of vertex indices) for a vertex index
def neighbor_indices(vertex_index, vertices, faces):
    verts = [vertex_index]
    for f in faces:
        for i in range(3):
            if f[i] == vertex_index:
                for j in range(3):
                    if f[j] not in verts:
                        verts.append(f[j])
    verts.remove(vertex_index)
    return verts
    
# Calculates the source (non-sparse) matrix P
def source_matrix(p_index, vertices, neighbor_indices):
    P_list = []
    for p in neighbor_indices:
        P_list.append( vertices[p_index] - vertices[p] )
    return Matrix(P_list)
    
# Calculates the target (non-sparse) matrix Q
def target_matrix(p_index, vertices, neighbor_indices):
    Q_list = []
    for q in neighbor_indices:
        Q_list.append( vertices[p_index] - vertices[q] )
    return Matrix(Q_list)
    
# Returns a triple of three dense matrices that are the Singular Value Decomposition (SVD)
def SVD(P, Q):

    S = P.transpose() * Q
    
    (U, Sigma, V) = numpy.linalg.svd(S, full_matrices=True)

    # Make sure that the result of the singular value decomposition is a triple of numpy.matrix and not some other datastructure.
    return (U, Sigma, V)

# Returns the dense matrix R
def rigid_transformation_matrix(U, Sigma, V):
    Ri = U * V.transpose()
    if numpy.linalg.det(Ri) == -1:
        # reflection instead of rotation
        smallest_index = 0
        for j in range(3):
            if Sigma[j][j] < Sigma[smallest_index][smallest_index]:
                smallest_index = j
        Sigma = numpy.identity(3)
        Sigma[smallest_index][smallest_index] = -1
        Ri = U * Sigma * V.transpose()
        
    return Ri

# Returns a list of rigid transformation matrices R, one for each vertex (make sure the indices match)
# the list_of_1_rings is a list of lists of neighbor_indices
def local_step(source_vertices, target_vertices, list_of_1_rings):
    locals = []
    for i in range( len(source_vertices)):
        P = source_matrix(i, source_vertices, list_of_1_rings[i])
        Q = target_matrix(i, target_vertices, list_of_1_rings[i])
        
        (U, Sigma, V) = SVD (P, Q)
        Ri = rigid_transformation_matrix(U, Sigma, V)
        locals.append(Ri)
    return locals

# Returns the triplets of sparse d_0 matrix
def d_0(vertices, faces):
    
    return [(1,1,0.2), (1,2,0.3)]
    
# Return the sparse diagonal weight matrix W
def weights(vertices, faces):
    return ddm.Sparse_Matrix([], 1, 1)

# Returns the right hand side of least-squares system
def RHS(vertices, faces):

    # You need to convert the end result to a dense vector
    
    return Vector([1,2,4,5,5,6])
    
# Returns a list of vertices coordinates (make sure the indices match)
def global_step(vertices, rigid_matrices):

    # TODO: Construct RHS
    
    # TODO: solve separately by x, y and z (use only a single vector)



    return [(1,1,1), (1,1,1), (1,1,1)]
    
# Returns the left hand side of least-squares system
def precompute(vertices, faces):
    
    # TODO: construct d_0 and split them into d_0|I and d_0|B
    
    # TODO: construct LHS with the elements above and Cholesky it

    return ddm.Sparse_Matrix([], 1, 1)

# Initial guess, returns a list of identity matrices for each vertex
def initial_guess(vertices):
    return [Matrix(), Matrix(), Matrix()]

# WARNING: Changed signature!!! original: (vertices, faces, max_movement)
def ARAP_iteration(vertices, faces, max_movement, one_rings):

    # TODO: local step
    locals = local_step(vertices, vertices, one_rings)
    #apply local transforms
    #for v in V:
    
    #global_step()
    
    # TODO: global step

    new_vertices = []
    # Returns the new target vertices as a list (make sure the indices match)
    for i in range( len(vertices) ):
        v = vertices[i]
        #transform each vertex according to locals
        
        new_v_vector = numpy.matmul(list(locals[i]), list(v))
        new_vertices.append( new_v_vector )
     
    return new_vertices
    
def DDM_Practical5(context):
    print ("Running Practical 5")
    max_movement = 0.001
    tolerance = 0.1
    max_iterations = 100
    
    selected_obj = context.scene.objects.active
    
    V = get_vertices(context)
    F = get_faces(context)
    
    print ("Creating one-Rings list")
    one_rings = []
    for i in range(len(V)):
        one_rings.append(neighbor_indices(i, V, F))
    
    print("Getting handles")
    # TODO: get handles
    handles = get_handles(selected_obj)
    print ("Getting mesh data")
    # TODO: get mesh data
    print ("Precomputing Cholesky")
    # TODO: precompute
    print ("ARAP-ing...")
    # TODO: initial guess
    initial_Ri = numpy.identity(3)
    difference = 1
    iterations = 0
    
    new_V = ARAP_iteration(V, F, max_movement, one_rings)
    
    #show_mesh(V, F, selected_obj, context)
    show_mesh(new_V, F, selected_obj, context)
    # TODO: ARAP until tolerance
    print ("Didelidone")
    pass

# Builds a mesh using a list of triangles
# This function is the same as from the previous practical
def show_mesh(vertices, faces, selected_obj, context):
    
    me = bpy.data.meshes.new("Mesh")
    ob = bpy.data.objects.new("mesh", me)
    ob.scale = selected_obj.scale
    ob.location = selected_obj.location
    ob.rotation_euler = selected_obj.rotation_euler
    context.scene.objects.link(ob)
    
    edges = []
    verts = []
    for v in vertices:
        verts.append( tuple(list(v)) )
    
    me.from_pydata(verts, edges, faces)
    me.update()
    return
#########################################################################
# You may place extra variables and functions here to keep your code tidy
#########################################################################

d0I = Matrix()
d0I_neg = Matrix()

# Find the vertices within the bounding box by transforming them into the bounding box's local space and then checking on axis aligned bounds.
def get_handle_vertices(vertices, bounding_box_transform, mesh_transform):

    result = []

    # Fibd the transform into the bounding box's local space
    bounding_box_transform_inv = bounding_box_transform.copy()
    bounding_box_transform_inv.invert()
    
    # For each vertex, transform it to world space then to the bounding box local space and check if it is within the canonical cube x,y,z = [-1, 1]
    for i in range(len(vertices)):
        vprime = vertices[i].co.copy()
        vprime.resize_4d()
        vprime = bounding_box_transform_inv * mesh_transform * vprime
        
        x = vprime[0]
        y = vprime[1]
        z = vprime[2]
        
        if (-1 <= x) and (x <= 1) and (-1 <= y) and (y <= 1) and (-1 <= z) and (z <= 1):
            result.append(i)

    return result

# Returns the local transform of the object
def get_transform_of_object(name):
    return bpy.data.objects[name].matrix_basis
    
# Finds the relative transform from matrix M to T
def get_relative_transform(M, T):
    
    Minv = M.copy()
    Minv.invert()
        
    return T * Minv
    
# Returns a list of handles and their transforms
def get_handles(source):
    
    result = []
    
    mesh_transform = get_transform_of_object(source.name)
    
    # Only search up to (and not including) this number of handles
    max_handles = 10
    
    # For all numbered handles
    for i in range(max_handles):
    
        # Construct the handles representative name
        handle_name = 'handle_' + str(i)
        
        # If such a handle exists
        if bpy.data.objects.get(handle_name) is not None:
            
            # Find the extends of the aligned bounding box
            bounding_box_transform = get_transform_of_object(handle_name)
            
            # Interpret the transform as a bounding box for selecting the handles
            handle_vertices = get_handle_vertices(source.data.vertices, bounding_box_transform, mesh_transform)
            
            # If a destination box exists
            handle_dest_name = handle_name + '_dest'
            if bpy.data.objects.get(handle_dest_name) is not None:
                
                bounding_box_dest_transform = get_transform_of_object(handle_dest_name)
                
                result.append( (handle_vertices, get_relative_transform(bounding_box_transform, bounding_box_dest_transform) ) ) 
                
            else:
            
                # It is a rigid handle
                m = Matrix([])
                m = identity(4)
                result.append( (handle_vertices, m) )
            
    return result
    
# Slices a list of triplets and returns two lists of triplets based on a list of fixed columns
# For example if you have a set of triplets T from a matrix with 8 columns, and the fixed columns are [2, 4, 7]
# then all triplets that appear in column [1, 3, 5, 6] are put into "right_triplets" and all triplets that appear in column [2, 4, 7] are put into the set "left_triplets"
def slice_triplets(triplets, fixed_colums):

    left_triplets = []
    right_triplets = []

    # First find the complement of the fixed column set, by finding the maximum column number that appear in the triplets
    max_column = 0
    
    for triplet in triplets:
        if (triplet[1] > max_column):
            max_column = triplet[1]
    
    # and constructing the other columns from those
    other_columns = [x for x in range(0, max_column + 1) if x not in fixed_colums]
    
    # Now split the triplets
    for triplet in triplets:
    
        if (triplet[1] in fixed_colums):
            new_column_index = fixed_colums.index(triplet[1])
            left_triplets.append( (triplet[0], new_column_index, triplet[2]) )
        else:
            new_column_index = other_columns.index(triplet[1])
            right_triplets.append( (triplet[0], new_column_index, triplet[2]) )
            
    return (left_triplets, right_triplets)