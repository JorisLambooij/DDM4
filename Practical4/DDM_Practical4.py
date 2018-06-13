# File created on: 2018-05-08 15:24:47.265919
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
import math
from mathutils import Vector

# Place additional imports here

class Mesh():

    # Construct a new mesh according to some data
    def __init__(self, vertices, faces):
    
        # The vertices are stored as a list of vectors
        self.vertices = vertices
        
        # The faces are stored as a list of triplets of vertex indices
        self.faces = faces
    
        # The uv-coordinates are stored as a list of vectors, with their indices corresponding to the indices of the vertices (there is exactly one uv-coordinate per vertex)
        self.uv_coordinates = []
        for vertex in vertices:
            self.uv_coordinates.append(Vector( (vertex[0], vertex[2]) ))
    
        self.build_edge_list()
    
    # This function builds an edge list from the faces of the current mesh and stores it internally.
    # Make sure that each edge is unique. Remember that order does NOT matter, e.g. (1, 0) is the same edge as (0, 1).
    # The indices of the edges should correspond to the locations of the weights in your weight calculation.
    # All subsequent calls that do something with edges should return their indices. Getting the actual edge can then be done by calling get_edge(index).
    def build_edge_list(self):
    
        self.edges = []
        for triangle in self.faces:
            for i in range(3):
                j = i + 1
                if j == 3:
                    j = 0
                edge = (triangle[i], triangle[j])
                edge_r = (triangle[j], triangle[i])
                if edge not in self.edges and edge_r not in self.edges:
                    self.edges.append( edge )
        
    # ACCESSORS
    
    def get_vertices(self):
        return self.vertices
        
    def get_vertex(self, index):
        return self.vertices[index]
    
    def get_edges(self):
        return self.edges
        
    def get_edge(self, index):
        return self.edges[index]
    
    def get_faces(self):
        return self.faces
        
    def get_face(self, index):
        return self.faces[index]
        
    def get_uv_coordinates(self):
        return self.uv_coordinates
        
    def get_uv_coordinate(self, index):
        return self.uv_coordinates[index]
    
    # Returns the list of vertex coordinates belonging to a face.
    def get_face_vertices(self, face):
        return [ self.get_vertex(face[0]), self.get_vertex(face[1]), self.get_vertex(face[2]) ]
    
    # Looks up the edges belonging to this face in the edge list and returns their INDICES (not value). Make sure that each edge is unique (recall that (1, 0) == (0, 1)). These should match the order of your weights.
    def get_face_edges(self, face):
        indices = []
        for i in range (3):
            j = i + 1
            if j == 3:
                j = 0
            edge = (face[i], face[j])
            edge_r = (face[j], face[i])
            
            try:
                indices.append (self.edges.index(edge))
            except:
                indices.append (self.edges.index(edge_r))
            
        return indices
    
    # Returns the vertex coordinates of the vertices of the given edge (a pair of vertex indices e.g. (0,1) ) 
    def get_edge_vertices(self, edge):
        return [ self.get_vertex(edge[0]), self.get_vertex(edge[1])]
    
    # Returns the flap of the given edge belonging to edge_index, that is two faces connected to the edge 1 for a boundary edge, 2 for internal edges
    def get_flaps(self, edge_index):
        flap = []
        def face_contains_edge(face):
            for i in range (3):
                j = i + 1
                if j == 3:
                    j = 0
                f_edge = (face[i], face[j])
                f_edge_r = (face[j], face[i])
                
                edge_vs = self.edges[edge_index]
                if edge_vs == f_edge or edge_vs == f_edge_r:
                    return True
            return False

        for face in self.faces:
            if face_contains_edge(face):
                flap.append(face)
        # Watch out: edges might be on the boundary
        
        return flap
        
    # Returns the length of the given edge with edge_index
    def get_edge_length(self, edge_index):
        
        edge = self.edges[edge_index]
        
        p1 = self.vertices[edge[0]]
        p2 = self.vertices[edge[1]]
        e = p1 - p2
        return (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]) ** 0.5
        
    # Returns whether the edge has two adjacent faces
    def is_boundary_edge(self, edge_index):
        flap = self.get_flaps(edge_index)
        return len(flap) == 2
    
    # Returns the boundary of the mesh by returning the indices of the edges (from the internal edge list) that lie around the boundary.
    def boundary_edges(self):
        
        boundary = []
        for edge in self.edges:
            if is_boundary_edge:
                if edge[0] not in boundary:
                    boundary.append (edge[0])
                if edge[1] not in boundary:
                    boundary.append (edge[1])
                
        return boundary
        
    # Place any other accessors you need here
    
# This function is called when the DDM operator is selected in Blender.
def DDM_Practical4(context):
    
    # TODO: remove example code and implement Practical 4
    
    # Example mesh construction
    
    # The vertices are stored as a list of vectors (ordered by index)
    vertices = [Vector( (0,0,0) ), Vector( (0,1,0) ), Vector( (1,1,0) ), Vector( (1,0,0) )]
    
    # The faces are stored as triplets (triangles) of vertex indices, polygons need to be triangulated as in previous practicals. This time edges should also be extracted.
    faces = [ (0, 1, 2), (0, 2, 3) ]
    
    # Construct a mesh with this data
    M = get_mesh()
    #M = Mesh(vertices, faces)
    
    # M.get_face_edges( (0, 1, 2) )
    # show_mesh(M, "test mesh")
    
    # You can now use the accessors to access the mesh data
    #print(M.get_edges())
    
    # An example of the creation of sparse matrices
    A = ddm.Sparse_Matrix([(0, 0, 4), (1, 0, 12), (2, 0, -16), (0, 1, 12), (1, 1, 37), (2, 1, -43), (0, 2, -16), (1, 2, -43), (2, 2, 98)], 3, 3)
    
    # Sparse matrices can be multiplied and transposed
    B = A.transposed() * A
    
    # Cholesky decomposition on a matrix
    B.Cholesky()
    
    # Solving a system with a certain rhs given as a list
    rhs = [2, 2, 2]
    
    x = B.solve(rhs);
    
    # A solution should yield the rhs when multiplied with B, ( B * x - v should be zero)
    #print(Vector( B * x ) - Vector(rhs) )
    
    # You can drop the matrix back to a python representation using 'flatten'
    #print(B.flatten())
    uni_weights = uniform_weights(M)
    show_mesh(Convex_Boundary_Method(M, uni_weights), "Uniform")
    # TODO: show_mesh on a copy of the active mesh with uniform UV coordinates, call this mesh "Uniform"
    
    # TODO: show_mesh on a copy of the active mesh with cot UV coordinates, call this mesh "Cot"
    
    # TODO: show_mesh on a copy of the active mesh with boundary free UV coordinates, call this mesh "LSCM"
    

# You may place extra functions here

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

# Returns the weights for each edge of mesh M.
# It therefore consists of a list of real numbers such that the index matches the index of the edge list in M.
def cotan_weights(M):
    
    # TODO: implement yourself
    weights = []
    for edgeIndex in range(len(M.edges)):
        flaps = M.get_flaps(edgeIndex)
        if len(flaps) == 1:
            weights.append(0.0)
        else:
            w = 0.0
            edges = M.get_edge_vertices[M.edges[edgeIndex]]
            for flap in flaps:
                for vertex in [n for n in M.get_face_vertices[flap] if n not in edges]:
                    w += 1 / math.tan((vertex - edges[0]).angle(vertex - edges[1]))
            w /= 2
            weights.append(w)
    
    return weights
    
# Same as above but for uniform weights
def uniform_weights(M):

    # TODO: implement yourself
    weights = []
    for edgeIndex in range(len(M.edges)):
        if M.is_boundary_edge(edgeIndex) == 1:
            weights.append(0)
        else:
            weights.append(1)
    return weights
    
# Given a set of weights, return M with the uv-coordinates set according to the passed weights
def Convex_Boundary_Method(M, weights):

    #Construct d0 sparse matrix
    #Maybe do this after getting boundary edges so we can split the d0 at construction
    d0_list = []
    for e_i in range(len(M.edges)):
        edge = M.edges[e_i]
        d0_list.append((e_i, edge[0], -1))
        d0_list.append((e_i, edge[1], 1))
    d0 = ddm.Sparse_Matrix(d0_list, len(M.edges), len(M.get_vertices()))

    boundary_edges = []
    for edgeIndex in range(len(M.edges)):
        if M.is_boundary_edge(edgeIndex):
            boundary_edges.append(M.edges[edgeIndex])
    
    boundary_vertex_indices = []
    for edge in boundary_edges:
        if edge[0] not in boundary_vertex_indices:
            boundary_vertex_indices.append(edge[0])
        if edge[1] not in boundary_vertex_indices:
            boundary_vertex_indices.append(edge[1])

    inner_edge_vertices = M.get_vertices().copy()
    count = 0
    for i in range(len(boundary_vertex_indices)):
        inner_edge_vertices.pop(boundary_vertex_indices[i] - i)


    return Mesh(inner_edge_vertices, [])

# Using Least Squares Conformal Mapping, calculate the uv-coordinates of a given mesh M and return M with those uv-coordinates applied
def LSCM(M):

    # TODO: implement yourself

    return M
    
# Builds a Mesh class object from the active object in the scene.
# in essence this function extracts data from the scene and returns it as a (simpler) Mesh class, triangulated where nessecary.
def get_mesh():

    obj = bpy.context.selected_objects[0]
    print (obj.name)
    
    verts = obj.data.vertices
    polys = obj.data.polygons
    
    vertices = []
    for vertex in verts:
        vertices.append(vertex.co)
        
    faces = []
    for poly in polys:
        faces.append( poly.vertices )
        
    return Mesh(vertices, faces)
    
# Given a Mesh class M, create a new object with name in the scene with the data from M
def show_mesh(M, name):
    
    me = bpy.data.meshes.new("Mesh")
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.objects.link(ob)
    
    edges = []
    faces = []
    verts = []
    for tri in M.faces:
        verts_indices = [0, 0, 0]
        for j in range(3):
            if tri[j] not in verts:
                verts_indices[j] = len(verts)
                verts.append(M.vertices[tri[j]])
            else:
                verts_indices[j] = verts.index(tri[j])
        faces.append( tuple(verts_indices) )
    
    
    me.from_pydata(verts, edges, faces)
    
    return
    
    pass