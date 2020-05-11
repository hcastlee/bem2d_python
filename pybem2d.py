import os,sys
import ctypes
import numpy as _np

class pybem2d:
    
  def __init__(self,nodes,elems,nQuad,basisOrder,basisType,k):
      """ Construct an instance of the pybem2d class with the geometric and integration information.
      
      Parameters
      ----------
      nodes: np.ndarray of shape (2,NN)
          List of points, x,y
      elems: np.ndarray of shape (2,NE)
          Surface connectivity. Can be an open surface.
          This library will figure out adjacency
          NE = NN-1
      """
      self.nodes = nodes
      self.elems = elems
      self.nQuad = nQuad
      self.wavenumber = k
      # build quadrature rule
      self.create_quad_rule()
      
      if basisOrder >= 1.0:
         print("order not implemented")
      self.basisOrder = order
      
      if basisType == 'DP':
        self.basisType = 'discontinuous'
      else if basisType = 'P':
        self.basisType = 'continuous'
        
        
      # compute normals, global_map, eta, functions
      self.prepare_surface()
        
        
  def get_coef_slp(I,J):
      """ Compute coefficient
      
      Parameters:
      -----------
      I, - row index of the matrix
      J, - column index of the matrix
  
      This function is what you need for computing the integral of the Green function
      """
    
  def get_coef_slp_matrix(II,JJ):
    """ Compute coefficient matrix
    
    Parameters:
    -----------
    II, - row indicies of the matrix, not necesarily consecutive
    J, - column indices of the matrix, not necessarily consecutive

    This function is what you need for computing the integral of the Green function
    block matrix version is used by Pierre Marchand's Htool.
    """

  def create_quad_rule(self):
      """ computes gauss-quadrature rules and stores result in the clas parameters"""
      self.eta, self.wts = _np.polynomials.legendre.leggauss(self.nQuad)
  
  def verify_geometry(self):
      pass
  def compute_normals(self):
    """ Calculate the normal vector for the mesh """
    
    normals = np.zeros(elemsShape)
    for idx in _np.range(0,self.nElems)
      c_nodes = self.nodes[:,self.elems[idx,:]];
      dx = (c_nodes[0,1] - c_nodes[0,0])
      dz = (c_nodes[1,1] - c_nodes[1,0])
      normalMag = np.sqrt(dx*dx + dz*dz)
      dx = dx/normalMag;
      dz = dz/normalMag;
      # normal direction is defined as to the right between node 0 and node 1
      
      normals(0,idx) = -dz;
      normals(1,idx) = dx;
    
    self.normals = normals

  def create_global_to_local_map():
    """Create map between global dof numbering and local dofs.
    This will allow a function to requrest the global coefficient number
    A class array global_to_local is created that contains the mappings
    from the global DOF (aka row and column of the discretized matrix) to
    the local DOF, the basis functions contributing to each row and column.
    If the value of global_to_local_map is -1, then there is not basis function
    that contributes. This is useful for finding edges and a closed domain
    
    gobal_to_local_map: nBasis x nGlobalDOF
    The values in this array correspond to the element number over which the
    the integration occurs.
      
    The first index corresponds the basis functions. For DP0, there is only one 
    basis function. For P1, there are two basis function. For now, no more basis
    functions are planned to be implemented
    
    TODO: Implement the continuous and discontinuous basis function in their own
      class and use polymorphism. Right now I have a lot of conditionals.
      """
    
    # for discontinouous polynomials, the collocation points live on the nodes
    # only order 0 discontinuous polynomials are implemented for now.
    # A refactor is probably needed for more flexibility
    if self.basisType == 'discontinuous':
      # number of dofs contributing to the global index is 1
      global_to_local_map = np.full((1, self.nElems) , -1, dtype=int)
      # loop through the elements in order, which are the same as global DOF.
      for idx in range(0,self.nElems):
          global_to_local_map[0,idx] = idx;
      self.global_to_local_map = global_to_local_map 
      
    if self.basisType == 'continuous':
        # linear basis functions have two dofs that contribute to each 
        global_to_local_map = np.full((2 , self.nNodes) , -1,dtype=int)
        # loop through nodes, which are the same as global DOF
        # Nodes to elems map is a list of elements that are attached to each node
        # in fact, this is the same as the global to local map, so long as I put them
        # in the right spot.
        nodes_to_elems_map = np.full((2,self.nNodes) , -1 , dtype=int )
        for idx in range(0,self.nElems)
          node1 = self.elems[0,idx]
          node2 = self.elems[1,idx]
          nodes_to_elems_map[0,node1] = idx
          nodes_to_elems_map[1,node2] = idx
          global_to_local_map[0,node1] = idx
          global_to_local_map[1,node2] = idx
        # end node loop.
        self.global_to_local_map = global_to_local_map
    # loop through to check if there are any boundary points. 
    # THis needs to be done for 
    boundaryNodes = np.zeros((1,self.nNodes),dtype=bool)
    boundaryElems = np.zeros((1,self.nElems),dtype=bool)
    
    for idx in range(0,self.nNodes)
        
    
  def prepare_surface(self):
    nodesShape = self.nodes.shape()
    if nodesShape[0] != 2
      print("Nodes must have a shape of 2xNN, where NN is the number of nodes")
   self.nNodes = nodesShape[1]
   
   elemsShape = self.elems.shape()
   if elemsShape[0] != 2
      print("Elems must have a shape of 2xNE, where NE is the number of elems")
   self.nElems = elemsShape[1]
   
   self.nNodes = nNodes
   self.nElems = nElems
   self.compute_normals()
   self.create_global_to_local_map()
  