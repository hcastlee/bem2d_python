import os,sys
import ctypes
import numpy as _np

def scale_quadrule(eta,wts,a1,b1,a2,b2,etaNew,wtsNew):
    """Scale eta and wts quad rule that originally went from a1 to b1, to
    a2 to b2 """
    # Move to 0,1
    oldRange = b1 - a1
    newRange = b2 - a2
    oldCenter = (a1+b1)/2
    newCenter = (a2+b2)/2
    etaNew = (eta - oldCenter)/oldRange*newRange + newCenter
    wtsNew = wts/oldRange*newRange
  
  def tensorquad(x,wx,y,wy):
       """Combine two quadrules in the x and y direction into a 2D tensor rule"""
    
       nx=len(x)
       ny=len(y)
       m=numpy.mgrid[0:nx,0:ny].reshape(2,-1)
       x2=numpy.vstack((x[m[0]],y[m[1]]))
       w2=wx[m[0]]*wy[m[1]]
       return x2,w2   
class pybem2d:

    
    
  def __init__(self,nodes,elems,nQuad,nQuadSing,basisOrder,basisType,k):
      """ Construct an instance of the pybem2d class with the geometric and integration information.
      
      Parameters
      ----------
      nodes: np.ndarray of shape (2,NN)
          List of points, x,y
      elems: np.ndarray of shape (2,NE)
          Surface connectivity. Can be an open surface.
          This library will figure out adjacency
          NE = NN-1
      nQuad:
      nQuadSing:
      basisOrder:
      basisType:
      k:
      """
      self.nodes = nodes
      self.elems = elems
      self.nQuad = nQuad
      self.nQuadSing = nQuadSing
      self.wavenumber = k
      # build quadrature rule
      self.create_quad_rule()
      
      if basisOrder > 1.0:
         print("order not implemented. Error")
         break # this may not work...
      else
          self.basisOrder = order
          
      
      if basisType == 'DP':
        self.basisType = 'discontinuous'
      elif basisType == 'P':
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
      """computes gauss-quadrature rules and stores result in the class parameters
      Also compute refined quadrature rules for singular elements"""
      eta, wts = _np.polynomial.legendre.leggauss(self.nQuad)
      self.eta = (eta+1)/2.0
      self.wts = wts/2.0;
      
  def create_refined_composite_quad_rule(self):
      """Adapted from Timo Betcke's pybem2d library"""
      self.x_adapted=numpy.array([])
      self.w_adapted=numpy.array([])
      self.sigma = (np.sqrt(2.0)-1)*(np.sqrt(2.0)-1) # this I borrowed from Xavier Clayes' bemtool
      self.recDepth = 3;
      a,b=self.sigma,1
      for i in range(self.recDepth+1):
          self.eta_adapted1D=numpy.hstack([self.eta_adapted1D,a+(b-a)*self.eta])
          self.wts_adapted1D=numpy.hstack([self.wts_adapted1D,(b-a)*self.wts])
          a*=self.sigma
          b*=self.sigma


  def singular_quad_rule_dp0(self):
      """creates a singular quadrature rule for an element.
      Need to split the integration interval into (0,0.5) and (0.5,1)
      Use the adapted singular quadrature rule"""
      scale_quadrule(self.eta_adapted1D,self.wts_adapted1D,0,1,0,0.5,eta_first_half,wts_first_half)
      scale_quadrule(self.eta_adapted1D,self.wts_adapted1D,0,1,0.5,1.0,eta_second_half,wts_second_half)
      self.eta_singular_dp0 = _np.hstack([eta_first_half,eta_second_half])
      self.wts_singular_dp0 = _np.hstack([wts_first_half,wts_second_half])
      
      
 
          
  def singular_quad_rule_p1(self):
      """creates a singular quadrature rule for an element.
     don't need to split the integration up, since the singular integration
     takes place over two elements"""
      # using refined quad rule for each element. need to reverse quad rule for first element (idx0) and unreversed for second element (idx1). May be able to generalize to higher-order elements. 
      self.eta_singular_p1 = _np.vstack((_np.fliplr(self.eta_adapted1D),self.eta_adapted1D))
      self.wts_singular_p1 = _np.vstack((_np.fliplr(self.wts_adapted1D),self.wts_adapted1D))

       
  def define_shape_functions(self):
      """Define regular shape functions"""
      if self.basisType == 'discontinuous':
          self.basisNum = 1;
          self.phin = _np.ones((self.nBasis,1) , dtype=float64);
      elif self.basisType == 'continuous':
          self.basisNum = 1;
          self.phin = _np.zeros((2 , self.nBasis) , dtype=float64);
          #regullar shape functions
          self.phin(1,:) = (1.0 - self.eta)/2
          self.phin(2,:) = (1.0 + self.eta)/2
          # singuar shape functions times the jacobian

  
  def compute_normals(self):
    """ Calculate the normal vector for the mesh """
    
    normals = np.zeros(elemsShape)
    for idx in _np.range(0,self.nElems):
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
      # disable checking boundary nodes and elems for now...
      self.boundaryNodes = None
      self.boundaryElems = None
      
    elif self.basisType == 'continuous':
        # linear basis functions have two dofs that contribute to each 
        global_to_local_map = np.full((2 , self.nNodes) , -1,dtype=int)
        # loop through nodes, which are the same as global DOF
        # Nodes to elems map is a list of elements that are attached to each node
        # in fact, this is the same as the global to local map, so long as I put them
        # in the right spot.
        boundaryNodes = np.full((1,self.nNodes),-1,dtype=int)
        boundaryElems = np.full((1,self.nElems),-1,dtype=int)
        nodes_to_elems_map = np.full((2,self.nNodes) , -1 , dtype=int )
        for idx in range(0,self.nElems):
          node1 = self.elems[0,idx]
          node2 = self.elems[1,idx]
          nodes_to_elems_map[0,node1] = idx
          nodes_to_elems_map[1,node2] = idx
          global_to_local_map[0,node1] = idx
          global_to_local_map[1,node2] = idx
        # end node loop.
        self.global_to_local_map = global_to_local_map
        
        # loop through to check if there are any boundary points. 
        # Boundary points are nodes that are attached to only one element
        # Actuall, this information is already in there. if global_to_local_map ==-1, then we are on a boundary.
        # if global DOF is a boundary node, then return zero for the coefficient. 

    
   
            
        
    
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
  