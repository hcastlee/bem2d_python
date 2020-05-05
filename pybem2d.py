import os,sys
import ctypes
import numpy as _np

class pybem2d:
    
  def __init__(self,nodes,elems,nQuad,k):
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
      # build quadrature
      
      # compute normals, global_map, eta, functions
      
        
        
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
  def compue_normals(self):
    # compute normals
    
    normals = np.zeros(elemsShape)
    for idx in _np.range(0,self.nElems)
      self.
     
     
     
     

      
  def prepare_surface(self):
    nodesShape = self.nodes.shape()
    if nodesShape[0] != 2
      print("Nodes must have a shape of 2xNN, where NN is the number of nodes")
   nNodes = nodesShape[1]
   
   elemsShape = self.elems.shape()
   if elemsShape[0] != 2
      print("Elems must have a shape of 2xNE, where NE is the number of elems")
   nElems = elemsShape[1]
   
   self.nNodes = nNodes;
   self.nElems = nElems;
   self.normals = normals;