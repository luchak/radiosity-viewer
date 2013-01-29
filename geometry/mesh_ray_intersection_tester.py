#!/usr/bin/env python

import numpy

def PluckerCoords(start_points, end_points):
  if start_points.shape != end_points.shape:
    raise ValueError('Start and end points must have the same shape.')
  if start_points.shape[-1] != 3:
    raise ValueError('Coordinates must be in R3.')
  num_points = 1 if len(start_points.shape) == 1 else len(start_points)
  result = numpy.empty((num_points, 6))
  result[:,:3] = start_points - end_points
  result[:,3:] = numpy.cross(start_points, end_points)

  return result

class MeshRayIntersectionTester(object):
  def __init__(self, mesh):
    self.coords01 = PluckerCoords(
        mesh.vertices[mesh.faces[:,0]],
        mesh.vertices[mesh.faces[:,1]])
    self.coords12 = PluckerCoords(
        mesh.vertices[mesh.faces[:,1]],
        mesh.vertices[mesh.faces[:,2]])
    self.coords20 = PluckerCoords(
        mesh.vertices[mesh.faces[:,2]],
        mesh.vertices[mesh.faces[:,0]])

  def RayHandedness(self, ray_coords, edge_coords):
    return (edge_coords[:,5]*ray_coords[:,2] +
            edge_coords[:,4]*ray_coords[:,1] +
            edge_coords[:,0]*ray_coords[:,3] +
            edge_coords[:,2]*ray_coords[:,5] +
            edge_coords[:,1]*ray_coords[:,4] +
            edge_coords[:,3]*ray_coords[:,0])

  def RayIntersections(self, ray):
    ray_coords = PluckerCoords(ray[0], ray[0] + ray[1])

    hands01 = self.RayHandedness(ray_coords, self.coords01)
    hands12 = self.RayHandedness(ray_coords, self.coords12)
    hands20 = self.RayHandedness(ray_coords, self.coords20)

    def LogicalAnd3(a, b, c):
      return numpy.logical_and(a, numpy.logical_and(b, c))

    in_front = LogicalAnd3(hands01 < 0.0, hands12 < 0.0, hands20 < 0.0)
    in_back = LogicalAnd3(hands01 > 0.0, hands12 > 0.0, hands20 > 0.0)

    return numpy.arange(len(self.coords01))[numpy.logical_or(in_front, in_back)]
