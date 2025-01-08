"""
Kind of clusters user can add to the dataset with the creator.
"""
import numpy as np
from abc import ABC, abstractmethod

class Cluster(ABC):
    """
    Abstract class for "clusters" that can be added to the dataset.
    Clusters are defined by a shape, position, and relative density (between 0.0 and 1.0).
    
    A dataset of N points is created with the mixture model generative process, i.e.
    a cluster type is chosen with probability proportional to its relative density, and
    a point is generated from the chosen cluster's distribution.
    """

    @abstractmethod
    def density(self, x):
        """
        Return the density of the cluster at the point x.
        """
        pass

    @abstractmethod
    def generate(self):
        """
        Generate a point from the cluster.
        """
        pass

class Elipse(Cluster):
    def __init__(self, center, p0, axis_ratio=.5, density= 1.):
        """
        Create a cluster with an eliptical shape and uniform distribution.
        :param center: the centerpoint of the elipse
        :param p0: the first principal axis of the elipse
        :param axis_ratio: the ratio between the first and second principal axis
        :param density: the relative density of the cluster
        :param distribution: 
            'uniform' - points are uniformly distributed in the elipse
            'normal' - points are normally distributed in the elipse, using the radii as standard deviations
            'annular

        """
        self.center = center
        self.p0 = p0
        self.axis_ratio = axis_ratio
        self.density = density
    
