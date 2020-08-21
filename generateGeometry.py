# ------------------------------------------------------------------- #
# generateGeometry.py
#
# A script to generate an stl surface file of the Witch of Agnesi
# profile or the bump described in Liu et al. 
# http://dx.doi.org/10.1016/j.jweia.2016.03.001
#
# Notes:
# - For use in OpenFOAM, the vertices need to be unified (using 
# MeshLab, for instance)
# - The Liu bump will have a total length of exactly 2L, while the
# Agnesi bump extends for a WELL over L on each side. It is a smooth
# function, and at 10L from the peak, the height is ~1% of the peak 
# - Recommended resolution: 10m for 2d agnesi, 40m for 3d;
#                           5m for for 2d/3d liu
#
# Written by Matt Churchfield, 6 Feb 2019
# Updated by Regis Thedin, April 2020: Added Liu et al bumps.
# {matt.churchfield,regis.thedin}@nrel.gov
# ------------------------------------------------------------------- #

import numpy as np
from stl import mesh

# --------------------------- USER INPUT ---------------------------- #
# Bump type (agnesi2d, agnesi3d, liu2d, liu3d)
bump = 'agnesi3d'
L = 1000
h = 100
res = 40
# bump='liu2d'
# L = 100.0
# h = 40.0
# res = 5

# Extents of the outer domain (flat fringe around bump)
xMIN = -50001
xMAX = 50001
yMIN = -50001
yMAX = 50001
# ------------------------- END OF USER INPUT ----------------------- #

# Extents of the area covering the bump
if bump[:3]=='liu':
    xMin = -2*L + 20
    xMax =  2*L + 20
    yMin = -2*L + 20
    yMax =  2*L + 20
elif bump[:3]=='agn':
    xMin = -30*L  # At 30L from the center, the height is 0.0011h
    xMax =  30*L
    yMin = -30*L
    yMax =  30*L

assert (xMIN < xMin and xMAX > xMax and yMIN < yMin and yMAX > yMax), \
    "Outer domain limits should be larger to accomodate bump."
       
# Number of surface cells for bump area
lim = xMax-xMin
nx = int(lim/res)
ny = int(lim/res)
if bump[-2:]=='2d':
    ny=1

# Define the x and y arrays.
x = np.linspace(xMin,xMax,nx+1)
y = np.linspace(yMin,yMax,ny+1)

# Insert the limits of the domain. It is flat, a single points is enough
x = np.insert(x,0,xMIN)
x = np.insert(x,len(x),xMAX)
y = np.insert(y,0,yMIN)
y = np.insert(y,len(y),yMAX)

# Define a numpy array of surface vertices.
vertices = np.zeros((len(x)*len(y),3),dtype=float)

# Load the x and y data into the vertices array.
for j in range(len(y)):
    vertices[j*len(x):(j+1)*len(x),0] = x
    vertices[j*len(x):(j+1)*len(x),1] = y[j]   
    
# Define the terrain.
for k in range(vertices.shape[0]):
    xx = vertices[k,0]
    yy = vertices[k,1]
    if bump=='liu2d':
        if abs(xx)>L:  zz=0
        else:          zz = h * np.cos(np.pi*xx/(2*L))**2   # 2d bump   
    elif bump=='liu3d':
        rr = np.sqrt(xx**2 + yy**2)
        if rr>L:  zz=0
        else:     zz = h * np.cos(np.pi*rr/(2*L))**2        # 3d bump  
    elif bump=='agnesi2d':
        zz = h/(1.0+(xx/L)**2)                   # witch of agnesi 2d
    elif bump=='agnesi3d':
        rr = np.sqrt(xx**2 + yy**2)
        zz = h/(1.0+(rr/L)**2)                   # witch of agnesi 3d
    vertices[k,2] = zz   
    
# Define a numpy array of the faces and load it up with vertices.
faces = np.zeros((2*len(x)*len(y),3),dtype=int)
ii = 0
for j in range(len(y)-1):
    for i in range(len(x)-1):
        p0 = j*len(x)+i
        p1 = j*len(x)+i+1
        p2 = (j+1)*len(x)+i
        p3 = (j+1)*len(x)+i+1
        faces[ii] = [p0,p1,p2]
        faces[ii+1] = [p1,p3,p2]
        ii = ii+2

# Create the mesh
surface = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
   for j in range(3):
      surface.vectors[i][j] = vertices[f[j],:]

# Write the mesh to file in stl format.
surface.save(f'./bump_{bump}_L{int(L)}_h{int(h)}.stl')
