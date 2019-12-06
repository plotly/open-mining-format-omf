import omf  
import vectormath 
import numpy as np
import plotly.graph_objects as go

class TypeError(Exception):
    pass

class DataError(Exception):
    pass
    
def pointset_to_scatter3d(projO, pointset, usercolor=None, markersize=8, dataindex=0):
    """
    projO: vectormath.vector.Vector3;  Project origin point; default (0,0,0)
    pointset: an instance of the omf.PointsetElement class
    usercolor:  a Plotly admissible rgb, hex, or CSS named color
    markersize -  user defined marker size 
    -------
    Returns an instance of go.Scatter3d
    """
    if not isinstance(projO, vectormath.vector.Vector3) or\
        not isinstance(pointset, omf.PointSetElement):
        raise TypeError("Either the first argument is not a vectormath.vector.Vector3 or\n" +\
                          "the second one is not an omf.PointSetElement instance")
    pgeom = pointset.geometry                      
    x = projO.x + pgeom.vertices.array.x 
    y = projO.y + pgeom.vertices.array.y
    z = projO.z + pgeom.vertices.array.z
    
    if len(pointset.data) == 0:
        markercolor = f"rgb{pointset.color}"  if usercolor is  None else usercolor
    else:
        if dataindex < len(pointset.data):
            markercolor = pointset.data[dataindex].array.array
        else:
            raise ValueError(f"dataindex must be  less than {len(pointset.data)}")
    tracename = f"{pointset.subtype}s" if pointset.name.isspace() else pointset.name
      
    return go.Scatter3d(x=x, y=y, z=z, 
                        name=tracename,
                        mode='markers', 
                        marker_color=markercolor,
                        marker_size=markersize
                       )

def lineset_to_scatter3d(projO, lineset, usercolor=None, linewidth=5, markersize=1):
    """
    projO: vectormath.vector.Vector3;  Project origin point; default (0,0,0)
    lineset: omf.LineSetElement object
    usercolor: a Plotly admissible rgb, hex or CSS named color
    linewidth: user defined line_width
    markersize: user defined marker_size 
    -------
    Returns an instance of go.Scatter3d
    """
    if not isinstance(projO, vectormath.vector.Vector3) or  not isinstance(lineset, omf.LineSetElement):
        raise TypeError("Either the first argument is not a vectormath.vector.Vector3 or\n" +\
                          "the second one is not an omf.PointSetElement instance")
    points = lineset.geometry.vertices.array 
    edges = lineset.geometry.segments.array
    edx = []
    edy = []
    edz = []
    for e in edges:
        edx.extend([projO.x + points[e[0]].x,  projO.x + points[e[1]].x, None]) 
        edy.extend([projO.y + points[e[0]].y,  projO.y + points[e[1]].y, None]) 
        edz.extend([projO.z + points[e[0]].z,  projO.z + points[e[1]].z, None]) 
    
    linecolor = f"rgb{lineset.color}" if usercolor is None else usercolor
    tracename = f"{lineset.subtype}s" if lineset.name.isspace() else lineset.name
      
    return go.Scatter3d(x=edx, y=edy, z=edz, 
                        name=tracename,
                        marker_size=markersize, 
                        line_color=linecolor,
                        line_width=linewidth 
                       )
                       
def meshelement_to_plotly(projO, surfelement, unicolor = False, usercolor=None):
    """
    projO: vectormath.vector.Vector3;  Project origin point; default (0,0,0)
    surfelement: an instance of omf.Surface with omf.SurfaceGeometry
    unicolor: boolean to point out whether the surface is colored by a simple color or not
    ---------
    returns a go.Mesh3d instance
    """
    if not isinstance(projO, vectormath.vector.Vector3) or  not isinstance(surfelement, omf.SurfaceElement):
        raise TypeError("Either the first argument is not a vectormath.vector.Vector3 or\n" +\
                          "the second one is not an omf.SurfaceElement instance")
    if not  isinstance(surfelement.geometry, omf.SurfaceGeometry): 
        raise TypeError("This surface element doesn't have an  omf.SurfaceGeometry" )
    sgeom = surfelement.geometry
    meshO = sgeom.origin
    verts = sgeom.vertices.array
    i, j, k = sgeom.triangles.array.T
    if unicolor:
        color = f"rgb{surfelement.color}" if usercolor==None else usercolor
        intensity = None
    else:
        if len(surfelement.data) == 0:
            intensity = verts.z
        else:
            locations = [d.location  for d in surfelement.data]
            if 'vertices' in locations:
                dataindex = locations.index('vertices')
                intensity = surfelement.data[dataindex].array.array
            else:
                intensity = verts.z
            color=None        
    return go.Mesh3d(x=projO.x + meshO.x + verts.x,
                     y=projO.y + meshO.y + verts.y,
                     z=projO.z + meshO.z + verts.z,
                     i=i,
                     j=j,
                     k=k,
                     intensity=intensity,
                     flatshading=True,
                     color=color,
                     name=surfelement.name
                        )
                        
                        
def surfgrid_to_surface(projO, surfelement, unicolor=False, usercolor=None, colorscale='deep'):
    """
    projO: vectormath.vector.Vector3;  Project origin point; default (0,0,0)
    surfelement: an omf.SurfaceElement
    unicolor: boolean. True if the surface is to be colored with a unique color
    usercolor: Plotly rgb, or hex color code
    --------
    returns an instance of go.Surface class
    """
    if not isinstance(projO, vectormath.vector.Vector3) or  not isinstance(surfelement, omf.SurfaceElement):
        raise TypeError("Either the first argument is not a vectormath.vector.Vector3 or\n" +\
                          "the second one is not an omf.SurfaceElement instance")
    if not isinstance(surfelement.geometry, omf.SurfaceGridGeometry):
        raise TypeError("This surface element doesn't have an) omf.SurfaceGridGeometry")
    sgeom = surfelement.geometry
    surfO = sgeom.origin 
    axu = sgeom.axis_u 
    axv = sgeom.axis_v  
    axu.normalize()
    axv.normalize()
    axw = axu.cross(axv) # cross product axu x axv
    # Define the transformation matrix from (O; axu, axv, axw)
    # reference system to (O; i, j, k)
    T = np.array([axu, axv, axw]).transpose()
    x = np.concatenate(([0],  np.cumsum(sgeom.tensor_u)))
    y = np.concatenate(([0],  np.cumsum(sgeom.tensor_v)))
    x, y = np.meshgrid(x,y)
    z = sgeom.offset_w.array.reshape(x.shape)
    # Calculate the coordinates of the surface points  with respect to
    # the real world system: (O; i, j, k)
    coords = np.einsum('ik, kjl->ijl', T, np.stack((x, y, z)))
    x, y, z = coords
    # perform two consecutive translations to get the coordinates of the surface points
    # with respect to the project ref syst
    x = projO.x + surfO.x + x
    y = projO.y + surfO.y + y
    z = projO.z + surfO.z + z
    if unicolor:
        color = f"rgb{surfelement.color}" if usercolor is None else usercolor
        colorscale = [[0, color], [1, color]]
        showscale=False
    else: showscale = True    
    if len(surfelement.data) == 0:
        surfacecolor = None
    else:
        locations = [d.location  for d in surfelement.data]
        if 'vertices' in locations:
                dataindex = locations.index('vertices')
                surfacecolor = surfelement.data[dataindex].array.array.reshape(z.shape) 
        else:
            surfacecolor=None
        color=None 
    return go.Surface(x=x, y=y, z=z, 
                      surfacecolor=surfacecolor, 
                      colorscale=colorscale,
                      showscale=showscale,
                      colorbar_len=0.7)  
    
def cube_points(T, position3d, slength= np.ones(3)):
    """
    T : array of shape (3,3); defines the matrix of changing the coordinates from a basis to the 3d canonical basis
    position3d:  either a 3-list or an array of shape(3,) 
    slength 3-list or an array of shape (3,); it gives the cube side lengths in the direction x, y, respectively z
    -----------
    returns the rescaled and T-rotated cube at the given position
    """
    
    cube = np.array([[0, 0, 0], # template for a 3d cube; #each row, cube[k], defines the coordinates of a cube vertex 
                     [1, 0, 0], # with respect to the reference system (O; axis_u, axis_v, axis_w)
                     [1, 1, 0],
                     [0, 1, 0],
                     [0, 0, 1],
                     [1, 0, 1],
                     [1, 1, 1],
                     [0, 1, 1]], dtype=float)
                     
    cube = cube * slength # rescale the cube to sides of lengths slength[0], slength[1], slength[2]
    # calculate the coordinates of cube vertices with respect to the canonical base
    cube =(np.dot(T, cube.T)).T + np.asarray(position3d)  
    return cube


def triangulate_cube_faces(T, positions, slengths):
    """
    This function sets up all volume  voxels, extract their defining vertices as arrays of shape (3,)
    and reduces their number, by determining the array of unique vertices
    the voxel faces are triangularized and from their faces one extracts the lists of indices I, J, K
    to define a mesh3d;
    
    T: array of shape (3,3); matrix of change of coordinates
    positions:  array of shape (N, 3) containing all cube (voxel) positions of an object
    slengths: array of floats of shape (N,3), containing on each row the side lengths of the cube at the corresponding position
    -----
    returns an instance of the class go.Mesh3d
    """ 
    
    positions = np.asarray(positions)
    if positions.shape[1] != 3:
        raise ValueError("Wrong shape for positions of cubes in your data")
    if np.asarray(slengths).shape != positions.shape:
        slengths = np.ones(positions.shape)
    all_cubes = [cube_points(T, pos, length) for pos, length in zip(positions, slengths)]
    p, q, r = np.array(all_cubes).shape
    vertices, ixr = np.unique(np.array(all_cubes).reshape(p*q, r), return_inverse=True, axis=0)
    I = []
    J = []
    K = []
    # each triplei (i, j, k) defines a face/triangle
    for k in range(len(all_cubes)):
        I.extend(np.take(ixr, [8*k, 8*k+2, 8*k+4, 8*k+6,8*k+5, 8*k+2, 8*k+4, 8*k+3, 8*k+6, 8*k+3, 8*k+4, 8*k+1])) 
        J.extend(np.take(ixr, [8*k+1, 8*k+3, 8*k+5, 8*k+7, 8*k+1, 8*k+6, 8*k, 8*k+7, 8*k+2, 8*k+7, 8*k+5, 8*k])) 
        K.extend(np.take(ixr, [8*k+2, 8*k, 8*k+6, 8*k+4, 8*k+2, 8*k+5, 8*k+3, 8*k+4, 8*k+3, 8*k+6, 8*k+1, 8*k+4]))
  
    return vertices, I, J, K   
   
def volume_to_isosurface(projO, volelement):
    if not isinstance(projO, vectormath.vector.Vector3) or  not isinstance(volelement, omf.VolumeElement):
        raise TypeError("Either the first argument is not a vectormath.vector.Vector3 or\n" +\
                          "the second one is not an  instance of omf.VolumeElement")
    vgeom = volelement.geometry
    # Check if the axes of the volume system of coordinates are the world coordinates axes
    if (vgeom.axis_u -[1,0,0]).length> 1.e-06 or (vgeom.axis_v -[0,1,0]).length> 1.e-06 or\
        (vgeom.axis_w -[0,0,1]).length > 1.e-06:
        raise DataError('you cannot extract an isosurface from this volume because its reference system'+ \
          '\naxes are non-parallel with project axes')
    else:     
        vol_x = np.cumsum(vgeom.tensor_u) + vgeom.origin.x + projO.x
        vol_y = np.cumsum(vgeom.tensor_v) + vgeom.origin.y + projO.y
        vol_z = np.cumsum(vgeom.tensor_w) + vgeom.origin.z + projO.z 
        vol_x, vol_y, vol_z = np.meshgrid(vol_x, vol_y, vol_z, indexing='ij')
        values = volelement.data[0].array.array
        return vol_x.flatten(), vol_y.flatten(), vol_z.flatten(), values
        
            
def volelement_to_voxels(projO, volelement, colorscale = 'Viridis'):
    """
    projO: vectormath.vector.Vector3;  Project origin point; default (0,0,0)
    volelement:  omf.VolumeElement 
    ----
    returns a go.Mesh3d instance defined by a triangulations of VolumeGridGeometry (i.e. voxels)
    """
    if not isinstance(projO, vectormath.vector.Vector3) or  not isinstance(volelement, omf.VolumeElement):
        raise TypeError("Either the first argument is not a vectormath.vector.Vector3 or\n" +\
                          "the second one is not an omf.VolumeElement instance")
    if len(volelement.data) ==  0:
         raise DataError('This volume cannot be voxelized. No ScalarData provided')
    else:  
        locations  = [d.location for d in volelement.data]
        if 'cells' not in locations:
             raise DataError('This volume element has no cell data')
    vgeom = volelement.geometry
    volO = vgeom.origin
    axu = vgeom.axis_u  
    axv = vgeom.axis_v  
    axw = vgeom.axis_w
    axu.normalize() #normalize in place
    axv.normalize()
    axw.normalize()
    # Define transformation matrix from coordinates with respect to (O; axu, axv, axw) 
    # to coords with respect to the  real world system, (O; i, j, k)
    T = np.array([axu, axv, axw]).transpose()
    # calculate the side lengths for cubes representing voxels
    lx, ly, lz = np.meshgrid(vgeom.tensor_u, vgeom.tensor_v, vgeom.tensor_w, indexing='ij')
    sx, sy, sz = lx.shape
    slengths = np.stack((lx, ly, lz), axis=3).reshape(sx*sy*sz, 3)
    #calculate volume vertices and force it to start to (0,0,0)
    x = np.cumsum(vgeom.tensor_u)-vgeom.tensor_u[0] 
    y = np.cumsum(vgeom.tensor_v)-vgeom.tensor_v[0]
    z = np.cumsum(vgeom.tensor_w)-vgeom.tensor_w[0]
    x, y, z = np.meshgrid(x, y, z, indexing = 'ij')
    coords = np.einsum('ik, kjlm->ijlm', T, np.stack((x, y, z)))
    x, y, z = coords
    # perform two consecutive translations to get the coordinates of the surface points
    # with respect to the project ref syst
    x = projO.x + volO.x + x
    y = projO.y + volO.y + y
    z = projO.z + volO.z + z
    positions = np.stack((x, y, z), axis=3).reshape(slengths.shape)
    
    vals = volelement.data[0].array.array
    idx = np.where(np.isnan(vals))[0]    
    positions[idx] = np.array([None]*3) 
    vertices, i, j, k  = triangulate_cube_faces(T, positions, slengths)
    X, Y, Z = vertices.T

    return go.Mesh3d(x=X, y=Y, z=Z, i=i, j=j, k=k, colorscale=colorscale, 
                   flatshading=True,  
                   intensity=Z, 
                   colorbar_thickness=25, colorbar_len=0.6)
   
                                                        
