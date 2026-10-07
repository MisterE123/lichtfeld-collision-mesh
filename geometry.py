# SPDX-License-Identifier: GPL-3.0-or-later
"""Sparse 3D voxel enclosure with merged planar surface faces."""
import math
import numpy as np

def opacity_keep(logits, threshold):
    """Raw Lichtfeld opacity is a logit; compare in logit space without overflow."""
    if not 0 < threshold < 1:
        raise ValueError('Opacity cutoff must be between zero and one.')
    values=np.asarray(logits)
    return np.isfinite(values) & (values >= math.log(threshold/(1-threshold)))

def smooth_surface(voxels, spacing, ceiling, roof_layer, smoothing=.35,
                   max_samples=32000000, max_faces=2000000):
    from scipy.ndimage import gaussian_filter
    from skimage.measure import marching_cubes
    if not 0 <= smoothing <= .45:
        raise ValueError('Smoothing must be between 0 and 0.45 voxels.')
    positions=np.asarray(sorted(voxels),dtype=np.int64)
    low=positions.min(axis=0)-2
    shape=positions.max(axis=0)-low+3
    if math.prod(map(int,shape))>max_samples:
        raise ValueError('Smooth remesh grid exceeds 32 million samples. Choose coarser detail or a smaller region.')
    field=np.zeros(tuple(shape),dtype=np.float32)
    indices=positions-low
    field[tuple(indices.T)]=1.
    if smoothing:
        field=gaussian_filter(field,sigma=smoothing,mode='constant',cval=0.,truncate=3.)
    vertices,faces,_,_=marching_cubes(field,level=.5,spacing=(spacing,)*3,
                                     allow_degenerate=False,method='lewiner')
    if len(faces)>max_faces:
        raise ValueError('Smooth remesh exceeds two million triangles. Choose coarser detail.')
    # Scalar samples live at voxel centers, not grid corners.
    vertices=vertices.astype(np.float64)+(low+.5)*spacing
    roof_bottom=roof_layer*spacing
    top_region=vertices[:,1]>=roof_bottom
    vertices[top_region,1]=roof_bottom+(vertices[top_region,1]-roof_bottom)*(ceiling-roof_bottom)/spacing
    top=vertices[:,1].max()
    vertices[np.isclose(vertices[:,1],top,atol=1e-7,rtol=0),1]=ceiling
    # Ensure glTF and OBJ use outward triangle winding.
    triangles=vertices[faces]
    volume=np.einsum('ij,ij->i',triangles[:,0],np.cross(triangles[:,1],triangles[:,2])).sum()
    if volume<0:faces=faces[:,[0,2,1]]
    return vertices.tolist(),faces.tolist()

def surface(voxels, spacing, ceiling, roof_layer, max_quads=1000000):
    planes = {}
    for p in voxels:
        for axis in range(3):
            other = [a for a in range(3) if a != axis]
            for sign in [-1,1]:
                q = list(p); q[axis] += sign
                if tuple(q) not in voxels:
                    key = (axis,sign,p[axis]+(sign>0))
                    planes.setdefault(key,set()).add((p[other[0]],p[other[1]]))
    vertices, faces, lookup = [], [], {}
    def vertex(p):
        point = [v*spacing for v in p]
        if p[1] == roof_layer+1: point[1] = ceiling
        key = tuple(point)
        if key not in lookup:
            lookup[key]=len(vertices);vertices.append(point)
        return lookup[key]
    quads=0
    for (axis,sign,level),tiles in sorted(planes.items()):
        other = [a for a in range(3) if a != axis]
        for u,v in sorted(tiles):
            if (u,v) not in tiles:continue
            width=1
            while (u+width,v) in tiles:width+=1
            height=1
            while all((u+i,v+height) in tiles for i in range(width)):height+=1
            for i in range(width):
                for j in range(height):tiles.remove((u+i,v+j))
            corners=[]
            for a,b in [(u,v),(u+width,v),(u+width,v+height),(u,v+height)]:
                p=[0,0,0];p[axis]=level;p[other[0]]=a;p[other[1]]=b;corners.append(vertex(p))
            if sign != (-1 if axis==1 else 1):corners.reverse()
            a,b,c,d=corners;faces.extend([[a,b,c],[a,c,d]])
            quads+=1
            if quads>max_quads:raise ValueError('Surface exceeds one million quads. Choose coarser detail.')
    return vertices,faces

def build(points, spacing=.1, wall_height=3., max_cells=100000, max_voxels=5000000,
          smooth=True, smoothing=.35):
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or not len(points):
        raise ValueError('Select ground/scene splats first.')
    if not np.isfinite(points).all() or not .025<=spacing<=1 or not .5<=wall_height<=20:
        raise ValueError('Invalid coordinates, spacing, or wall height.')
    points=points*[-1,-1,1]
    selected=np.unique(np.floor(points/spacing).astype(np.int64),axis=0)
    if len(selected)>max_voxels:raise ValueError('Too many selected voxels. Choose coarser detail.')
    solids={tuple(map(int,p)) for p in selected}
    floor={}
    for x,y,z in solids:floor[x,z]=min(floor.get((x,z),y),y)
    if len(floor)>max_cells:raise ValueError('Mesh exceeds 100,000 footprint cells. Choose coarser detail.')
    highest=float(points[:,1].max())
    ceiling=highest+wall_height
    roof_layer=math.ceil(ceiling/spacing)-1
    boundary_edges=0
    for (x,z),bottom in floor.items():
        exposed=sum(k not in floor for k in [(x-1,z),(x+1,z),(x,z-1),(x,z+1)])
        boundary_edges+=exposed
        if exposed:
            count=roof_layer-bottom+1
            if len(solids)+count>max_voxels:raise ValueError('Enclosure exceeds five million voxels. Choose coarser detail.')
            solids.update((x,y,z) for y in range(bottom,roof_layer+1))
        else:solids.add((x,roof_layer,z))
        if len(solids)>max_voxels:raise ValueError('Enclosure exceeds five million voxels. Choose coarser detail.')
    vertices,faces=(smooth_surface(solids,spacing,ceiling,roof_layer,smoothing)
                    if smooth else surface(solids,spacing,ceiling,roof_layer))
    return {'version':2,'spacing':spacing,'wallHeight':wall_height,
            'ceilingHeight':ceiling,'highestSelectedHeight':highest,
            'surfaceMode':'smooth' if smooth else 'blocky','smoothing':smoothing if smooth else 0.,
            'cells':[[x,z,y*spacing] for (x,z),y in sorted(floor.items())],
            'voxels':sorted(solids),'vertices':vertices,'faces':faces,'boundaryEdges':boundary_edges}

def vertex_normals(mesh):
    vertices=np.asarray(mesh['vertices'],dtype=np.float32)
    faces=np.asarray(mesh['faces'],dtype=np.int32)
    triangles=vertices[faces]
    normals=np.zeros_like(vertices)
    n=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    for i in range(3):np.add.at(normals,faces[:,i],n)
    normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-20)
    return normals

def write_obj(mesh, path):
    with open(path,'w',encoding='utf-8') as f:
        f.write('# Lichtfeld collision mesh; native world coordinates; meters\n')
        for x,y,z in mesh['vertices']:f.write(f'v {-x:.9g} {-y:.9g} {z:.9g}\n')
        if mesh.get('surfaceMode')=='smooth':
            for x,y,z in vertex_normals(mesh):f.write(f'vn {-x:.9g} {-y:.9g} {z:.9g}\n')
            f.write('s 1\n')
            for a,b,c in mesh['faces']:f.write(f'f {a+1}//{a+1} {b+1}//{b+1} {c+1}//{c+1}\n')
        else:
            for a,b,c in mesh['faces']:f.write(f'f {a+1} {b+1} {c+1}\n')
