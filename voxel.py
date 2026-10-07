# SPDX-License-Identifier: GPL-3.0-or-later
"""PlayCanvas voxel 1.1 writer: sparse octree, engine-space floor/wall solids."""
import json
import math
import os
import struct
import tempfile
from collections import deque
from pathlib import Path

def encode_voxels(voxels, resolution, scene_bounds):
    if not voxels:
        raise ValueError('No collision voxels to export.')
    origin = [min(p[a] for p in voxels)//4*4 for a in range(3)]
    maximum = [(max(p[a] for p in voxels)//4+1)*4 for a in range(3)]
    blocks = {}
    for p in voxels:
        q = tuple(p[a]-origin[a] for a in range(3))
        key = tuple(v//4 for v in q)
        bit = q[0]%4 + (q[1]%4)*4 + (q[2]%4)*16
        blocks[key] = blocks.get(key, 0) | (1 << bit)
    depth = max(1, (max((maximum[a]-origin[a])//4 for a in range(3))-1).bit_length())
    root = {}
    for key, mask in blocks.items():
        node = root
        for shift in range(depth-1, 0, -1):
            octant = ((key[0]>>shift)&1)|(((key[1]>>shift)&1)<<1)|(((key[2]>>shift)&1)<<2)
            node = node.setdefault(octant, {})
        octant = (key[0]&1)|((key[1]&1)<<1)|((key[2]&1)<<2)
        node[octant] = mask
    nodes, leaf_data = [0], []
    queue = deque([(root, 0)])
    interiors = mixed = 0
    while queue:
        node, index = queue.popleft()
        if isinstance(node, int):
            if node == (1 << 64)-1:
                nodes[index] = 0xFF000000
            else:
                if len(leaf_data)//2 >= 1 << 24:
                    raise ValueError('Voxel format leaf limit exceeded.')
                nodes[index] = len(leaf_data)//2
                leaf_data.extend([node & 0xFFFFFFFF, node >> 32]); mixed += 1
        else:
            if len(nodes)+len(node) > 1 << 24:
                raise ValueError('Voxel format node limit exceeded.')
            first = len(nodes)
            nodes[index] = (sum(1<<octant for octant in node)<<24)|first
            interiors += 1
            for octant in sorted(node):
                queue.append((node[octant],len(nodes))); nodes.append(0)
    meta = dict(version='1.1',asset={'generator':'Lichtfeld Collision Mesh 0.4.0'},
                gridBounds={'min':[v*resolution for v in origin],'max':[v*resolution for v in maximum]},
                sceneBounds=scene_bounds,voxelResolution=resolution,leafSize=4,
                treeDepth=depth,numInteriorNodes=interiors,numMixedLeaves=mixed,
                nodeCount=len(nodes),leafDataCount=len(leaf_data))
    binary = struct.pack(f'<{len(nodes)+len(leaf_data)}I',*(nodes+leaf_data))
    return meta, binary

def voxelize(mesh, max_voxels=5000000):
    if 'voxels' in mesh:
        voxels = {tuple(v) for v in mesh['voxels']}
        if len(voxels)>max_voxels:
            raise ValueError('Voxel export exceeds five million solid voxels.')
        vertices = mesh['vertices']
        bounds = {'min':[min(p[a] for p in vertices) for a in range(3)],
                  'max':[max(p[a] for p in vertices) for a in range(3)]}
        return encode_voxels(voxels,mesh['spacing'],bounds)
    s = mesh['spacing']
    cells = {(x,z):h for x,z,h in mesh['cells']}
    corners = {}
    for (x,z),h in cells.items():
        for key in [(x,z),(x+1,z),(x,z+1),(x+1,z+1)]:
            corners.setdefault(key,[]).append(h)
    heights = {key:sum(v)/len(v) for key,v in corners.items()}
    voxels = set()
    for x,z in cells:
        values = [heights[k] for k in [(x,z),(x+1,z),(x,z+1),(x+1,z+1)]]
        # Conservative rasterization: solid floor layer, inward one-voxel walls.
        low = math.floor(min(values)/s)-1
        boundary = any(k not in cells for k in [(x-1,z),(x+1,z),(x,z-1),(x,z+1)])
        high = math.ceil((max(values)+(mesh['wallHeight'] if boundary else 0))/s)-1
        if len(voxels)+high-low+1 > max_voxels:
            raise ValueError('Voxel export exceeds 5 million solid voxels. Choose coarser mesh detail.')
        voxels.update((x,y,z) for y in range(low,high+1))
    vertices = mesh['vertices']
    bounds = {'min':[min(p[a] for p in vertices) for a in range(3)],
              'max':[max(p[a] for p in vertices) for a in range(3)]}
    return encode_voxels(voxels,s,bounds)

def write_voxel(mesh, path):
    path = Path(path)
    if not str(path).lower().endswith('.voxel.json'):
        raise ValueError('Choose a filename ending in .voxel.json.')
    binary_path = path.with_name(path.name[:-len('.voxel.json')]+'.voxel.bin')
    meta,binary = voxelize(mesh)
    # Prepare both files before replacing either destination; publish header last.
    staged = []
    try:
        for destination,data in [(binary_path,binary),
                                 (path,(json.dumps(meta,indent=2)+'\n').encode('utf-8'))]:
            with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.collision-',delete=False) as f:
                staged.append((Path(f.name),destination))
                f.write(data)
        for temporary,destination in staged:
            os.replace(temporary,destination)
    finally:
        for temporary,_ in staged:
            temporary.unlink(missing_ok=True)
    return binary_path


