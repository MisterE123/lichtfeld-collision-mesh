import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import numpy as np

root=Path(__file__).parents[1]
def module(name):
    spec=importlib.util.spec_from_file_location(name,root/(name+'.py'))
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result
geometry,voxel,glb=module('geometry'),module('voxel'),module('glb')

def decode(meta,binary):
    assert len(binary)==(meta['nodeCount']+meta['leafDataCount'])*4
    words=struct.unpack('<'+str(len(binary)//4)+'I',binary)
    nodes=words[:meta['nodeCount']];leaves=words[meta['nodeCount']:]
    result=set();seen=set();mixed=interior=0
    def visit(index,origin,size,depth):
        nonlocal mixed,interior
        assert index not in seen;seen.add(index)
        word=nodes[index]
        if word==0xFF000000:
            for x in range(size):
                for y in range(size):
                    for z in range(size):result.add(tuple(origin[a]+(x,y,z)[a] for a in range(3)))
        elif word>>24==0:
            assert depth==meta['treeDepth'] and size==4
            mask=leaves[2*word]|(leaves[2*word+1]<<32)
            assert 0<mask<(1<<64)-1
            mixed+=1
            for bit in range(64):
                if mask&(1<<bit):result.add((origin[0]+bit%4,origin[1]+(bit//4)%4,origin[2]+bit//16))
        else:
            interior+=1
            mask=word>>24;first=word&0xFFFFFF;half=size//2
            assert first>index
            for octant in range(8):
                if mask&(1<<octant):
                    child=first+(mask&((1<<octant)-1)).bit_count()
                    visit(child,tuple(origin[a]+((octant>>a)&1)*half for a in range(3)),half,depth+1)
    visit(0,(0,0,0),4*(1<<meta['treeDepth']),0)
    assert len(seen)==meta['nodeCount']
    assert mixed==meta['numMixedLeaves'] and interior==meta['numInteriorNodes']
    start=[round(v/meta['voxelResolution']) for v in meta['gridBounds']['min']]
    return {tuple(p[a]+start[a] for a in range(3)) for p in result}

# Full solid leaf plus mixed masks, high bits, distant sparse octants, negatives.
solid={(x,y,z) for x in range(-4,0) for y in range(4) for z in range(4)}
solid|={(15,7,3),(16,1,20),(-2,18,4),(0,0,0)}
meta,binary=voxel.encode_voxels(solid,.1,{'min':[-.4,0,0],'max':[1.7,1.9,2.1]})
assert meta['version']=='1.1' and decode(meta,binary)==solid

mesh=geometry.build(np.array([[-(x+.5)*.1,0,(z+.5)*.1] for x in range(5) for z in range(5)]),wall_height=.5)
meta,binary=voxel.voxelize(mesh)
occupied=decode(meta,binary)
assert (2,0,2) in occupied and (2,1,2) not in occupied
assert (2,4,2) in occupied # ceiling spans interior
assert (0,4,2) in occupied and (0,5,2) not in occupied
assert (5,0,2) not in occupied
try:voxel.voxelize(mesh,max_voxels=1)
except ValueError:pass
else:raise AssertionError('Voxel budget not enforced')
with tempfile.TemporaryDirectory() as tmp:
    header=Path(tmp)/'room-collision.voxel.json'
    bin_path=voxel.write_voxel(mesh,header)
    assert bin_path.name=='room-collision.voxel.bin'
    assert decode(json.loads(header.read_text()),bin_path.read_bytes())==occupied
    target=Path(tmp)/'room-collision.glb';glb.write_glb(mesh,target)
    data=target.read_bytes();magic,version,total=struct.unpack_from('<III',data)
    assert (magic,version,total)==(0x46546C67,2,len(data))
    n,kind=struct.unpack_from('<II',data,12);assert kind==0x4E4F534A and n%4==0
    doc=json.loads(data[20:20+n]);offset=20+n
    size,kind=struct.unpack_from('<II',data,offset);assert kind==0x004E4942
    blob=data[offset+8:];assert len(blob)==size==doc['buffers'][0]['byteLength']
    views=doc['bufferViews']
    vertices=np.frombuffer(blob,dtype='<f4',count=len(mesh['vertices'])*3).reshape(-1,3)
    assert np.allclose(vertices,np.asarray(mesh['vertices'])*[-1,-1,1])
    # Compare actual OBJ output to GLB vertices in the same native world frame.
    obj=Path(tmp)/'room-collision.obj';geometry.write_obj(mesh,obj)
    obj_vertices=np.array([[float(v) for v in line.split()[1:]] for line in obj.read_text().splitlines() if line.startswith('v ')])
    assert np.allclose(vertices,obj_vertices)
    assert not any(k in doc['nodes'][0] for k in ['rotation','translation','scale','matrix'])
    normal_offset=views[1]['byteOffset']
    normals=np.frombuffer(blob,dtype='<f4',offset=normal_offset,count=len(vertices)*3).reshape(-1,3)
    assert np.isfinite(normals).all() and np.allclose(np.linalg.norm(normals,axis=1),1)
    faces=np.frombuffer(blob,dtype='<u4',offset=views[2]['byteOffset']).reshape(-1,3)
    assert np.array_equal(faces,mesh['faces'])
    # Asymmetric translated fixture catches axis swaps and accidental origin rotation.
    translated=geometry.build(np.array([[2.35,-1.25,-3.15],[2.45,-1.25,-3.15]]),wall_height=.5)
    for playcanvas_frame in [False,True]:
        glb.write_glb(translated,target,playcanvas_frame=playcanvas_frame)
        data=target.read_bytes();n=struct.unpack_from('<I',data,12)[0]
        exported=np.frombuffer(data,dtype='<f4',offset=20+n+8,count=len(translated['vertices'])*3).reshape(-1,3)
        expected=np.asarray(translated['vertices'])*(1 if playcanvas_frame else np.array([-1,-1,1]))
        assert np.allclose(exported,expected)
print('Export checks passed: independent octree decoding, solid/mixed leaves, masks, coordinates, floor/wall occupancy, file pairs, limits, GLB chunks and geometry.')

