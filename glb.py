# SPDX-License-Identifier: GPL-3.0-or-later
"""Standalone glTF 2.0 binary collision mesh, native world frame by default."""
import json
import struct
from pathlib import Path
import numpy as np

def write_glb(mesh, path, playcanvas_frame=False):
    vertices = np.asarray(mesh['vertices'],dtype='<f4')
    if not playcanvas_frame:
        # Undo the generator's internal Z-180 rotation, matching scene.add_mesh
        # and OBJ world vertices. Translation/scale are already baked by generate.
        vertices = (vertices * np.array([-1,-1,1],dtype='<f4')).astype('<f4')
    faces = np.asarray(mesh['faces'],dtype='<u4')
    normals = np.zeros_like(vertices)
    triangles = vertices[faces]
    face_normals = np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    for i in range(3):
        np.add.at(normals,faces[:,i],face_normals)
    lengths = np.linalg.norm(normals,axis=1)
    normals /= np.maximum(lengths[:,None],1e-20)
    positions = vertices.tobytes()
    normal_bytes = normals.astype('<f4').tobytes()
    indices = faces.tobytes()
    binary = positions+normal_bytes+indices
    views = [dict(buffer=0,byteOffset=0,byteLength=len(positions),target=34962),
             dict(buffer=0,byteOffset=len(positions),byteLength=len(normal_bytes),target=34962),
             dict(buffer=0,byteOffset=len(positions)+len(normal_bytes),byteLength=len(indices),target=34963)]
    document = dict(asset={'version':'2.0','generator':'Lichtfeld Collision Mesh 0.4.0'},
                    scene=0,scenes=[{'nodes':[0]}],nodes=[{'mesh':0,'name':'Collision mesh'}],
                    meshes=[{'primitives':[{'attributes':{'POSITION':0,'NORMAL':1},'indices':2,'material':0,'mode':4}]}],
                    materials=[{'name':'Collision preview','doubleSided':True,
                                'pbrMetallicRoughness':{'baseColorFactor':[.2,.7,.9,1.],'metallicFactor':0.,'roughnessFactor':1.}}],
                    buffers=[{'byteLength':len(binary)}],bufferViews=views,
                    accessors=[{'bufferView':0,'componentType':5126,'count':len(vertices),'type':'VEC3',
                                'min':vertices.min(axis=0).tolist(),'max':vertices.max(axis=0).tolist()},
                               {'bufferView':1,'componentType':5126,'count':len(vertices),'type':'VEC3'},
                               {'bufferView':2,'componentType':5125,'count':faces.size,'type':'SCALAR'}])
    header = json.dumps(document,separators=(',',':')).encode('utf-8')
    header += b' '*((-len(header))%4)
    binary += b'\0'*((-len(binary))%4)
    output = struct.pack('<III',0x46546C67,2,12+8+len(header)+8+len(binary))
    output += struct.pack('<II',len(header),0x4E4F534A)+header
    output += struct.pack('<II',len(binary),0x004E4942)+binary
    Path(path).write_bytes(output)


