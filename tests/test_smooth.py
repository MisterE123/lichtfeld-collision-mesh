import importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('geometry',Path(__file__).parents[1]/'geometry.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
ground=[[-(x+.5)*.1,-.02,(z+.5)*.1] for x in range(7) for z in range(7)]
points=np.array(ground+[[-.35,-1.23,.35],[-.45,-1.23,.35]])
m=g.build(points,wall_height=.7)
assert m['surfaceMode']=='smooth'
v=np.asarray(m['vertices']);f=np.asarray(m['faces'])
assert np.isclose(v[:,1].max(),1.93)
triangles=v[f];cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
assert (np.linalg.norm(cross,axis=1)>1e-12).all()
assert np.einsum('ij,ij->i',triangles[:,0],np.cross(triangles[:,1],triangles[:,2])).sum()>0
# A closed manifold: every undirected mesh edge has exactly two triangle users.
edges=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
_,counts=np.unique(edges,axis=0,return_counts=True)
assert (counts==2).all()
# Rounded edges actually change geometry, rather than just shading blocky faces.
fraction=v/.1-np.round(v/.1)
assert np.any((abs(fraction[:,0])>1e-4)&(abs(fraction[:,2])>1e-4))
assert np.any((v[:,1]>1.15)&(v[:,1]<1.35)) # floating blob survives
assert not np.any((v[:,0]>.3)&(v[:,0]<.4)&(v[:,2]>.3)&(v[:,2]<.4)&(v[:,1]>.4)&(v[:,1]<1.0))
# Filtering a high transparent floater prevents it raising the roof.
with_floater=np.concatenate([points,[[-.35,-8.,.35]]])
alpha=np.array([.9]*len(points)+[.02])
logits=np.log(alpha/(1-alpha))
keep=g.opacity_keep(logits,.15)
assert keep.sum()==len(points) and not keep[-1]
assert np.isclose(g.build(with_floater[keep],wall_height=.7)['ceilingHeight'],1.93)
assert np.array_equal(g.opacity_keep(np.array([-1000.,1000.,np.nan]),.15),[False,True,False])
try:g.smooth_surface({(0,0,0)},.1,1.,9,max_samples=1)
except ValueError:pass
else:raise AssertionError('Dense grid budget not enforced')
print('Smooth checks passed: rounded geometry, closed manifold, outward winding, exact ceiling, floating component, air gap, opacity/logit filter, filtered ceiling and grid budget.')


