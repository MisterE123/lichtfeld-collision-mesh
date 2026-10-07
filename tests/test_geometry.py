import importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('geometry',Path(__file__).parents[1]/'geometry.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
points=np.array([[-(x+.5)*.1,0,(z+.5)*.1] for x in range(5) for z in range(5) if (x,z)!=(2,2)])
m=g.build(points,smooth=False)
assert len(m['cells'])==24 and m['boundaryEdges']==24
assert np.isclose(m['ceilingHeight'],3.)
assert np.isclose(max(p[1] for p in m['vertices']),3.)
assert (2,29,2) not in set(map(tuple,m['voxels'])) # hole stays outside enclosure
assert all(len(set(face))==3 for face in m['faces'])
assert np.isfinite(m['vertices']).all()
disjoint=g.build(np.array([[-.05,0,.05],[-1.05,0,.05]]),smooth=False)
assert len(disjoint['cells'])==2 and disjoint['boundaryEdges']==8
try:g.build(points,max_cells=2)
except ValueError:pass
else:raise AssertionError('Budget not enforced')
ground=[[-(x+.5)*.1,-.02,(z+.5)*.1] for x in range(7) for z in range(7)]
blob=[[-.35,-1.23,.35],[-.45,-1.23,.35]]
floating=g.build(np.array(ground+blob),wall_height=.7,smooth=False)
solids=set(map(tuple,floating['voxels']))
assert np.isclose(floating['ceilingHeight'],1.93)
assert np.isclose(max(p[1] for p in floating['vertices']),1.93)
assert (3,12,3) in solids and (3,6,3) not in solids # no artificial support pillar
assert (3,19,3) in solids # solid ceiling
triangles=np.asarray(floating['vertices'])[np.asarray(floating['faces'])]
area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
assert (area>0).all()
volume=np.einsum('ij,ij->i',triangles[:,0],np.cross(triangles[:,1],triangles[:,2])).sum()/6
expected=len(solids)*.1**3-len(floating['cells'])*.1**2*(2.-1.93)
assert np.isclose(volume,expected), (volume,expected)
assert len(floating['faces']) < len(solids)*2 # merged faces reduce detail
print('Mesh checks passed: ceiling formula and closure, hole walls, floating layers and air gap, signed volume, merged faces, disconnected areas, finite triangles, budget guard.')


