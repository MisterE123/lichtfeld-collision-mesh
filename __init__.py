# SPDX-License-Identifier: GPL-3.0-or-later
from pathlib import Path
import numpy as np
import lichtfeld as lf
from .geometry import build, write_obj, opacity_keep, vertex_normals
from .save_dialog import collision_filename, choose_export_path
from .voxel import write_voxel
from .glb import write_glb

active = None
status = 'Select floor or ground splats, then generate a collision mesh.'
detail = 1
wall_height = 3.
surface_mode = 0
smoothing = .35
ignore_transparent = True
opacity_cutoff = .15
export_format = 0
glb_frame = 0
export_name = 'splat-model-collision.obj'

def generate():
    global active, status, export_name
    scene = lf.get_scene()
    if scene is None or lf.context().is_training:
        raise ValueError('Load a scene and pause training first.')
    nodes = [n for n in scene.get_nodes() if n.type == lf.scene.NodeType.SPLAT]
    total = sum(n.gaussian_count for n in nodes)
    mask = scene.selection_mask
    if mask is None:
        raise ValueError('Select floor/ground splats first.')
    mask = np.asarray(mask.cpu().numpy()).reshape(-1).astype(bool)
    if len(mask) != total:
        raise ValueError('Selection layout does not match this scene.')
    chunks, source_names, offset, ignored = [], [], 0, 0
    for n in nodes:
        selected = mask[offset:offset+n.gaussian_count].copy(); offset += n.gaussian_count
        if not scene.is_node_effectively_visible(n.id) or not selected.any():
            continue
        data = n.splat_data()
        if data.has_deleted_mask():
            selected &= ~np.asarray(data.deleted.cpu().numpy()).reshape(-1).astype(bool)
        if ignore_transparent:
            candidates=np.flatnonzero(selected)
            logits=np.asarray(data.opacity_raw.cpu().numpy()).reshape(-1)[candidates]
            keep=opacity_keep(logits,opacity_cutoff)
            ignored+=int((~keep).sum())
            selected[candidates[~keep]]=False
        xyz = np.asarray(data.means_raw.cpu().numpy())[selected]
        if not len(xyz):
            continue
        source_names.append(n.name)
        matrix = np.asarray(n.world_transform).reshape(4,4)
        chunks.append(xyz @ matrix[:3,:3].T + matrix[:3,3])
    if not chunks or not sum(len(c) for c in chunks):
        raise ValueError('No selected splats remain. Lower the opacity cutoff or disable the transparent-splat filter.')
    mesh = build(np.concatenate(chunks), [.05,.1,.2,.5,1.][detail], wall_height,
                 smooth=surface_mode == 0,smoothing=smoothing)
    # Add the generated mesh in native world coordinates.
    vertices = np.asarray(mesh['vertices'],dtype=np.float32)*[-1,-1,1]
    normals = lf.Tensor.from_numpy((vertex_normals(mesh)*[-1,-1,1]).astype(np.float32)) if surface_mode == 0 else None
    scene.add_mesh('Selected-splat collision mesh',lf.Tensor.from_numpy(vertices.astype(np.float32)),
                   lf.Tensor.from_numpy(np.asarray(mesh['faces'],dtype=np.int32)),normals=normals)
    active = mesh
    export_name = collision_filename(source_names[0])
    status = f"Generated {mesh['surfaceMode']} mesh: {len(mesh['faces']):,} triangles. Ignored {ignored:,} transparent splats. Ceiling at {mesh['ceilingHeight']:.3f} m."

class CollisionPanel(lf.ui.Panel):
    id = 'lichtfeld_collision_mesh.panel'
    label = 'Collision Mesh'
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 112
    template = str(Path(__file__).with_name('panel.rml'))
    def draw(self, ui):
        global detail, wall_height, active, status, export_format, glb_frame
        global surface_mode, smoothing, ignore_transparent, opacity_cutoff
        ui.heading('Selected splats → collision mesh')
        ui.text_wrapped('Select ground plus any floating blobs to include. 3D voxels preserve separate height layers. Walls close the selection footprint; a ceiling closes the top at highest selected splat + wall height. Native -Y is up; scene units must be meters.')
        _, detail = ui.combo('Mesh detail',detail,['0.05 m','0.1 m (default)','0.2 m','0.5 m','1 m'])
        _, surface_mode = ui.combo('Voxel surface',surface_mode,['Smooth remesh (default)','Blocky merged faces'])
        if surface_mode == 0:
            _, smoothing = ui.slider_float('Surface smoothing (voxels)',smoothing,0.,.45)
        _, ignore_transparent = ui.checkbox('Ignore transparent splats / floaters',ignore_transparent)
        if ignore_transparent:
            _, opacity_cutoff = ui.slider_float('Minimum splat opacity',opacity_cutoff,.01,.95)
            ui.text_wrapped('Filters selected splats below this opacity before meshing and calculating the ceiling. High-opacity floating objects remain.')
        _, wall_height = ui.slider_float('Wall height (m)',wall_height,.5,10.)
        _, export_format = ui.combo('Export format',export_format,['OBJ mesh','PlayCanvas voxel (.voxel.json + .voxel.bin)','GLB mesh'])
        _, glb_frame = ui.combo('GLB coordinate frame',glb_frame,['Lichtfeld world (same as OBJ)','PlayCanvas splat import (180 degrees about Z)'])
        if ui.button('Generate from selected splats'):
            try: generate()
            except Exception as exc: status = str(exc); lf.log.error(status)
        if active is not None:
            if export_format == 1:
                ui.text_wrapped('Voxel export uses the same 3D solids as the mesher, including floating blobs and ceiling. Mesh detail is voxel size. Keep both exported files together.')
            if ui.button('Export collision mesh'):
                try:
                    if active is None: raise ValueError('Generate a mesh first.')
                    suffix = ['.obj','.voxel.json','.glb'][export_format]
                    name = export_name[:-4]+suffix
                    chosen = choose_export_path(name,voxel=export_format == 1,glb=export_format == 2)
                    if chosen:
                        selected_path, export_format = chosen
                        path = Path(selected_path)
                        if export_format == 1:
                            binary_path = write_voxel(active,path)
                            status = f'Exported {path} and {binary_path.name}'
                        elif export_format == 2:
                            write_glb(active,path,playcanvas_frame=glb_frame == 1)
                            status = f'Exported {path}'
                        else:
                            write_obj(active,path)
                            status = f'Exported {path}'
                except Exception as exc: status = str(exc)
        ui.text_wrapped(status)
        ui.text_wrapped('Export the mesh for use as collision geometry in your target viewer or engine.')

def on_load(): lf.register_class(CollisionPanel)
def on_unload():
    global active
    active = None
    lf.unregister_class(CollisionPanel)

