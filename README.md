# Selected-splat Collision Mesh for Lichtfeld Studio

Standalone smooth 3D voxel mesher for selected splats, with perimeter walls and a ceiling. Mesh detail presets: 0.05, **0.1**, 0.2, 0.5, and 1 meter. Wall height defaults to 3 meters.

## Use

1. Restart Lichtfeld and open **Collision Mesh**.
2. Select ground covering the desired area, plus floating blobs or other geometry to include. For rooms, end the ground selection at open doorways. Select the ground beneath floating objects too, so it defines the enclosure footprint.
3. Choose mesh detail, **Voxel surface** (Smooth remesh is the default), smoothing amount, and wall height. Optionally enable **Ignore transparent splats / floaters** and set **Minimum splat opacity** (default 0.15). Then **Generate from selected splats**. The mesh appears in the scene tree.
4. Choose **Export format**: OBJ, GLB, or PlayCanvas voxel. **Export collision mesh** opens the Windows save picker with `model-name-collision` and the appropriate suffix. All three formats are also in **Save as type**. Cancel writes nothing. Delete or hide old generated meshes after regeneration.

## Meshing and ceiling

Selected centers are rasterized into a sparse 3D grid, preserving separate height layers, floating blobs, and overhangs. The generator adds solid walls along the perimeter and holes of the selected horizontal footprint, plus a ceiling over that footprint. Disconnected footprint components remain separate. Ceiling height is **highest selected splat center + wall height**, measured upward along native -Y. OBJ and GLB ceiling surfaces end at that exact height. Selecting a high floating blob raises the common ceiling accordingly.

**Smooth remesh** extracts an interpolated surface with Lewiner marching cubes after a light Gaussian filter of the 3D occupancy field. This rounds voxel edges in the actual geometry, similar in intent to Blender's voxel remesh; it is not Blender's implementation. OBJ, GLB, and the scene preview have smooth vertex normals. Smoothing defaults to 0.35 voxels and can be set from 0 to 0.45. A **Blocky merged faces** mode remains available for simpler planar meshes. Voxel size controls detail and triangle count. Thin shapes and edges can change slightly during smoothing. The smooth mesher limits its dense grid to 32 million samples and its surface to two million triangles; select a smaller area or coarser detail if it exceeds that budget. Marching cubes reference: https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.marching_cubes

**Ignore transparent splats / floaters** is an optional pre-mesh filter, enabled by default. It removes all selected splats whose individual opacity is below the cutoff, including low-opacity floating artifacts. Raw opacity logits are converted by an equivalent stable threshold comparison. This uses per-splat opacity rather than accumulated rendered opacity, and does not identify floaters by connectivity: dense low-opacity surfaces may also be excluded, while high-opacity floating objects remain. Excluded splats do not affect the footprint or the ceiling height. The filter changes only the generated collision mesh, not the splat scene or selection.

The generator samples selected centers rather than each Gaussian's full ellipsoid. Sparse selection can leave floor gaps; select dense ground or use coarser spacing. Cells with multiple selected heights retain separate layers. Perimeter walls occupy the inside boundary voxel column. Binary voxel roofs are quantized to voxel size and may extend less than one voxel beyond the exact mesh ceiling. The binary format remains solid/empty voxels: smoothing applies to OBJ/GLB/preview geometry, while voxel export retains the underlying occupancy grid.


Scene units must be meters, with native **-Y up**. Limits: 100,000 footprint cells, five million solid voxels, and one million merged surface quads. The extension creates and exports geometry; the consuming engine handles collision and navigation.

## Export formats

- **OBJ**: native Lichtfeld world coordinates.
- **GLB**: standalone glTF 2.0 binary mesh, normals, and preview material. **GLB coordinate frame** defaults to **Lichtfeld world (same as OBJ)**, matching the generated scene mesh. Node transforms are already baked; the GLB node has no additional transform. An explicit PlayCanvas option rotates 180 degrees about Z.
- **PlayCanvas voxel**: version 1.1 `.voxel.json` / `.voxel.bin` file pair in PlayCanvas engine coordinates. Uses exactly the mesher's solid voxel set, including floating blobs, walls, and roof. Keep the two files together with the same stem. Mesh detail is voxel size. Informational scene bounds describe the collision proxy, not the entire source Gaussian scene. Repeated exports replace both files.

Voxel output follows the [PlayCanvas voxel format specification](https://developer.playcanvas.com/user-manual/splat-transform/voxel-format/). Export a GLB separately if required. Earlier GLBs always applied the PlayCanvas rotation; native alignment now defaults to the Lichtfeld frame.

## Install and verification

Copy `lichtfeld_collision_mesh` into `%USERPROFILE%\.lichtfeld\plugins\` and install its NumPy, SciPy, and scikit-image dependencies through Lichtfeld's plugin setup. No other extension is required.

Smooth geometry tests verify rounded vertices, closed manifold edges, outward winding, floating components, air gaps, opacity filtering, filtered ceiling height, and dense-grid limits. Geometry checks cover the ceiling formula, holes, disconnected regions, floating voxels with air underneath, nondegenerate triangles, signed enclosed volume, face merging, and budgets. Export checks independently decode octrees and validate GLB geometry and native/PlayCanvas frames. Plugin validation passes. Live operation and loading the exports in the destination engine remain to be verified.

License: GPL-3.0-or-later.

