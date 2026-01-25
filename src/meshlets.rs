use meshopt::{VertexDataAdapter, build_meshlets, simplify, typed_to_bytes, SimplifyOptions};

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VertexStorage {
    // Storage buffers align vec3 to 16 bytes in WGSL, so use vec4 and padding
    // to keep the Rust/WGSL layouts compatible.
    pub position: [f32; 4],
    pub tex_coords: [f32; 2],
    pub _pad: [f32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshletDesc {
    pub vertex_offset: u32,
    pub vertex_count: u32,
    pub index_offset: u32,
    pub index_count: u32,
}

pub struct MeshletData {
    pub vertex_storage: Vec<VertexStorage>,
    pub meshlet_vertices: Vec<u32>,
    pub meshlet_indices: Vec<u32>,
    pub meshlet_descs: Vec<MeshletDesc>,
}

/// Load an OBJ mesh and build meshlets using meshoptimizer.
/// Returns:
/// - vertex_storage: full vertex buffer for mesh shader
/// - meshlet_vertices: local->global vertex remap per meshlet
/// - meshlet_indices: local triangle indices per meshlet
/// - meshlet_descs: offsets/counts into the meshlet buffers
/// `lod_ratio` in (0, 1] controls simplification amount (1.0 = full detail).
pub fn load_obj_meshlets(
    path: &str,
    max_meshlet_verts: usize,
    max_meshlet_prims: usize,
    lod_ratio: f32,
) -> anyhow::Result<MeshletData> {
    let (models, _materials) = tobj::load_obj(
        path,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
    )?;
    let mesh = &models
        .first()
        .ok_or_else(|| anyhow::anyhow!("No meshes found in OBJ"))?
        .mesh;

    let vertex_count = mesh.positions.len() / 3;
    let mut vertex_storage = Vec::with_capacity(vertex_count);
    for i in 0..vertex_count {
        let pos = [
            mesh.positions[i * 3],
            mesh.positions[i * 3 + 1],
            mesh.positions[i * 3 + 2],
            1.0,
        ];
        let tex_coords = if mesh.texcoords.len() >= (i * 2 + 2) {
            [mesh.texcoords[i * 2], mesh.texcoords[i * 2 + 1]]
        } else {
            [0.0, 0.0]
        };
        vertex_storage.push(VertexStorage {
            position: pos,
            tex_coords,
            _pad: [0.0, 0.0],
        });
    }

    let mut indices = mesh.indices.clone();
    if lod_ratio < 1.0 {
        let target_index_count =
            ((indices.len() as f32) * lod_ratio).floor() as usize / 3 * 3;
        let target_index_count = target_index_count.max(3);
        let vertex_adapter = VertexDataAdapter::new(
            typed_to_bytes(&mesh.positions),
            3 * std::mem::size_of::<f32>(),
            0,
        )?;
        indices = simplify(
            &indices,
            &vertex_adapter,
            target_index_count,
            1e-3,
            SimplifyOptions::None,
            None,
        );
    }
    // Provide meshopt with a view of raw position data (xyz f32).
    let vertex_adapter = VertexDataAdapter::new(
        typed_to_bytes(&mesh.positions),
        3 * std::mem::size_of::<f32>(),
        0,
    )?;
    // Build meshlets with meshoptimizer; cone_weight=0 disables cone culling data.
    let meshlets = build_meshlets(
        &indices,
        &vertex_adapter,
        max_meshlet_verts,
        max_meshlet_prims,
        0.0,
    );

    let mut meshlet_descs = Vec::with_capacity(meshlets.len());
    for meshlet in &meshlets.meshlets {
        meshlet_descs.push(MeshletDesc {
            vertex_offset: meshlet.vertex_offset as u32,
            vertex_count: meshlet.vertex_count as u32,
            index_offset: meshlet.triangle_offset as u32,
            index_count: meshlet.triangle_count as u32,
        });
    }

    let meshlet_vertices = meshlets.vertices;
    let meshlet_indices: Vec<u32> = meshlets.triangles.iter().map(|&t| t as u32).collect();

    Ok(MeshletData {
        vertex_storage,
        meshlet_vertices,
        meshlet_indices,
        meshlet_descs,
    })
}
