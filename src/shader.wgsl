// Mesh shaders are an experimental WGSL extension in wgpu.
// This shader renders meshlets (small clusters) generated on the CPU.
enable wgpu_mesh_shader;

struct CameraUniform {
    view_proj: mat4x4<f32>,
}

// Vertex data comes from a storage buffer (no vertex stage).
struct VertexInput {
    position: vec4<f32>,
    tex_coords: vec2<f32>,
    _pad: vec2<f32>,
}

// Meshlet descriptor points into meshlet vertex/triangle buffers.
// vertex_offset/count index meshlet_vertices.
// index_offset/count index meshlet_indices (triangles).
struct MeshletDesc {
    vertex_offset: u32,
    vertex_count: u32,
    index_offset: u32,
    index_count: u32,
}

// Meshlet buffers (group 0).
// - vertices: full vertex buffer
// - meshlet_vertices: local->global vertex remap per meshlet
// - meshlet_indices: local triangle indices per meshlet
// - meshlets: meshlet descriptors (offsets/counts)
@group(0) @binding(0) var<storage, read> vertices: array<VertexInput>;
@group(0) @binding(1) var<storage, read> meshlet_vertices: array<u32>;
@group(0) @binding(2) var<storage, read> meshlet_indices: array<u32>;
@group(0) @binding(3) var<storage, read> meshlets: array<MeshletDesc>;

// Camera uniform (group 1).
@group(1) @binding(0) var<uniform> camera: CameraUniform;

// Texture/sampler (group 2). Currently unused in fs_main.
@group(2) @binding(0) var t_diffuse: texture_2d<f32>;
@group(2) @binding(1) var s_diffuse: sampler;

// Dynamic uniform used to offset meshlet IDs for chunked draws.
struct MeshletParams {
    base_meshlet: u32,
    meshlet_count: u32,
    _pad0: u32,
    _pad1: u32,
}

// Meshlet params (group 3).
@group(3) @binding(0) var<uniform> meshlet_params: MeshletParams;

const MAX_MESHLET_VERTS: u32 = 64u;
const MAX_MESHLET_PRIMS: u32 = 124u;

// Vertex output to the fragment shader.
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) color: vec3<f32>,
}

// Each primitive outputs triangle indices into the meshlet-local vertex list.
struct PrimitiveOutput {
    @builtin(triangle_indices) indices: vec3<u32>,
}

// Mesh shader output arrays are per-workgroup.
// We allocate fixed-size arrays and fill only [0..vertex_count/primitive_count).
struct MeshOutput {
    @builtin(vertices) vertices: array<VertexOutput, MAX_MESHLET_VERTS>,
    @builtin(primitives) primitives: array<PrimitiveOutput, MAX_MESHLET_PRIMS>,
    @builtin(vertex_count) vertex_count: u32,
    @builtin(primitive_count) primitive_count: u32,
}

var<workgroup> mesh_out: MeshOutput;

// One mesh workgroup per meshlet.
@mesh(mesh_out)
@workgroup_size(1)
fn ms_main(@builtin(workgroup_id) wg: vec3<u32>) {
    let meshlet_id = meshlet_params.base_meshlet + wg.x;
    // Guard against out-of-range meshlet IDs in the final chunk.
    if (meshlet_id >= meshlet_params.meshlet_count) {
        mesh_out.vertex_count = 0u;
        mesh_out.primitive_count = 0u;
        return;
    }
    let meshlet = meshlets[meshlet_id];

    // Tell the rasterizer how many vertices/primitives this workgroup emits.
    mesh_out.vertex_count = meshlet.vertex_count;
    mesh_out.primitive_count = meshlet.index_count;

    // Emit transformed vertices for this meshlet.
    var i = 0u;
    loop {
        if (i >= meshlet.vertex_count) {
            break;
        }
        let global_index = meshlet_vertices[meshlet.vertex_offset + i];
        let v = vertices[global_index];
        mesh_out.vertices[i].clip_position = camera.view_proj * v.position;
        mesh_out.vertices[i].tex_coords = v.tex_coords;
        // Simple hash-based color per meshlet.
        let h = meshlet_id * 1664525u + 1013904223u;
        let r = f32((h >> 0u) & 255u) / 255.0;
        let g = f32((h >> 8u) & 255u) / 255.0;
        let b = f32((h >> 16u) & 255u) / 255.0;
        mesh_out.vertices[i].color = vec3<f32>(r, g, b);
        i = i + 1u;
    }

    // Emit per-triangle indices (local to this meshlet).
    var p = 0u;
    loop {
        if (p >= meshlet.index_count) {
            break;
        }
        let base = meshlet.index_offset + p * 3u;
        mesh_out.primitives[p].indices = vec3<u32>(
            meshlet_indices[base],
            meshlet_indices[base + 1u],
            meshlet_indices[base + 2u],
        );
        p = p + 1u;
    }
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Visualize meshlet IDs via color (ignore texture for now).
    return vec4<f32>(in.color, 1.0);
}
