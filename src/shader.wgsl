// Mesh shaders are an experimental WGSL extension in wgpu.
// They replace the vertex stage with task + mesh stages.
enable wgpu_mesh_shader;

struct CameraUniform {
    view_proj: mat4x4<f32>,
}

// Geometry is read from storage buffers (no fixed-function vertex input).
struct VertexInput {
    position: vec4<f32>,
    tex_coords: vec2<f32>,
    _pad: vec2<f32>,
}

// Per-instance data for meshlet generation.
struct InstanceInput {
    model_matrix: mat4x4<f32>,
}

// Task payload is written by the task shader and read by the mesh shader.
struct TaskPayload {
    instance_index: u32,
}

// Mesh input buffers (group 0).
@group(0) @binding(0) var<storage, read> vertices: array<VertexInput>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> instances: array<InstanceInput>;
@group(0) @binding(3) var<storage, read_write> dispatch_args: DispatchIndirect;
// Compute writes visibility flags here.
@group(0) @binding(4) var<storage, read_write> meshlet_visible_rw: array<u32>;
// Mesh shader reads visibility flags here.
@group(0) @binding(5) var<storage, read> meshlet_visible: array<u32>;

// Camera uniform (group 1).
@group(1) @binding(0) var<uniform> camera: CameraUniform;

// Texture/sampler for fragment shading (group 2).
@group(2) @binding(0) var t_diffuse: texture_2d<f32>;
@group(2) @binding(1) var s_diffuse: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

struct PrimitiveOutput {
    @builtin(triangle_indices) indices: vec3<u32>,
    @builtin(cull_primitive) cull: bool,
}

// Mesh shader outputs are per-workgroup. We must provide:
// - output vertex array
// - output primitive array (triangle indices)
// - counts for both
struct MeshOutput {
    @builtin(vertices) vertices: array<VertexOutput, 5>,
    @builtin(primitives) primitives: array<PrimitiveOutput, 3>,
    @builtin(vertex_count) vertex_count: u32,
    @builtin(primitive_count) primitive_count: u32,
}

// Layout for draw_mesh_tasks_indirect dispatch arguments.
struct DispatchIndirect {
    x: u32,
    y: u32,
    z: u32,
}

var<task_payload> payload: TaskPayload;
var<workgroup> mesh_out: MeshOutput;

// Compute stage that builds the indirect dispatch arguments on the GPU.
// This is the "GPU-driven" step: the CPU never decides the mesh task count.
@compute
@workgroup_size(1)
fn cs_main() {
    let instance_count = arrayLength(&instances);
    dispatch_args.x = instance_count;
    dispatch_args.y = 1u;
    dispatch_args.z = 1u;

    // Example visibility: keep even instances, cull odd ones.
    // Replace this with frustum/occlusion logic as needed.
    var i = 0u;
    loop {
        if (i >= instance_count) {
            break;
        }
        meshlet_visible_rw[i] = select(0u, 1u, (i & 1u) == 0u);
        i = i + 1u;
    }
}

@task
@payload(payload)
@workgroup_size(1)
fn ts_main(@builtin(workgroup_id) task_id: vec3<u32>) -> @builtin(mesh_task_size) vec3<u32> {
    // Each task workgroup corresponds to one instance.
    payload.instance_index = task_id.x;
    // Emit exactly one mesh workgroup per task workgroup.
    return vec3<u32>(1u, 1u, 1u);
}

@mesh(mesh_out)
@payload(payload)
@workgroup_size(1)
fn ms_main() {
    // The mesh shader builds the final vertices and triangles for this instance.
    let instance_index = payload.instance_index;
    let inst = instances[instance_index];

    mesh_out.vertex_count = 5u;
    mesh_out.primitive_count = 3u;

    // Generate transformed vertices.
    var i = 0u;
    loop {
        if (i >= 5u) {
            break;
        }
        let v = vertices[i];
        let world_pos = inst.model_matrix * vec4<f32>(v.position.xyz, 1.0);
        mesh_out.vertices[i].clip_position = camera.view_proj * world_pos;
        mesh_out.vertices[i].tex_coords = v.tex_coords;
        i = i + 1u;
    }

    // Emit triangle indices (same topology as the old index buffer).
    mesh_out.primitives[0].indices = vec3<u32>(indices[0], indices[1], indices[2]);
    mesh_out.primitives[1].indices = vec3<u32>(indices[3], indices[4], indices[5]);
    mesh_out.primitives[2].indices = vec3<u32>(indices[6], indices[7], indices[8]);

    // Read visibility computed by the compute pass.
    let visible = meshlet_visible[instance_index] != 0u;
    mesh_out.primitives[0].cull = !visible;
    mesh_out.primitives[1].cull = !visible;
    mesh_out.primitives[2].cull = !visible;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Fragment stage is unchanged from a traditional pipeline.
    return textureSample(t_diffuse, s_diffuse, in.tex_coords);
}
