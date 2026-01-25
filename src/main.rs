mod camera;
mod texture;

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::{Arc, mpsc};

use wgpu::util::DeviceExt as _;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowAttributes},
};

use crate::camera::{Camera, CameraController, CameraUniform};

const MAX_MESHLET_VERTS: usize = 64;
const MAX_MESHLET_PRIMS: usize = 126;

pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    window: Arc<Box<dyn Window>>,
    camera: Camera,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    depth_texture: crate::texture::Texture,
    // Mesh pipelines use the same handle type as render pipelines, but are created
    // with create_mesh_pipeline and executed via draw_mesh_tasks.
    render_pipeline: wgpu::RenderPipeline,
    // Mesh shaders read geometry and per-instance data from storage buffers.
    mesh_bind_group: wgpu::BindGroup,
    vertex_storage_buffer: wgpu::Buffer,
    meshlet_vertex_buffer: wgpu::Buffer,
    meshlet_index_buffer: wgpu::Buffer,
    meshlet_desc_buffer: wgpu::Buffer,
    meshlet_params_buffer: wgpu::Buffer,
    meshlet_params_bind_group: wgpu::BindGroup,
    diffuse_bind_group: wgpu::BindGroup,
    diffuse_texture: texture::Texture,
    meshlet_count: u32,
    max_task_workgroups_per_dimension: u32,
    max_task_workgroup_total_count: u32,
    meshlet_params_stride: u64,
    meshlet_params_draws: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct VertexStorage {
    // Storage buffers align vec3 to 16 bytes in WGSL, so use vec4 and padding
    // to keep the Rust/WGSL layouts compatible.
    position: [f32; 4],
    tex_coords: [f32; 2],
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MeshletDesc {
    vertex_offset: u32,
    vertex_count: u32,
    index_offset: u32,
    index_count: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MeshletParams {
    base_meshlet: u32,
    meshlet_count: u32,
    _pad: [u32; 2],
}

impl State {
    pub async fn new(window: Arc<Box<dyn Window>>) -> anyhow::Result<Self> {
        let size = window.surface_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await?;

        // debug
        let info = adapter.get_info();
        println!(
            "adapter: {:?} {:?} {:?}",
            info.name, info.backend, info.device_type
        );
        println!("features: {:?}", adapter.features());

        // Mesh shaders are experimental in wgpu, so request the feature and
        // mesh-shader-specific limits up front.
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::EXPERIMENTAL_MESH_SHADER,
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                required_limits: wgpu::Limits::defaults()
                    .using_recommended_minimum_mesh_shader_values(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await?;
        let device_limits = device.limits();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|format| format.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let present_mode = surface_caps
            .present_modes
            .iter()
            .copied()
            .find(|mode| *mode == wgpu::PresentMode::Fifo)
            .unwrap_or(surface_caps.present_modes[0]);
        let alpha_mode = surface_caps
            .alpha_modes
            .iter()
            .copied()
            .find(|mode| *mode == wgpu::CompositeAlphaMode::Auto)
            .unwrap_or(surface_caps.alpha_modes[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        println!(
            "Surface config: format={:?}, present_mode={:?}, alpha_mode={:?}, size={}x{}",
            config.format, config.present_mode, config.alpha_mode, config.width, config.height
        );
        let mut is_surface_configured = false;
        if config.width > 0 && config.height > 0 {
            surface.configure(&device, &config);
            is_surface_configured = true;
        }

        let diffuse_bytes = include_bytes!("happy-tree.png");
        let diffuse_texture =
            texture::Texture::from_bytes(&device, &queue, diffuse_bytes, "happy-tree.png").unwrap();

        // Texture/sampler group for the fragment shader (same as traditional pipeline).
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let diffuse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                },
            ],
        });

        let camera = Camera {
            eye: glam::Vec3::new(0.0, 3.0, 2.0),
            target: glam::Vec3::new(0.0, 0.1, 0.0),
            up: glam::Vec3::Y,
            aspect: config.width as f32 / config.height as f32,
            fovy: 60.0_f32.to_radians(),
            znear: 0.1,
            zfar: 100.0,
        };

        let camera_controller = CameraController::new(0.03);

        let camera_uniform = CameraUniform::new(&camera);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // The camera uniform is consumed by the mesh shader, not a vertex shader.
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Camera Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::MESH,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let depth_texture =
            texture::Texture::create_depth_texture(&device, &config, "Depth texture");

        // WGSL contains mesh + fragment entry points.
        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        // Geometry + meshlet storage buffers for the mesh shader.
        let mesh_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Mesh Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::MESH,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::MESH,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::MESH,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::MESH,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let meshlet_params_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Meshlet Params Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::MESH,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        // Pipeline layout order must match @group indices in WGSL:
        // group(0)=mesh buffers, group(1)=camera, group(2)=texture/sampler.
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &mesh_bind_group_layout,
                    &camera_bind_group_layout,
                    &texture_bind_group_layout,
                    &meshlet_params_bind_group_layout,
                ],
                immediate_size: 0,
            });

        // Mesh pipeline: task + mesh + fragment (no vertex stage).
        let render_pipeline = device.create_mesh_pipeline(&wgpu::MeshPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            task: None,
            mesh: wgpu::MeshState {
                module: &shader,
                entry_point: Some("ms_main"),
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Load the bunny mesh and build meshlets for the mesh shader.
        let (models, _materials) = tobj::load_obj(
            "src/stanford-bunny.obj",
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

        let indices = &mesh.indices;
        let mut meshlet_descs: Vec<MeshletDesc> = Vec::new();
        let mut meshlet_vertices: Vec<u32> = Vec::new();
        let mut meshlet_indices: Vec<u32> = Vec::new();

        let mut current_map: HashMap<u32, u32> = HashMap::new();
        let mut current_vertex_offset = 0u32;
        let mut current_index_offset = 0u32;
        let mut current_vertex_count = 0u32;
        let mut current_prim_count = 0u32;

        for tri in indices.chunks(3) {
            if tri.len() < 3 {
                break;
            }
            let tri_indices = [tri[0], tri[1], tri[2]];
            let mut new_vertices = 0u32;
            for &idx in &tri_indices {
                if !current_map.contains_key(&idx) {
                    new_vertices += 1;
                }
            }
            if current_vertex_count + new_vertices > MAX_MESHLET_VERTS as u32
                || current_prim_count + 1 > MAX_MESHLET_PRIMS as u32
            {
                if current_prim_count > 0 {
                    meshlet_descs.push(MeshletDesc {
                        vertex_offset: current_vertex_offset,
                        vertex_count: current_vertex_count,
                        index_offset: current_index_offset,
                        index_count: current_prim_count,
                    });
                    current_vertex_offset = meshlet_vertices.len() as u32;
                    current_index_offset = meshlet_indices.len() as u32;
                    current_vertex_count = 0;
                    current_prim_count = 0;
                    current_map.clear();
                }
            }

            for &idx in &tri_indices {
                let local = if let Some(&local) = current_map.get(&idx) {
                    local
                } else {
                    let local = current_vertex_count;
                    current_map.insert(idx, local);
                    meshlet_vertices.push(idx);
                    current_vertex_count += 1;
                    local
                };
                meshlet_indices.push(local);
            }
            current_prim_count += 1;
        }

        if current_prim_count > 0 {
            meshlet_descs.push(MeshletDesc {
                vertex_offset: current_vertex_offset,
                vertex_count: current_vertex_count,
                index_offset: current_index_offset,
                index_count: current_prim_count,
            });
        }

        let vertex_storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Storage Buffer"),
            contents: bytemuck::cast_slice(&vertex_storage),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let meshlet_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Meshlet Vertex Buffer"),
            contents: bytemuck::cast_slice(&meshlet_vertices),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let meshlet_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Meshlet Index Buffer"),
            contents: bytemuck::cast_slice(&meshlet_indices),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let meshlet_desc_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Meshlet Desc Buffer"),
            contents: bytemuck::cast_slice(&meshlet_descs),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let meshlet_count = meshlet_descs.len() as u32;
        let max_x = device_limits
            .max_task_mesh_workgroups_per_dimension
            .min(device_limits.max_task_mesh_workgroup_total_count);
        let mut params = Vec::new();
        let mut base = 0u32;
        while base < meshlet_count {
            let remaining = meshlet_count - base;
            let x = remaining.min(max_x);
            params.push(MeshletParams {
                base_meshlet: base,
                meshlet_count,
                _pad: [0, 0],
            });
            base += x;
        }
        let stride = {
            let align = device_limits.min_uniform_buffer_offset_alignment as u64;
            let size = std::mem::size_of::<MeshletParams>() as u64;
            ((size + align - 1) / align) * align
        };
        let mut params_bytes = vec![0u8; (stride as usize) * params.len()];
        for (i, p) in params.iter().enumerate() {
            let bytes = bytemuck::bytes_of(p);
            let offset = i * stride as usize;
            params_bytes[offset..offset + bytes.len()].copy_from_slice(bytes);
        }
        let meshlet_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Meshlet Params Buffer"),
            contents: &params_bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Bind group for mesh shader inputs (all storage buffers).
        let mesh_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mesh Bind Group"),
            layout: &mesh_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vertex_storage_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: meshlet_vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: meshlet_index_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: meshlet_desc_buffer.as_entire_binding(),
                },
            ],
        });
        let meshlet_params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Meshlet Params Bind Group"),
            layout: &meshlet_params_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &meshlet_params_buffer,
                    offset: 0,
                    size: NonZeroU64::new(stride),
                }),
            }],
        });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured,
            window,
            camera,
            camera_controller,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            depth_texture,
            render_pipeline,
            mesh_bind_group,
            vertex_storage_buffer,
            meshlet_vertex_buffer,
            meshlet_index_buffer,
            meshlet_desc_buffer,
            meshlet_params_buffer,
            meshlet_params_bind_group,
            diffuse_bind_group,
            diffuse_texture,
            meshlet_count,
            max_task_workgroups_per_dimension: device_limits.max_task_mesh_workgroups_per_dimension,
            max_task_workgroup_total_count: device_limits.max_task_mesh_workgroup_total_count,
            meshlet_params_stride: stride,
            meshlet_params_draws: params.len() as u32,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;

            self.depth_texture =
                texture::Texture::create_depth_texture(&self.device, &self.config, "Depth texture");
        }
    }

    fn update(&mut self) {
        self.camera_controller.update_camera(&mut self.camera);
        self.camera_uniform.update_view_proj(&self.camera);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
    }

    pub fn render(&self) -> Result<(), wgpu::SurfaceError> {
        self.window.request_redraw();

        if !self.is_surface_configured {
            return Ok(());
        }

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.mesh_bind_group, &[]);
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(2, &self.diffuse_bind_group, &[]);
            let max_x = self
                .max_task_workgroups_per_dimension
                .min(self.max_task_workgroup_total_count);
            let mut base = 0u32;
            let mut draw_index = 0u32;
            while base < self.meshlet_count && draw_index < self.meshlet_params_draws {
                let remaining = self.meshlet_count - base;
                let x = remaining.min(max_x);
                let offset = self.meshlet_params_stride * draw_index as u64;
                render_pass.set_bind_group(3, &self.meshlet_params_bind_group, &[offset as u32]);
                render_pass.draw_mesh_tasks(x, 1, 1);
                base += x;
                draw_index += 1;
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn handle_key(
        &mut self,
        event_loop: &(dyn ActiveEventLoop + 'static),
        code: KeyCode,
        is_pressed: bool,
    ) {
        match (code, is_pressed) {
            (KeyCode::Escape, true) => event_loop.exit(),
            _ => {
                self.camera_controller.handle_key(code, is_pressed);
            }
        }
    }
}

pub struct App {
    state: Option<State>,
    receiver: mpsc::Receiver<State>,
}

impl App {
    pub fn new(receiver: mpsc::Receiver<State>) -> Self {
        Self {
            state: None,
            receiver,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, _event_loop: &dyn ActiveEventLoop) {
        // On desktop platforms, `resumed` is not emitted in winit 0.31.
        // Keep this empty to avoid relying on a callback that won't fire.
    }

    fn can_create_surfaces(&mut self, event_loop: &dyn ActiveEventLoop) {
        // winit 0.31 emits this on desktop and expects window/surface creation here.
        if self.state.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(WindowAttributes::default())
                .unwrap(),
        );

        self.state = Some(pollster::block_on(State::new(window)).unwrap());
    }

    fn proxy_wake_up(&mut self, event_loop: &dyn ActiveEventLoop) {
        self.state = self.receiver.try_iter().last();
    }

    fn window_event(
        &mut self,
        event_loop: &dyn ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let state = match &mut self.state {
            Some(canvas) => canvas,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::SurfaceResized(size) => state.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    // Reconfigure the surface if it's lost or outdated
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let size = state.window.surface_size();
                        state.resize(size.width, size.height);
                    }
                    Err(e) => {
                        // log::error!("Unable to render {}", e);
                    }
                };
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => state.handle_key(event_loop, code, key_state.is_pressed()),
            _ => {}
        }
    }
}

pub fn run() -> anyhow::Result<()> {
    let event_loop = EventLoop::new()?;

    let (sender, receiver) = mpsc::channel();

    let app = App::new(receiver);
    event_loop.run_app(app)?;

    Ok(())
}

fn main() {
    run().unwrap()
}
