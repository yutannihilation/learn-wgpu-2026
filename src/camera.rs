// TODO: change this to Vec3A
pub struct Camera {
    pub(crate) eye: glam::Vec3,
    pub(crate) target: glam::Vec3,
    pub(crate) up: glam::Vec3,
    pub(crate) aspect: f32,
    pub(crate) fovy: f32,
    pub(crate) znear: f32,
    pub(crate) zfar: f32,
}

impl Camera {
    fn build_view_projection_matrix(&self) -> glam::Mat4 {
        let view = glam::Mat4::look_at_rh(self.eye, self.target, self.up);
        let proj = glam::Mat4::perspective_rh_gl(self.fovy, self.aspect, self.znear, self.zfar);

        proj * view
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    view_proj: glam::Mat4,
}

impl CameraUniform {
    pub fn new(camera: &Camera) -> Self {
        Self {
            view_proj: camera.build_view_projection_matrix(),
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix()
    }
}
