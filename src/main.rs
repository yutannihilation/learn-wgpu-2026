use std::sync::{Arc, mpsc};

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowAttributes},
};

pub struct State {
    window: Arc<dyn Window>,
}

impl State {
    pub async fn new(window: Arc<dyn Window>) -> anyhow::Result<Self> {
        Ok(Self { window })
    }

    pub fn resize(&self, width: u32, height: u32) {
        // TODO
    }

    pub fn render(&self) {
        self.window.request_redraw();
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
    fn resumed(&mut self, event_loop: &dyn ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(WindowAttributes::default())
                .unwrap(),
        );
    }

    fn can_create_surfaces(&mut self, event_loop: &dyn ActiveEventLoop) {
        todo!()
    }

    fn window_event(
        &mut self,
        event_loop: &dyn ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        todo!()
    }

    fn proxy_wake_up(&mut self, event_loop: &dyn ActiveEventLoop) {
        todo!()
    }
}

fn main() {
    println!("Hello, world!");
}
