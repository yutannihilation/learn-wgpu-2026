use std::sync::{Arc, mpsc};

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowAttributes},
};

pub struct State {
    window: Arc<Box<dyn Window>>,
}

impl State {
    pub async fn new(window: Arc<Box<dyn Window>>) -> anyhow::Result<Self> {
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
                state.render();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => match (code, key_state.is_pressed()) {
                (KeyCode::Escape, true) => event_loop.exit(),
                _ => {}
            },
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
