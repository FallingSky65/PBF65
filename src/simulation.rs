use cgmath::{InnerSpace, Vector3};
use rand::RngExt;
use wgpu::util::DeviceExt;

use crate::boundbox::BoundBox;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SimulationParameters {
    pub n: u32,
    pub dt: f32,
}

#[allow(unused)]
pub struct Simulation {
    pub params: SimulationParameters,
    params_buffer: wgpu::Buffer,
    params_bind_group: wgpu::BindGroup,

    pub boundbox: BoundBox,

    // structure of arrays (SoA) architecture
    pub pos: wgpu::Buffer, // N x vec3
    vel: wgpu::Buffer, // N x vec3
    
    test_bind_group: wgpu::BindGroup,
    test_pipeline: wgpu::ComputePipeline,
}

impl Simulation {
    pub fn new(params: SimulationParameters, boxradius: f32, device: &wgpu::Device) -> Self {
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        let params_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("params bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("params bind group"),
            layout: &params_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let boundbox = BoundBox::new(boxradius, device);

        let mut rng = rand::rng();
        let distr_uniform = rand_distr::Uniform::new(-1.0, 1.0).unwrap();
        let distr_normal = rand_distr::Normal::new(0.0, 1.0).unwrap();

        let positions: Vec<[f32; 3]> = (0..params.n)
            .map(|_| [
                boxradius * rng.sample(distr_uniform),
                boxradius * rng.sample(distr_uniform),
                boxradius * rng.sample(distr_uniform),
            ])
            .collect();

        let pos = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pos buffer"),
            contents: bytemuck::cast_slice(&positions),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
        });
        
        let velocities: Vec<[f32; 3]> = (0..params.n)
            .map(|_|
                Vector3::new(
                    rng.sample(distr_normal),
                    rng.sample(distr_normal),
                    rng.sample(distr_normal),
                ).normalize_to(0.5).into()
            )
            .collect();

        let vel = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vel buffer"),
            contents: bytemuck::cast_slice(&velocities),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("compute.wgsl"));

        let test_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("test bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let test_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("test bind group"),
            layout: &test_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pos.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: vel.as_entire_binding(),
                },
            ],
        });

        let test_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("test pipeline layout"),
            bind_group_layouts: &[
                Some(&params_bind_group_layout),
                Some(&boundbox.bind_group_layout),
                Some(&test_bind_group_layout),
            ],
            immediate_size: 0,
        });

        let test_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("test compute pipeline"),
            layout: Some(&test_layout),
            module: &shader,
            entry_point: None,
            compilation_options: Default::default(),
            cache: Default::default(),
        });

        Self { params, params_buffer, params_bind_group, boundbox, pos, vel, test_bind_group, test_pipeline }
    }

    pub fn update_params(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[self.params]));
    }

    pub fn run_test(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.test_pipeline);
        pass.set_bind_group(0, &self.params_bind_group, &[]);
        pass.set_bind_group(1, &self.boundbox.bind_group, &[]);
        pass.set_bind_group(2, &self.test_bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
}
