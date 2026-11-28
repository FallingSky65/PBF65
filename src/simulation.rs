use cgmath::{InnerSpace, Vector3};
use rand::RngExt;
use wgpu::util::DeviceExt;

use crate::boundbox::BoundBox;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SimulationParameters {
    pub n: u32,         // number of particles
    pub m: u32,         // maximum number of neighbors, assume 64?
    pub num_cells: u32, // number of cells
    pub dt: f32,        // time step
    pub h: f32,         // smoothing radius
    pub rho0: f32,      // rest density
    pub cfm_epsilon: f32, // relaxation term epsilon in eqn 11
    pub s_corr_k: f32,  // k = 0.1 for s_corr calculation in eqn 13
    pub s_corr_n: f32,  // n = 4 for s_corr calculation in eqn 13
    pub gravity: f32,   // downwards (y-) acceleration due to gravity
    pub vorticity_eps: f32, // vorticity confinement strength in eqn 16
    pub xsph_c: f32,    // c = 0.01 XSPH viscosity coefficient in eqn 17
}

#[allow(unused)]
pub struct Simulation {
    pub params: SimulationParameters,
    params_buffer: wgpu::Buffer,
    params_bind_group: wgpu::BindGroup,

    pub boundbox: BoundBox,

    // structure of arrays (SoA) architecture
    pub pos: wgpu::Buffer, // N x vec3
    vel1: wgpu::Buffer, // N x vec3
    vel2: wgpu::Buffer, // N x vec3
    predpos1: wgpu::Buffer, // N x vec3
    predpos2: wgpu::Buffer, // N x vec3
    cellhash1: wgpu::Buffer, // N x u32
    cellhash2: wgpu::Buffer, // N x u32
    particleid1: wgpu::Buffer, // N x u32
    particleid2: wgpu::Buffer, // N x u32
    cell_ranges: wgpu::Buffer, // N x vec2<u32>
    neighbors: wgpu::Buffer, // N x M x u32
    lambda: wgpu::Buffer, // N x f32
    deltapos: wgpu::Buffer, // N x vec3
    omega: wgpu::Buffer, // N x vec3
    
    integrate: Pipeline,
    // pos=r, vel1=rw, predpos1=w
    calc_hashes: Pipeline,
    // predpos1=r, cellhash1=w, particleid1=w
    // radix_sort: wgpu::ComputePipeline, //TODO
    // sort (cellhash1,particleid1) by cellhash using pingpong
    find_ranges: Pipeline,
    // cellhash1=r, cell_ranges=w, predpos1=r, predpos2=w, particleid1=r
    find_neighbors: Pipeline,
    // predpos2=r, cell_ranges=r, neighbors=w

    // solver loop
    calc_lambda: Pipeline,
    // predpos2=r, neighbors=r, lambda=w
    calc_deltapos: Pipeline,
    // predpos2=r, neighbors=r, lambda=r, deltapos=w, (collision stuff)=r
    update_predpos: Pipeline,
    // predpos2=rw, deltapos=r

    velpos_update: Pipeline,
    // pos=rw, predpos2=r, vel2=w
    calc_vorticity: Pipeline,
    // pos=r, vel2=r, neighbors=r, omega=w
    vorticity_impulse: Pipeline,
    // pos=r, vel2=rw, neighbors=r, omega=r
    xsph_viscosity: Pipeline,
    // pos=r, vel2=r, neighbors=r, vel1=w
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

        let n: u64 = params.n as u64;

        let mut rng = rand::rng();
        let distr_uniform = rand_distr::Uniform::new(-1.0, 1.0).unwrap();
        let distr_normal = rand_distr::Normal::new(0.0, 1.0).unwrap();

        let positions: Vec<[f32; 3]> = (0..n)
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
        
        let vel1 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vel1 buffer"),
            size: n * std::mem::size_of::<[f32; 3]>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let vel2 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vel2 buffer"),
            size: n * std::mem::size_of::<[f32; 3]>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let predpos1 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("predpos1 buffer"),
            size: n * std::mem::size_of::<[f32; 3]>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let predpos2 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("predpos2 buffer"),
            size: n * std::mem::size_of::<[f32; 3]>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let cellhash1 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cellhash1 buffer"),
            size: n * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let cellhash2 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cellhash2 buffer"),
            size: n * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let particleid1 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particleid1 buffer"),
            size: n * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let particleid2 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particleid2 buffer"),
            size: n * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let cell_ranges = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cell_ranges buffer"),
            size: (params.num_cells as u64) * std::mem::size_of::<[u32; 2]>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let neighbors = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("neighbors buffer"),
            size: n * (params.m as u64) * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let lambda = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lambda buffer"),
            size: n * std::mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let deltapos = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("deltapos buffer"),
            size: n * std::mem::size_of::<[f32; 3]>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let omega = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("omega buffer"),
            size: n * std::mem::size_of::<[f32; 3]>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let integrate = Pipeline::new(
            device,
            wgpu::include_wgsl!("compute_shaders/integrate.wgsl"),
            &params_bind_group_layout,
            vec![pos.as_entire_binding()],
            vec![vel1.as_entire_binding(), predpos1.as_entire_binding()],
        );

        let calc_hashes = Pipeline::new(
            device,
            wgpu::include_wgsl!("compute_shaders/calc_hashes.wgsl"),
            &params_bind_group_layout,
            vec![predpos1.as_entire_binding()],
            vec![cellhash1.as_entire_binding(), particleid1.as_entire_binding()],
        );

        let find_ranges = Pipeline::new(
            device,
            wgpu::include_wgsl!("compute_shaders/find_ranges.wgsl"),
            &params_bind_group_layout,
            vec![cellhash1.as_entire_binding(), particleid1.as_entire_binding(), predpos1.as_entire_binding()],
            vec![cell_ranges.as_entire_binding(), predpos2.as_entire_binding()],
        );
        
        let find_neighbors = Pipeline::new(
            device,
            wgpu::include_wgsl!("compute_shaders/find_neighbors.wgsl"),
            &params_bind_group_layout,
            vec![predpos2.as_entire_binding(), cell_ranges.as_entire_binding()],
            vec![neighbors.as_entire_binding()],
        );
        
        let calc_lambda = Pipeline::new(
            device,
            wgpu::include_wgsl!("compute_shaders/calc_lambda.wgsl"),
            &params_bind_group_layout,
            vec![predpos2.as_entire_binding(), neighbors.as_entire_binding()],
            vec![lambda.as_entire_binding()],
        );
        
        let calc_deltapos = Pipeline::new(
            device,
            wgpu::include_wgsl!("compute_shaders/calc_deltapos.wgsl"),
            &params_bind_group_layout,
            vec![predpos2.as_entire_binding(), neighbors.as_entire_binding(), lambda.as_entire_binding()],
            vec![deltapos.as_entire_binding()],
        );
        
        let update_predpos = Pipeline::new(
            device,
            wgpu::include_wgsl!("compute_shaders/update_predpos.wgsl"),
            &params_bind_group_layout,
            vec![deltapos.as_entire_binding()],
            vec![predpos2.as_entire_binding()],
        );
        
        let velpos_update = Pipeline::new(
            device,
            wgpu::include_wgsl!("compute_shaders/velpos_update.wgsl"),
            &params_bind_group_layout,
            vec![predpos2.as_entire_binding()],
            vec![pos.as_entire_binding(), vel2.as_entire_binding()],
        );
        
        let calc_vorticity = Pipeline::new(
            device,
            wgpu::include_wgsl!("compute_shaders/calc_vorticity.wgsl"),
            &params_bind_group_layout,
            vec![pos.as_entire_binding(), vel2.as_entire_binding(), neighbors.as_entire_binding()],
            vec![omega.as_entire_binding()],
        );
        
        let vorticity_impulse = Pipeline::new(
            device,
            wgpu::include_wgsl!("compute_shaders/vorticity_impulse.wgsl"),
            &params_bind_group_layout,
            vec![pos.as_entire_binding(), neighbors.as_entire_binding(), omega.as_entire_binding()],
            vec![vel2.as_entire_binding()],
        );
        
        let xsph_viscosity = Pipeline::new(
            device,
            wgpu::include_wgsl!("compute_shaders/xsph_viscosity.wgsl"),
            &params_bind_group_layout,
            vec![pos.as_entire_binding(), vel2.as_entire_binding(), neighbors.as_entire_binding()],
            vec![vel1.as_entire_binding()],
        );

        Self { params, params_buffer, params_bind_group, boundbox, pos, vel1, vel2, predpos1, predpos2, cellhash1, cellhash2, particleid1, particleid2, cell_ranges, neighbors, lambda, deltapos, omega, integrate, calc_hashes, find_ranges, find_neighbors, calc_lambda, calc_deltapos, update_predpos, velpos_update, calc_vorticity, vorticity_impulse, xsph_viscosity }
    }

    pub fn update_params(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[self.params]));
    }

    pub fn run_test(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&Default::default());


        // TODO: reset cell ranges
    }
}

struct Pipeline {
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
}

impl Pipeline {
    fn new(
        device: &wgpu::Device,
        shader_desc: wgpu::ShaderModuleDescriptor<'_>,
        uniform_layout: &wgpu::BindGroupLayout,
        bindings_r: Vec<wgpu::BindingResource<'_>>,
        bindings_rw: Vec<wgpu::BindingResource<'_>>,
    ) -> Self {
        let shader = device.create_shader_module(shader_desc);
        
        let layout_entries_r: Vec<wgpu::BindGroupLayoutEntry> = (0..bindings_r.len())
            .map(|i| wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout_entries_rw: Vec<wgpu::BindGroupLayoutEntry> = (0..bindings_rw.len())
            .map(|i| wgpu::BindGroupLayoutEntry {
                binding: (i + layout_entries_r.len()) as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: [layout_entries_r, layout_entries_rw].concat().as_slice(),
        });

        let entries: Vec<wgpu::BindGroupEntry> = [bindings_r, bindings_rw].concat()
            .into_iter()
            .enumerate()
            .map(|(i, binding)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: binding,
            })
            .collect();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: entries.as_slice(),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[
                Some(&uniform_layout),
                Some(&bind_group_layout),
            ],
            immediate_size: 0,
        });
        
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: None,
            compilation_options: Default::default(),
            cache: Default::default(),
        });

        Self { bind_group, pipeline }
    }
}
