use wgpu::util::DeviceExt;

#[allow(unused)]
pub struct BoundBox {
    boundbox: cgmath::Matrix4<f32>,
    buffer: wgpu::Buffer,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    pub vertex_buffer: wgpu::Buffer,
}

impl BoundBox {
    pub fn new(boxradius: f32, device: &wgpu::Device) -> Self {
        let boundbox = cgmath::Matrix4::from_scale(boxradius);
        let boundbox_raw: [[f32; 4]; 4] = boundbox.into();
        
        let buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("BoundBox Buffer"),
                contents: bytemuck::cast_slice(&[boundbox_raw]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );
        
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            ],
            label: Some("boundbox_bind_group_layout"),
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }
            ],
            label: Some("boundbox_bind_group"),
        });
   
        const BOX_VERTICES: [[[f32; 3]; 2]; 12] = [
            [[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0]],
            [[-1.0, -1.0, -1.0], [-1.0, 1.0, -1.0]],
            [[-1.0, -1.0, -1.0], [-1.0, -1.0, 1.0]],
            [[1.0, -1.0, -1.0], [1.0, 1.0, -1.0]],
            [[1.0, 1.0, -1.0], [-1.0, 1.0, -1.0]],
            [[-1.0, 1.0, -1.0], [-1.0, 1.0, 1.0]],
            [[-1.0, 1.0, 1.0], [-1.0, -1.0, 1.0]],
            [[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0]],
            [[1.0, -1.0, 1.0], [1.0, -1.0, -1.0]],
            [[-1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            [[1.0, -1.0, 1.0], [1.0, 1.0, 1.0]],
            [[1.0, 1.0, -1.0], [1.0, 1.0, 1.0]],
        ];
        
        let vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Box Vertex Buffer"),
                contents: bytemuck::cast_slice(&BOX_VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        Self { boundbox, buffer, bind_group_layout, bind_group, vertex_buffer }
    }
}
