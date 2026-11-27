struct CameraUniform {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> box: mat4x4<f32>;

struct QuadOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

@vertex
fn quad_vs(
    @location(0) vertex_pos: vec2<f32>,
    @location(1) instance_pos: vec3<f32>,
) -> QuadOutput {
    var out: QuadOutput;
    
    out.position = instance_pos;
    out.uv = vertex_pos;

    out.clip_position = camera.view * vec4<f32>(instance_pos, 1.0);
    out.clip_position = camera.proj * (out.clip_position + 0.1 * vec4<f32>(vertex_pos, 0.0, 0.0));

    return out;
}

@fragment
fn quad_fs(in: QuadOutput) -> @location(0) vec4<f32> {
    if length(in.uv) > 1.0 {
        discard;
    }
    return vec4<f32>(in.uv * 0.5 + 0.5, 0.0, 1.0);
}

@vertex
fn box_vs(@location(0) vertex_pos: vec3<f32>) -> @builtin(position) vec4<f32> {
    return camera.proj * camera.view * box * vec4<f32>(vertex_pos, 1.0);
}

@fragment
fn box_fs(@builtin(position) clip_position: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
