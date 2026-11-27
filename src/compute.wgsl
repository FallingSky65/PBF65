struct Params {
    n: u32,
    dt: f32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(1) @binding(0) var<uniform> box: mat4x4<f32>;

@group(2) @binding(0) var<storage, read_write> pos: array<vec3<f32>>;
@group(2) @binding(1) var<storage, read_write> vel: array<vec3<f32>>;

@compute
@workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>
) {
    let gid = global_invocation_id.x;
    if (gid >= params.n) {
        return;
    }

    var p = pos[gid] + vel[gid] * params.dt;
    if (p.x >= 1.0 || p.x <= -1.0) {
        vel[gid].x *= -1.0;
    }
    if (p.y >= 1.0 || p.y <= -1.0) {
        vel[gid].y *= -1.0;
    }
    if (p.z >= 1.0 || p.z <= -1.0) {
        vel[gid].z *= -1.0;
    }

    pos[gid] += vel[gid] * params.dt;
}
