struct Params {
    n: u32,         // number of particles
    m: u32,         // maximum number of neighbors, assume 64?
    num_cells: u32, // number of cells
    dt: f32,        // time step
    h: f32,         // smoothing radius
    rho0: f32,      // rest density
    cfm_epsilon: f32, // relaxation term epsilon in eqn 11
    s_corr_k: f32,  // k = 0.1 for s_corr calculation in eqn 13
    s_corr_n: f32,  // n = 4 for s_corr calculation in eqn 13
    gravity: f32,   // downwards (y-) acceleration due to gravity
    vorticity_eps: f32, // vorticity confinement strength in eqn 16
    xsph_c: f32,    // c = 0.01 XSPH viscosity coefficient in eqn 17
};
@group(0) @binding(0) var<uniform> params: Params;

@group(1) @binding(0) var<storage, read> pos: array<vec3<f32>>;
@group(1) @binding(1) var<storage, read_write> vel1: array<vec3<f32>>;
@group(1) @binding(2) var<storage, read_write> predpos1: array<vec3<f32>>;

@compute
@workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>
) {
    let gid = global_invocation_id.x;
    if (gid >= params.n) {
        return;
    }

    let f_ext = vec3<f32>(0.0, -params.gravity, 0.0);
    vel1[gid] += params.dt * f_ext;
    predpos1[gid] = pos[gid] + params.dt * vel1[gid];
}
