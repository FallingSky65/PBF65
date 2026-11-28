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

@group(1) @binding(0) var<storage, read> predpos1: array<vec3<f32>>;
@group(1) @binding(1) var<storage, read_write> cellhash1: array<u32>;
@group(1) @binding(2) var<storage, read_write> particleid1: array<u32>;

// Thomas Wang hash from Bob Jenkin's website:
// https://www.burtleburtle.net/bob/hash/integer.html
fn hash(seed: u32) -> u32 {
    var a: u32 = seed;
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);
    return a;
}

fn hash3D(v: vec3<i32>) -> u32 {
    let x: u32 = bitcast<u32>(v.x);
    let y: u32 = bitcast<u32>(v.y);
    let z: u32 = bitcast<u32>(v.z);
    return hash(x ^ hash(y ^ hash(z)));
}

@compute
@workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>
) {
    let gid = global_invocation_id.x;
    if (gid >= params.n) {
        return;
    }

    cellhash1[gid] = hash3D(vec3<i32>(floor(predpos1[gid]/params.h))) % params.num_cells;
    particleid1[gid] = gid;
}
