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

@group(1) @binding(0) var<storage, read> predpos2: array<vec3<f32>>;
@group(1) @binding(1) var<storage, read> cell_ranges: array<vec2<u32>>;
@group(1) @binding(2) var<storage, read_write> neighbors: array<u32>;

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

const adjacents: array<vec3<i32>, 27> = array(
    vec3<i32>(-1, -1, -1), vec3<i32>(-1, -1, 0), vec3<i32>(-1, -1, 1),
    vec3<i32>(-1, 0, -1), vec3<i32>(-1, 0, 0), vec3<i32>(-1, 0, 1),
    vec3<i32>(-1, 1, -1), vec3<i32>(-1, 1, 0), vec3<i32>(-1, 1, 1),
    vec3<i32>(0, -1, -1), vec3<i32>(0, -1, 0), vec3<i32>(0, -1, 1),
    vec3<i32>(0, 0, -1), vec3<i32>(0, 0, 0), vec3<i32>(0, 0, 1),
    vec3<i32>(0, 1, -1), vec3<i32>(0, 1, 0), vec3<i32>(0, 1, 1),
    vec3<i32>(1, -1, -1), vec3<i32>(1, -1, 0), vec3<i32>(1, -1, 1),
    vec3<i32>(1, 0, -1), vec3<i32>(1, 0, 0), vec3<i32>(1, 0, 1),
    vec3<i32>(1, 1, -1), vec3<i32>(1, 1, 0), vec3<i32>(1, 1, 1),
);

@compute
@workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>
) {
    let gid = global_invocation_id.x;
    if (gid >= params.n) {
        return;
    }

    let cell_pos = vec3<i32>(floor(predpos2[gid]/params.h));
    let p1: vec3<f32> = predpos2[gid];

    let offset: u32 = gid * params.m;
    
    var i: u32 = 0u;

    for (var j: u32 = 0u; j < 27u; j += 1u) {
        let cell: vec3<i32> = cell_pos + adjacents[j];
        let range: vec2<u32> = cell_ranges[hash3D(cell) % params.num_cells];
        for (var k: u32 = range.x; k < range.y; k += 1u) {
            if (length(predpos2[k] - p1) <= params.h) {
                neighbors[offset + i] = k;
                i += 1u;
                if (i == params.m) {
                    return;
                }
            }
        }
    }

    for (; i < params.m; i += 1u) {
        neighbors[offset + i] = params.n;
    }
}
