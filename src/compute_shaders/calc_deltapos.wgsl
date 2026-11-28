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
@group(1) @binding(1) var<storage, read> neighbors: array<u32>;
@group(1) @binding(2) var<storage, read> lambda: array<f32>;
@group(1) @binding(3) var<storage, read_write> deltapos: array<vec3<f32>>;

// W_poly6 and W_spiky from Muller 2003:
// https://matthias-research.github.io/pages/publications/sca03.pdf
fn W_poly6(r: vec3<f32>) -> f32 {
    let h: f32 = params.h;
    let diff: f32 = h * h - dot(r, r);
    return select(
        0.0,
        315.0/(64.0 * 3.1415926 * h*h*h*h*h*h*h*h*h) * diff*diff*diff,
        diff >= 0.0,
    );
}

fn grad_W_spiky(r: vec3<f32>) -> vec3<f32> {
    let h: f32 = params.h;
    let l: f32 = length(r);
    if (l < 1e-6 || h < l) { return vec3<f32>(0.0); }
    return -45.0/(3.1415926 * h*h*h*h*h*h) * (h - l)*(h - l) * r/l;
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
    
    let p_i = predpos2[gid];
    let lambda_i = lambda[gid];

    let W_dq: f32 = W_poly6(vec3<f32>(0.2 * params.h, 0.0, 0.0));
   
    var dp: vec3<f32> = vec3<f32>(0.0);

    for (var j: u32 = gid * params.m; j < (gid + 1u) * params.m; j += 1u) {
        let nb = neighbors[j];
        if (nb >= params.n) { break; }
        if (nb == gid) { continue; }

        let diff: vec3<f32> = p_i - predpos2[nb];

        let s_corr = -params.s_corr_k * pow( W_poly6(diff) / W_dq, params.s_corr_n);
        dp += (lambda_i + lambda[nb] + s_corr) * grad_W_spiky(diff);
    }

    deltapos[gid] = dp / params.rho0;

    // TODO: handle collisions using SDFs: Bridson et al. 2006
}
