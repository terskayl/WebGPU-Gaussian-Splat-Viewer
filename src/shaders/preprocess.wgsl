const SH_C0: f32 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<f32,7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

override workgroupSize: u32;
override sortKeyPerThread: u32;

struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,  // instance_count in DrawIndirect
    //data below is for info inside radix sort 
    padded_size: u32, 
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct RenderSettings {
    gaussian_scaling: u32,
    sh_deg: u32,
}

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct Splat {
    position: vec3<f32>,
    uv: vec2<f32>,
    radius: f32,
    covariance: vec3<f32>,
    color: vec3<f32>,
};

//TODO: bind your data here
@group(0) @binding(0)
var<storage, read> gaussians : array<Gaussian>;
@group(0) @binding(1)
var<storage, read> sh_coeff : array<u32>;

@group(1) @binding(0)
var<storage, read_write> out_splat : array<Splat>;

@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

@group(3) @binding(0)
var<uniform> camera: CameraUniforms;
@group(3) @binding(1)
var<uniform> render_settings: RenderSettings;

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    //TODO: access your binded sh_coeff, see load.ts for how it is stored
    // max sh deg is 3 -> 16 coefficients per splat, = 48 floats = 24 u32s
    
    // rg | ba
    if ((c_idx & 1) == 0) {
        let rg = unpack2x16float(sh_coeff[splat_idx * 24u + c_idx / 2  * 3u]);
        let bAlpha = unpack2x16float(sh_coeff[splat_idx * 24u + c_idx / 2 * 3u + 1u]);
        return vec3<f32>(rg.x, rg.y, bAlpha.x);
    } else { // r | gb | a
        let r = unpack2x16float(sh_coeff[splat_idx * 24u + c_idx / 2 * 3u + 1u]);
        let gb = unpack2x16float(sh_coeff[splat_idx * 24u + c_idx / 2 * 3u + 2u]);
        return vec3<f32>(r.x, gb.y, gb.x);
    }
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(v_idx, 0u);

    if sh_deg > 0u {

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);

        if sh_deg > 1u {

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return  max(vec3<f32>(0.), result);
}

fn eigenvalues(cov: mat2x2<f32>) -> vec2<f32> {
    let trace = cov[0][0] + cov[1][1];
    let det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
    let temp = sqrt(max(0.1, trace * trace / 4.0 - det));
    let lambda1 = trace / 2.0 + temp;
    let lambda2 = trace / 2.0 - temp;
    return vec2<f32>(lambda1, lambda2);
}


@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    //TODO: set up pipeline as described in instruction

    let vertex = gaussians[idx];
    let a = unpack2x16float(vertex.pos_opacity[0]);
    let b = unpack2x16float(vertex.pos_opacity[1]);
    let world_pos = vec3<f32>(a.x, a.y, b.x);
    let view_pos = (camera.view * vec4<f32>(world_pos, 1.0)).xyz;
    var clip_pos = camera.proj * vec4<f32>(view_pos, 1.0);
    clip_pos /= clip_pos.w;
    out_splat[idx].position = clip_pos.xyz;

    let sh_deg = 3u; // TODO u32(render_settings.sh_deg);
    let eye = camera.view_inv[3].xyz;
    let color = computeColorFromSH(normalize(eye - world_pos), idx, sh_deg);
    out_splat[idx].color = color;

    // Find Covariance and Radius
    let gaussian = gaussians[idx];
    var scale = mat3x3<f32>( 1.0, 0.0, 0.0,
                             0.0, 1.0, 0.0,
                             0.0, 0.0, 1.0);

    // TODO idk while render_settings isn't working.
                             // exp as stored in log space
    scale[0][0] = 1.0 * exp(unpack2x16float(gaussian.scale[0]).x);
    scale[1][1] = 1.0 * exp(unpack2x16float(gaussian.scale[0]).y);
    scale[2][2] = 1.0 * exp(unpack2x16float(gaussian.scale[1]).x);

    var quat = gaussian.rot;
    let qx = unpack2x16float(quat[0]).x;
    let qy = unpack2x16float(quat[0]).y;
    let qz = unpack2x16float(quat[1]).x;
    let qw = unpack2x16float(quat[1]).y;

    let rotation = mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)),
        vec3<f32>(2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)),
        vec3<f32>(2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy))
    );


    var cov3D = transpose(rotation * scale) * (scale * rotation);

    // help stability
    //cov3D[0][0] += 0.3f;
	//cov3D[1][1] += 0.3f;

    let proj_matrix = mat2x3<f32>(
        vec3<f32>(camera.focal.x / view_pos.z, 0.0, -camera.focal.x * view_pos.x / (view_pos.z * view_pos.z)),
        vec3<f32>(0.0, camera.focal.y / view_pos.z, -camera.focal.y * view_pos.y / (view_pos.z * view_pos.z) )
    );

    // let W = transpose(mat3x3f(
    //     camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz
    // ));

    let cov2D = transpose(proj_matrix) * cov3D * proj_matrix;

    out_splat[idx].covariance = vec3<f32>(cov2D[0][0], cov2D[0][1], cov2D[1][1]);

    let eigenvals = eigenvalues(cov2D);
    var radius = ceil(3.0 * sqrt(max(eigenvals.x, eigenvals.y)));
    // Nan Check
    if (radius != radius) {
        radius = 0.0;
    }


    // Store the radius in the output splat
    out_splat[idx].radius = radius;

    if (clip_pos.x > -1.2 || clip_pos.x < 1.2 || clip_pos.y > -1.2 || clip_pos.y < 1.2) {
        let curr_idx = atomicLoad(&sort_infos.keys_size);
        atomicAdd(&sort_infos.keys_size, 1u);
        let depth = view_pos.z;
        sort_depths[idx] = u32(100000000.0 - depth * 1000000.0); // make float depth into sortable uint
        sort_indices[idx] = idx;

        // increment DispatchIndirect.dispatch_x each time you reach limit for one dispatch of keys
        let keys_per_dispatch = workgroupSize * sortKeyPerThread; 
        if (curr_idx % keys_per_dispatch) == (keys_per_dispatch - 1u) {
            atomicAdd(&sort_dispatch.dispatch_x, 1u);
        }

    }



}