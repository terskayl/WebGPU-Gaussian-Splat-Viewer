struct VertexInput {
    @location(0) position: vec3<f32>,

    @location(1) instance_pos : vec3<f32>,
    @location(2) instance_uvRadAlpha  : vec4<f32>,
    @location(3) instance_cov : vec4<f32>,
    @location(4) instance_col : vec4<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
    @location (0) uvRadAlpha: vec4<f32>,
    @location (1) covariance: vec3<f32>,
    @location (2) color: vec3<f32>,
};

struct Splat {
    position: vec3<f32>,
    uvRadAlpha: vec4<f32>,
    covariance: vec3<f32>,
    color: vec3<f32>,
};

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(1) @binding(0)
var<storage, read> splats : array<Splat>;
@group(2) @binding(0)
var<storage, read> sort_indices : array<u32>;

@vertex
fn vs_main(in: VertexInput, @builtin(vertex_index) vertexId: u32, @builtin(instance_index) instance_id: u32) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass 
    var out: VertexOutput;

    let splatIdx = sort_indices[instance_id];
    let splat = splats[splatIdx];

    let vertices = array<vec2<f32>, 6>(
        vec2<f32>(-0.01, -0.01),
        vec2<f32>(0.01, -0.01),
        vec2<f32>(-0.01, 0.01),
        vec2<f32>(-0.01, 0.01),
        vec2<f32>(0.01, -0.01),
        vec2<f32>(0.01, 0.01),
    );


    // let scale = in.instance_uvRadAlpha.z / camera.viewport;
    // let pos = scale * 100.0 * vertices[vertexId] + in.instance_pos.xy;
    // out.position = vec4<f32>(pos, 0.0, 1.0); 
    // out.color = in.instance_col.xyz;

    let scale = splat.uvRadAlpha.z / camera.viewport;
    let pos = scale * 100.0 * vertices[vertexId] + splat.position.xy;
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.color = splat.color.xyz;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.);
}