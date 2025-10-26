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

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass 
    var out: VertexOutput;

    let scale = min(0.01, in.instance_uvRadAlpha.z);
    let pos = scale * 100.0 *  in.position + vec3<f32>(in.instance_pos.xy, 0.0);
    out.position = vec4<f32>(pos, 1.0); 
    out.color = in.instance_col.xyz;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.);
}