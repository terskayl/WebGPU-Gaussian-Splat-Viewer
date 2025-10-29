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
    @location (1) conic: vec3<f32>,
    @location (2) color: vec3<f32>,
    @location (3) center: vec2<f32>,
};

struct Splat {
    position: vec3<f32>,
    uvRadAlpha: vec4<f32>,
    conic: vec3<f32>,
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
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
    );

    let scale = splat.uvRadAlpha.z / camera.viewport;
    let pos = scale * vertices[vertexId] + splat.position.xy;
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.uvRadAlpha = splat.uvRadAlpha;
    out.color = splat.color.xyz;
    out.conic = splat.conic;
    out.center = splat.position.xy;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Position to NDC
    var position = (in.position.xy / camera.viewport) * 2.0f - 1.0f;
    position.y *= -1.0f;

    // compute the offset in screen-space position
    var diff = position.xy - in.center;
    diff *= camera.viewport * 0.5f;

    let exponent = -0.5 * (in.conic.x * diff.x * diff.x - 2.0 * in.conic.y * diff.x * diff.y + in.conic.z * diff.y * diff.y);
    if (exponent > 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let alpha = min(0.99, in.uvRadAlpha.w * exp(exponent));
    var color = vec4<f32>(in.color, alpha);
    color = vec4<f32>(color.rgb, color.a);

    return color;
}