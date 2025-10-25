struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) instanceData: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
};

struct Splat {
    //TODO: information defined in preprocess compute shader
    position: vec3<f32>,
    diffuseColor: vec3<f32>

};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass 
    var out: VertexOutput;
    
    let pos = in.position + vec3<f32>(in.instanceData, 0.0);
    out.position = vec4<f32>(pos, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(1.);
}