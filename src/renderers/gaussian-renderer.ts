import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {

}

// Utility to create GPU buffers
const createBuffer = (
  device: GPUDevice,
  label: string,
  size: number,
  usage: GPUBufferUsageFlags,
  data?: ArrayBuffer | ArrayBufferView
) => {
  const buffer = device.createBuffer({ label, size, usage });
  if (data) device.queue.writeBuffer(buffer, 0, data);
  return buffer;
};

export default function get_renderer(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer,
): GaussianRenderer {

  const sorter = get_sorter(pc.num_points, device);

  
  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================

  const nulling_data = new Uint32Array([0]);

  const n = pc.num_points;
  // Buffers for position, uv, covariance, radius, color
  const splat_buffer = createBuffer(
    device,
    'splat buffer - position, uv/rad/alpha, covariance, color',
    n * 16 * Float32Array.BYTES_PER_ELEMENT, // 4 for pos, 4 for uv/rad/alpha, 4 for cov, 4 for color
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX
  );

  const render_settings_buffer = createBuffer(
    device,
    'render settings buffer',
    2 * Uint32Array.BYTES_PER_ELEMENT,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  );

  const gaussian_scaling = 1; // can be parameterized later
  const sh_deg = 3; // hardcoded for now, can be parameterized later
  const render_settings_data = new Uint32Array([gaussian_scaling, sh_deg]);
  device.queue.writeBuffer(render_settings_buffer, 0, render_settings_data);

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================

  const camera_bind_group_layout: GPUBindGroupLayout = device.createBindGroupLayout({
    label: "camera bind group layout",
    entries: [{ // camera
      binding: 0,
      visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
      buffer: {type: 'uniform'}
    },
    { // render settings
      binding: 1,
      visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
      buffer: {type: 'uniform'}
    }]
  });

  const gaussian_bind_group_layout: GPUBindGroupLayout = device.createBindGroupLayout({
    label: "gaussians bind group layout",
    entries: [
      { // 3D Gaussians
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
        buffer: {type: 'read-only-storage'}
      },
      { // SH Coefficients
        binding: 1,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
        buffer: {type: 'read-only-storage'}
      }
    ]
  });
  
  const output_bind_group_layout: GPUBindGroupLayout = device.createBindGroupLayout({
    label: 'output bind group layout',
    entries: [
      { // Splat buffer - Position, UV/rad/alpha, Covariance, Color
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' }
      },
    ]
  });
  
  const sort_bind_group_layout: GPUBindGroupLayout = device.createBindGroupLayout({
    label: 'sort bind group layout',
    entries: [
      { // Sort info
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' }
      },
      { // Sort depths
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' }
      },
      { // Sort indices
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' }
      },
      { // Dispatch indirect
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' }
      }
    ]
  });

  const gaussian_bind_group: GPUBindGroup = device.createBindGroup({
    label: 'gaussian gaussians bind group',
    layout: gaussian_bind_group_layout,
    entries: [
      {binding: 0, resource: { buffer: pc.gaussian_3d_buffer }},
      {binding: 1, resource: { buffer: pc.sh_buffer }},
    ],
  });
  
  const output_bind_group: GPUBindGroup = device.createBindGroup({
    label: 'gaussian output bind group',
    layout: output_bind_group_layout,
    entries: [
      { binding: 0, resource: { buffer: splat_buffer } },
    ]
  });
  
  const camera_bind_group: GPUBindGroup = device.createBindGroup({
    label: 'gaussian camera bind group',
    layout: camera_bind_group_layout,
    entries: [{binding: 0, resource: { buffer: camera_buffer }},
    {binding: 1, resource: { buffer: render_settings_buffer }}
    ],
  });

  const preprocess_pipeline_layout: GPUPipelineLayout = device.createPipelineLayout({
    label: 'preprocess pipeline layout',
    bindGroupLayouts: [
      gaussian_bind_group_layout,
      output_bind_group_layout,
      sort_bind_group_layout,
      camera_bind_group_layout
    ]
  });
  
  const preprocess_pipeline = device.createComputePipeline({
    label: 'preprocess',
    layout: preprocess_pipeline_layout,
    compute: {
      module: device.createShaderModule({ code: preprocessWGSL }),
      entryPoint: 'preprocess',
      constants: {
        workgroupSize: C.histogram_wg_size,
        sortKeyPerThread: c_histogram_block_rows,
      },
    },
  });

  const sort_bind_group = device.createBindGroup({
    label: 'sort',
    layout: preprocess_pipeline.getBindGroupLayout(2),
    entries: [
      { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
      { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
    ],
  });


  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================

  const gaussian_render_pipeline_layout: GPUPipelineLayout = device.createPipelineLayout({
    label: 'gaussian pipeline layout',
    bindGroupLayouts: [
      camera_bind_group_layout,
      gaussian_bind_group_layout
    ]
  })

  const gaussian_render_shader: GPUShaderModule = device.createShaderModule({
    label: "gaussian rendering shader module",
    code: renderWGSL,
  })

  const quad_buffer: GPUBuffer = createBuffer(
    device,
    'quad buffer',
    4 * 3 * Float32Array.BYTES_PER_ELEMENT,
    GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    new Float32Array([
      -0.01, -0.01, 0,
      0.01, -0.01, 0,
      -0.01, 0.01, 0,
      0.01, 0.01, 0
    ])
  );

  const index_buffer: GPUBuffer = createBuffer(
    device,
    'quad index buffer',
    6 * Uint16Array.BYTES_PER_ELEMENT,
    GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    new Uint16Array([
      0, 1, 2,
      1, 3, 2
    ])
  );


  const vertexBuffers: GPUVertexBufferLayout[] = [
    { // Quad Vertices
      arrayStride: 3 * Float32Array.BYTES_PER_ELEMENT,
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x3' },
      ],
    },
    { // Instance Position
      arrayStride: 16 * Float32Array.BYTES_PER_ELEMENT,
      stepMode: 'instance',
      attributes: [
        { shaderLocation: 1, offset: 0, format: 'float32x3' },
      ],
    },
    { // Instance UV/Radius/Alpha
      arrayStride: 16 * Float32Array.BYTES_PER_ELEMENT,
      stepMode: 'instance',
      attributes: [
        { shaderLocation: 2, offset: 4 * Float32Array.BYTES_PER_ELEMENT, format: 'float32x4' },
      ],
    },
    { // Instance Covariance
      arrayStride: 16 * Float32Array.BYTES_PER_ELEMENT,
      stepMode: 'instance',
      attributes: [
        { shaderLocation: 3, offset: 8 * Float32Array.BYTES_PER_ELEMENT, format: 'float32x3' },
      ],
    },
    { // Instance Color
      arrayStride: 16 * Float32Array.BYTES_PER_ELEMENT,
      stepMode: 'instance',
      attributes: [
        { shaderLocation: 4, offset: 12 * Float32Array.BYTES_PER_ELEMENT, format: 'float32x3' },
      ],
    }
  ];

  const gaussian_render_pipeline: GPURenderPipeline = device.createRenderPipeline({
    label: "gaussian render pipeline",
    layout: gaussian_render_pipeline_layout,
    vertex: {
      module: gaussian_render_shader,
      entryPoint: 'vs_main',
      buffers: vertexBuffers,
    },
    fragment: {
      module: gaussian_render_shader,
      entryPoint: 'fs_main',
      targets: [{ format: presentation_format }],
    },
    primitive: {
      topology: 'triangle-list',
    },
  })

  // ===============================================
  //    Command Encoder Functions
  // ===============================================

  const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    
      const null_buffer = createBuffer(
          device, 'null_buffer', 4,
          GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, nulling_data
      );
    

      encoder.copyBufferToBuffer(
          null_buffer, 0,
          sorter.sort_info_buffer, 0,
          4
      );
      encoder.copyBufferToBuffer(
          null_buffer, 0,
          sorter.sort_dispatch_indirect_buffer, 0,
          4
      );

    const preprocess_pass = encoder.beginComputePass({
      label: 'gaussian preprocess pass',
    });
    preprocess_pass.setPipeline(preprocess_pipeline);
    preprocess_pass.setBindGroup(0, gaussian_bind_group);
    preprocess_pass.setBindGroup(1, output_bind_group);
    preprocess_pass.setBindGroup(2, sort_bind_group);
    preprocess_pass.setBindGroup(3, camera_bind_group);
    preprocess_pass.dispatchWorkgroups(C.histogram_wg_size);
    preprocess_pass.end();


    sorter.sort(encoder);

  const indirect_draw_data = new Uint32Array([
    // Initialize to 0 instance. The actual instance count is determined by the preprocess step.
    6, 0, 0, 0 // vertexCount, instanceCount, firstVertex, firstInstance
  ]);

  const indirect_draw_buffer = createBuffer(
    device,
    'indirect draw buffer',
    4 * 4,
    GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    indirect_draw_data
  );

    encoder.copyBufferToBuffer(
        sorter.sort_info_buffer, 0,
        indirect_draw_buffer, 4,
        4
    );

    const pass = encoder.beginRenderPass({
      label: 'gaussian render pass',
      colorAttachments: [
        {
          view: texture_view,
          clearValue: [0, 0, 0, 1],
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });

    pass.setPipeline(gaussian_render_pipeline);
    pass.setVertexBuffer(0, quad_buffer);
    pass.setVertexBuffer(1, splat_buffer);
    pass.setVertexBuffer(2, splat_buffer);
    pass.setVertexBuffer(3, splat_buffer);
    pass.setVertexBuffer(4, splat_buffer);
    pass.setBindGroup(0, camera_bind_group);
    pass.setBindGroup(1, gaussian_bind_group);
    //pass.setIndexBuffer(index_buffer, 'uint16');
    //pass.drawIndexed(6, n);
    pass.drawIndirect(indirect_draw_buffer, 0);

    pass.end();
  };

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      render(encoder, texture_view);
    },
    camera_buffer,
  };
}
