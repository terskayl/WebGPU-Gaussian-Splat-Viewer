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

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================
  const preprocess_pipeline = device.createComputePipeline({
    label: 'preprocess',
    layout: 'auto',
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

  const camera_bind_group_layout: GPUBindGroupLayout = device.createBindGroupLayout({
    label: "camera bind group layout",
    entries: [{
      binding: 0,
      visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
      buffer: {type: 'uniform'}
    }]
  });

  const gaussian_bind_group_layout: GPUBindGroupLayout = device.createBindGroupLayout({
    label: "gaussians bind group layout",
    entries: [{
      binding: 0,
      visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
      buffer: {type: 'read-only-storage'}
    }]
  });

  const camera_bind_group: GPUBindGroup = device.createBindGroup({
    label: 'gaussian camera bind group',
    layout: camera_bind_group_layout,
    entries: [{binding: 0, resource: { buffer: camera_buffer }}],
  });

  const gaussian_bind_group: GPUBindGroup = device.createBindGroup({
    label: 'gaussian gaussians bind group',
    layout: gaussian_bind_group_layout,
    entries: [
      {binding: 0, resource: { buffer: pc.gaussian_3d_buffer }},
    ],
  });

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

  const instanceCount = 1000;
  const instanceData = new Float32Array(instanceCount * 2);
  for (let i = 0; i < instanceCount; i++) {
    instanceData[i * 2 + 0] = (Math.random() - 0.5) * 2.0; // x offset
    instanceData[i * 2 + 1] = (Math.random() - 0.5) * 2.0; // y offset
  }
  const instanceBuffer = device.createBuffer({
    size: instanceData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(instanceBuffer, 0, instanceData);

  const vertexBuffers: GPUVertexBufferLayout[] = [
    { // Quad Vertices
      arrayStride: 3 * Float32Array.BYTES_PER_ELEMENT,
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x3' },
      ],
    },
    { // Instance Data - 2D Gaussian Data
      // What do we want? position, uv, cov, radius, and color?
      arrayStride: 2 * Float32Array.BYTES_PER_ELEMENT,
      stepMode: 'instance',
      attributes: [
        { shaderLocation: 1, offset: 0, format: 'float32x2' },
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
    pass.setVertexBuffer(1, instanceBuffer);
    pass.setBindGroup(0, camera_bind_group);
    pass.setBindGroup(1, gaussian_bind_group);
    pass.setIndexBuffer(index_buffer, 'uint16');
    pass.drawIndexed(6, instanceCount);

    pass.end();
  };

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      sorter.sort(encoder);
      render(encoder, texture_view);
    },
    camera_buffer,
  };
}
