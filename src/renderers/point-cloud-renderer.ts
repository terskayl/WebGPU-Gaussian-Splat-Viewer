import { PointCloud } from '../utils/load';
import pointcloud_wgsl from '../shaders/point_cloud.wgsl';
import { Renderer } from './renderer';

export default function get_renderer(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer): Renderer {
  const render_shader = device.createShaderModule({
    label: "point cloud shader",
    code: pointcloud_wgsl});
  const render_pipeline = device.createRenderPipeline({
    label: 'render',
    layout: 'auto',
    vertex: {
      module: render_shader,
      entryPoint: 'vs_main',
    },
    fragment: {
      module: render_shader,
      entryPoint: 'fs_main',
      targets: [{ format: presentation_format }],
    },
    primitive: {
      topology: 'point-list',
    },
  });

  const camera_bind_group = device.createBindGroup({
    label: 'point cloud camera',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [{binding: 0, resource: { buffer: camera_buffer }}],
  });

  const gaussian_bind_group = device.createBindGroup({
    label: 'point cloud gaussians',
    layout: render_pipeline.getBindGroupLayout(1),
    entries: [
      {binding: 0, resource: { buffer: pc.gaussian_3d_buffer }},
    ],
  });

  const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    const pass = encoder.beginRenderPass({
      label: 'point cloud render',
      colorAttachments: [
        {
          view: texture_view,
          loadOp: 'clear',
          storeOp: 'store',
        }
      ],
    });
    pass.setPipeline(render_pipeline);
    pass.setBindGroup(0, camera_bind_group);
    pass.setBindGroup(1, gaussian_bind_group);

    pass.draw(pc.num_points);
    pass.end();
  };

  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      render(encoder, texture_view);
    },

    camera_buffer,
  };
}