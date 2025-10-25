import { vec3, mat3, mat4, quat } from 'wgpu-matrix';
import { Camera } from './camera';

export class CameraControl {
  element: HTMLCanvasElement;
  constructor(private camera: Camera) {
    this.register_element(camera.canvas);
  }

  register_element(value: HTMLCanvasElement) {
    if (this.element && this.element != value) {
      this.element.removeEventListener('pointerdown', this.downCallback.bind(this));
      this.element.removeEventListener('pointermove', this.moveCallback.bind(this));
      this.element.removeEventListener('pointerup', this.upCallback.bind(this));
      this.element.removeEventListener('wheel', this.wheelCallback.bind(this));
    }

    this.element = value;
    this.element.addEventListener('pointerdown', this.downCallback.bind(this));
    this.element.addEventListener('pointermove', this.moveCallback.bind(this));
    this.element.addEventListener('pointerup', this.upCallback.bind(this));
    this.element.addEventListener('wheel', this.wheelCallback.bind(this));
    this.element.addEventListener('contextmenu', (e) => { e.preventDefault(); });
  }

  private panning = false;
  private rotating = false;
  private lastX: number;
  private lastY: number;

  downCallback(event: PointerEvent) {
    if (!event.isPrimary) {
      return;
    }

    if (event.button === 0) {
      this.rotating = true;
      this.panning = false;
    } else {
      this.rotating = false;
      this.panning = true;
    }
    this.lastX = event.pageX;
    this.lastY = event.pageY;
  }
  moveCallback(event: PointerEvent) {
    if (!(this.rotating || this.panning)) {
      return;
    }

    const xDelta = event.pageX - this.lastX;
    const yDelta = event.pageY - this.lastY;
    this.lastX = event.pageX;
    this.lastY = event.pageY;

    if (this.rotating) {
      this.rotate(xDelta, yDelta);
    } else if (this.panning) {
      this.pan(xDelta, yDelta);
    }
  }
  upCallback(event: PointerEvent) {
    this.rotating = false;
    this.panning = false;
    event.preventDefault();
  }
  wheelCallback(event: WheelEvent) {
    event.preventDefault();
    const delta = vec3.mulScalar(this.camera.look, -event.deltaY * 0.001);
    vec3.add(delta, this.camera.position, this.camera.position);
    this.camera.update_buffer();
  }

  rotate(xDelta: number, yDelta: number) {
    const lookAtVector = vec3.mulScalar(this.camera.look, 5);
    const lookAtLoc = vec3.add(this.camera.position, lookAtVector);
    this.camera.look = vec3.normalize(this.camera.look)
    var phi =  Math.acos(this.camera.look[1]);
    var theta = Math.atan2(this.camera.look[2], this.camera.look[0]);

    phi += -yDelta * 0.01;
    theta += -xDelta * 0.01;

    const EPS = 0.0001;
    phi = Math.max(EPS, Math.min(Math.PI - EPS, phi));

    let newDir = vec3.create(Math.sin(phi) * Math.cos(theta),
                             Math.cos(phi),
                             Math.sin(phi) * Math.sin(theta)) 

    const newLookAtVector = vec3.mulScalar(newDir, 5);
    this.camera.position = vec3.sub(lookAtLoc, newLookAtVector);
    this.camera.look = vec3.normalize(newDir);
    this.camera.right = vec3.normalize(vec3.cross([0, 1, 0], this.camera.look));
    this.camera.up = vec3.normalize(vec3.cross(this.camera.look, this.camera.right));

    this.camera.rotation = mat4.create(
      this.camera.right[0], this.camera.up[0], this.camera.look[0], 0,
      this.camera.right[1], this.camera.up[1], this.camera.look[1], 0,
      this.camera.right[2], this.camera.up[2], this.camera.look[2], 0,
      0, 0, 0, 1
    );

    //mat4.rotateY(r, -xDelta * 0.01, r);
    //mat4.rotateX(r, yDelta * 0.01, r);
    //const r = mat4.fromQuat(quat.fromEuler(yDelta * 0.01, -xDelta * 0.01, 0, 'xyz'));

    //mat4.mul(r, this.camera.rotation, this.camera.rotation);

    this.camera.update_buffer();
  }

  pan(xDelta: number, yDelta: number) {
    const d = vec3.copy(this.camera.up);
    vec3.mulScalar(d, -yDelta * 0.01, d);
    vec3.add(d, this.camera.position, this.camera.position);
    vec3.copy(this.camera.right, d);
    vec3.mulScalar(d, -xDelta * 0.01, d);
    vec3.add(d, this.camera.position, this.camera.position);
    this.camera.update_buffer();
  }
};