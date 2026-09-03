export class CameraController {
  constructor(camera, domElement, controls) {
    this.camera = camera;
    this.domElement = domElement;
    this.controls = controls;
    this.followVehicle = null;
  }

  setFollowVehicle(vehicle) {
    this.followVehicle = vehicle;
  }

  update() {
    if (this.followVehicle) {
      const pos = this.followVehicle.current_position || this.followVehicle.position || {};
      const target = { x: pos.lon ?? pos.x ?? 0, z: pos.lat ?? pos.z ?? 0 };
      this.controls.target.set(target.x, 0, target.z);
      this.camera.position.lerp({ x: target.x + 60, y: 45, z: target.z + 60 }, 0.1);
    }
  }
}
