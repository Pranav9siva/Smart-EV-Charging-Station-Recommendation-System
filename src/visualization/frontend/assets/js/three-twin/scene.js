import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.161.0/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.161.0/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'https://cdn.jsdelivr.net/npm/three@0.161.0/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'https://cdn.jsdelivr.net/npm/three@0.161.0/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'https://cdn.jsdelivr.net/npm/three@0.161.0/examples/jsm/postprocessing/UnrealBloomPass.js';
import { SSAOPass } from 'https://cdn.jsdelivr.net/npm/three@0.161.0/examples/jsm/postprocessing/SSAOPass.js';

export class DigitalTwinScene {
  constructor(container, options = {}) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.08;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x050816);
    this.scene.fog = new THREE.Fog(0x050816, 180, 1200);
    this.camera = new THREE.PerspectiveCamera(55, container.clientWidth / container.clientHeight, 0.1, 5000);
    this.camera.position.set(180, 140, 220);

    this.clock = new THREE.Clock();
    this.group = new THREE.Group();
    this.scene.add(this.group);

    this.roadGroup = new THREE.Group();
    this.vehicleGroup = new THREE.Group();
    this.stationGroup = new THREE.Group();
    this.trafficLightGroup = new THREE.Group();
    this.buildingGroup = new THREE.Group();
    this.treeGroup = new THREE.Group();
    this.routeGroup = new THREE.Group();
    this.streetLightGroup = new THREE.Group();
    this.sidewalkGroup = new THREE.Group();
    this.signGroup = new THREE.Group();
    this.parkingGroup = new THREE.Group();
    this.pedestrianGroup = new THREE.Group();
    this.weatherGroup = new THREE.Group();
    this.group.add(this.roadGroup, this.vehicleGroup, this.stationGroup, this.trafficLightGroup, this.buildingGroup, this.treeGroup, this.routeGroup, this.streetLightGroup, this.sidewalkGroup, this.signGroup, this.parkingGroup, this.pedestrianGroup, this.weatherGroup);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.enablePan = true;
    this.controls.target.set(0, 0, 0);
    this.controls.maxPolarAngle = Math.PI / 2.2;
    this.controls.minDistance = 40;
    this.controls.maxDistance = 800;

    this.state = {
      vehicles: [],
      stations: [],
      trafficLights: [],
      network: { edges: [], bounds: {} },
      selectedVehicleId: null,
      followVehicleId: null,
      lastFrame: 0,
      vehicleNodes: new Map(),
    };
    this.timeOfDay = 0.35;
    this.weather = { rain: true, fog: true, night: false };
    this._initLights();
    this._initEnvironment();
    this._attachResize();
    this._animate();
  }

  _initLights() {
    this.ambientLight = new THREE.AmbientLight(0x84a4ff, 0.28);
    this.hemiLight = new THREE.HemisphereLight(0x9bdcff, 0x243041, 0.7);
    this.sunLight = new THREE.DirectionalLight(0xffffff, 0.95);
    this.sunLight.position.set(140, 220, 160);
    this.sunLight.castShadow = true;
    this.sunLight.shadow.camera.near = 0.5;
    this.sunLight.shadow.camera.far = 1400;
    this.sunLight.shadow.mapSize.set(1024, 1024);
    this.scene.add(this.ambientLight, this.hemiLight, this.sunLight);
  }

  _initEnvironment() {
    this._createSkybox();
    this._createGround();
    this._createWeatherParticles();
    this._createPostProcessing();
    this._createPedestrians();
  }

  _createSkybox() {
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 256;
    const ctx = canvas.getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, 0, 256);
    gradient.addColorStop(0, '#0f172a');
    gradient.addColorStop(0.4, '#38bdf8');
    gradient.addColorStop(1, '#f8fafc');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 512, 256);
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    const sky = new THREE.Mesh(
      new THREE.SphereGeometry(12000, 32, 32),
      new THREE.MeshBasicMaterial({ map: texture, side: THREE.BackSide })
    );
    this.scene.add(sky);
  }

  _createGround() {
    const groundGeometry = new THREE.PlaneGeometry(12000, 12000);
    const groundMaterial = new THREE.MeshPhysicalMaterial({ color: 0x0b1220, roughness: 0.98, metalness: 0.02, clearcoat: 0.05 });
    const ground = new THREE.Mesh(groundGeometry, groundMaterial);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    ground.frustumCulled = true;
    this.scene.add(ground);

    const reflectivePlane = new THREE.Mesh(
      new THREE.PlaneGeometry(12000, 12000),
      new THREE.MeshPhysicalMaterial({ color: 0x1f2937, metalness: 0.55, roughness: 0.14, transparent: true, opacity: 0.45 })
    );
    reflectivePlane.rotation.x = -Math.PI / 2;
    reflectivePlane.position.y = 0.05;
    reflectivePlane.receiveShadow = true;
    this.scene.add(reflectivePlane);
  }

  _createWeatherParticles() {
    const particleCount = 1800;
    const positions = new Float32Array(particleCount * 3);
    for (let index = 0; index < particleCount; index += 1) {
      positions[index * 3 + 0] = (Math.random() - 0.5) * 4000;
      positions[index * 3 + 1] = Math.random() * 220 + 80;
      positions[index * 3 + 2] = (Math.random() - 0.5) * 4000;
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const material = new THREE.PointsMaterial({ color: 0x8ec5ff, size: 0.8, transparent: true, opacity: 0.7 });
    this.rainPoints = new THREE.Points(geometry, material);
    this.weatherGroup.add(this.rainPoints);
  }

  _createPostProcessing() {
    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloomPass = new UnrealBloomPass(new THREE.Vector2(this.container.clientWidth, this.container.clientHeight), 0.28, 0.35, 0.2);
    this.ssaoPass = new SSAOPass(this.scene, this.camera, this.container.clientWidth, this.container.clientHeight);
    this.ssaoPass.kernelRadius = 8;
    this.ssaoPass.minDistance = 0.001;
    this.ssaoPass.maxDistance = 0.1;
    this.composer.addPass(this.bloomPass);
    this.composer.addPass(this.ssaoPass);
  }

  _createPedestrians() {
    const pedestrianCount = 24;
    for (let index = 0; index < pedestrianCount; index += 1) {
      const group = new THREE.Group();
      const body = new THREE.Mesh(new THREE.BoxGeometry(0.9, 1.8, 0.6), new THREE.MeshStandardMaterial({ color: 0x334155 }));
      const head = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.5, 0.5), new THREE.MeshStandardMaterial({ color: 0xf1f5f9 }));
      head.position.y = 1.5;
      group.add(body, head);
      group.position.set((Math.random() - 0.5) * 1000, 0.9, (Math.random() - 0.5) * 1000);
      group.userData = { speed: 0.5 + Math.random() * 0.7, phase: Math.random() * Math.PI * 2, drift: (Math.random() - 0.5) * 0.8 };
      group.castShadow = true;
      this.pedestrianGroup.add(group);
    }
  }

  _attachResize() {
    window.addEventListener('resize', () => this.resize());
  }

  resize() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    this.renderer.setSize(width, height);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    if (this.composer) {
      this.composer.setSize(width, height);
    }
  }

  _animate() {
    requestAnimationFrame(() => this._animate());
    const delta = this.clock.getDelta();
    this.render(delta);
    this.controls.update();
    if (this.composer) {
      this.composer.render();
    } else {
      this.renderer.render(this.scene, this.camera);
    }
  }

  render(delta) {
    this.timeOfDay = (this.timeOfDay + delta * 0.005) % 1;
    const dayFactor = 0.5 + 0.5 * Math.sin(this.timeOfDay * Math.PI * 2);
    this.weather.night = dayFactor < 0.42;
    const dayColor = new THREE.Color(this.weather.night ? 0x050816 : 0x87ceeb);
    const fogColor = new THREE.Color(this.weather.night ? 0x050816 : 0x91c1ff);
    this.scene.background.copy(dayColor);
    this.scene.fog.color.copy(fogColor);
    this.scene.fog.density = this.weather.fog ? (this.weather.night ? 0.0018 : 0.0011) : 0.0004;
    this.ambientLight.intensity = this.weather.night ? 0.16 : 0.28;
    this.hemiLight.intensity = this.weather.night ? 0.5 : 0.8;
    this.sunLight.intensity = this.weather.night ? 0.12 : 0.95;
    this.sunLight.position.set(160 * Math.cos(this.timeOfDay * Math.PI * 2), 220 + 120 * Math.sin(this.timeOfDay * Math.PI * 2), 160 * Math.sin(this.timeOfDay * Math.PI * 2));

    this.vehicleGroup.children.forEach((child) => {
      if (child.userData?.animate) child.userData.animate(delta);
    });
    this.trafficLightGroup.children.forEach((child) => {
      if (child.userData?.animate) child.userData.animate(delta);
    });
    this.stationGroup.children.forEach((child) => {
      if (child.userData?.animate) child.userData.animate(delta);
    });
    this.routeGroup.children.forEach((child) => {
      if (child.userData?.animate) child.userData.animate(delta);
    });
    this.pedestrianGroup.children.forEach((child) => {
      const walk = Math.sin(this.clock.elapsedTime * 2 + child.userData.phase) * 0.2;
      child.position.x += Math.sin(this.clock.elapsedTime + child.userData.phase) * 0.01 + child.userData.drift * 0.001;
      child.position.z += 0.02 + child.userData.speed * 0.003;
      child.rotation.y = Math.PI / 2;
      child.position.y = 0.9 + walk * 0.05;
      if (child.position.z > 1200) child.position.z = -1200;
      if (child.position.x > 1200) child.position.x = -1200;
    });
    if (this.rainPoints) {
      const positions = this.rainPoints.geometry.attributes.position.array;
      for (let index = 0; index < positions.length / 3; index += 1) {
        positions[index * 3 + 1] -= 4 + (this.weather.rain ? 1 : 0);
        if (positions[index * 3 + 1] < 20) {
          positions[index * 3 + 1] = 220 + Math.random() * 50;
          positions[index * 3 + 0] = (Math.random() - 0.5) * 4000;
          positions[index * 3 + 2] = (Math.random() - 0.5) * 4000;
        }
      }
      this.rainPoints.geometry.attributes.position.needsUpdate = true;
      this.rainPoints.material.opacity = this.weather.rain ? 0.7 : 0;
      this.rainPoints.visible = this.weather.rain;
    }
    if (this.state.followVehicleId) {
      const followNode = this.vehicleGroup.children.find((child) => child.userData?.vehicle?.id === this.state.followVehicleId);
      if (followNode) {
        const target = followNode.position.clone().add(new THREE.Vector3(40, 50, 40));
        this.camera.position.lerp(target, 0.06);
        this.controls.target.lerp(followNode.position.clone().add(new THREE.Vector3(0, 8, 0)), 0.08);
      }
    }
  }

  setRoadNetwork(network) {
    const nextNetwork = network || { edges: [], bounds: {} };
    if (this.state.network === nextNetwork || this.state.networkFingerprint === JSON.stringify(nextNetwork)) {
      return;
    }
    this.state.network = nextNetwork;
    this.state.networkFingerprint = JSON.stringify(nextNetwork);
    this.roadGroup.clear();
    this.sidewalkGroup.clear();
    this.signGroup.clear();
    this.parkingGroup.clear();
    this.streetLightGroup.clear();
    const roadMaterial = new THREE.MeshStandardMaterial({ color: 0x2f3b4d, roughness: 0.95, metalness: 0.08 });
    const laneMaterial = new THREE.MeshStandardMaterial({ color: 0xe5e7eb, emissive: 0x111827, emissiveIntensity: 0.25 });
    const sidewalkMaterial = new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.95, metalness: 0.02 });
    const bounds = network.bounds || {};
    const centerX = ((bounds.minX || 0) + (bounds.maxX || 0)) / 2;
    const centerZ = ((bounds.minY || 0) + (bounds.maxY || 0)) / 2;
    this.group.position.set(-centerX, 0, -centerZ);
    this.frameNetwork(bounds);

    (network.edges || []).forEach((edge, edgeIndex) => {
      const points = edge.points || [];
      if (points.length < 2) return;
      for (let index = 1; index < points.length; index += 1) {
        const start = new THREE.Vector3(points[index - 1][0], 0.12, points[index - 1][1]);
        const end = new THREE.Vector3(points[index][0], 0.12, points[index][1]);
        const length = start.distanceTo(end);
        const road = new THREE.Mesh(new THREE.BoxGeometry(length, 0.24, 7.5), roadMaterial);
        road.position.copy(start.clone().lerp(end, 0.5));
        road.lookAt(end);
        road.rotation.x = 0;
        road.rotation.z = 0;
        road.castShadow = true;
        road.receiveShadow = true;
        road.userData.edgeId = edge.id;
        road.userData.baseColor = roadMaterial.color.clone();
        this.roadGroup.add(road);

        const sidewalk = new THREE.Mesh(new THREE.BoxGeometry(length, 0.12, 12.5), sidewalkMaterial);
        sidewalk.position.copy(start.clone().lerp(end, 0.5));
        sidewalk.position.y = 0.16;
        sidewalk.lookAt(end);
        sidewalk.rotation.x = 0;
        sidewalk.rotation.z = 0;
        this.sidewalkGroup.add(sidewalk);

        const marking = new THREE.Mesh(new THREE.BoxGeometry(Math.max(1.5, length), 0.05, 1.2), laneMaterial);
        marking.position.copy(start.clone().lerp(end, 0.5));
        marking.position.y = 0.18;
        marking.lookAt(end);
        marking.rotation.x = 0;
        marking.rotation.z = 0;
        this.roadGroup.add(marking);

        if (index % 2 === 0) {
          this._createStreetLight(start, end, edgeIndex + index);
          this._createRoadSign(start, end, edgeIndex + index);
        }
        if (edgeIndex % 3 === 0) {
          this._createParkingLot(start, end, edgeIndex + index);
        }
      }
    });

    this._createCityBlocks(network);
    this._createTrees(network);
  }

  frameNetwork(bounds = this.state.network?.bounds || {}) {
    const span = Math.max(Number(bounds.width) || 0, Number(bounds.height) || 0, 100);
    const distance = span * 1.18;
    this.camera.position.set(0, distance, distance * 0.12);
    this.controls.target.set(0, 0, 0);
    this.controls.maxDistance = Math.max(1200, span * 3.0);
    this.controls.update();
  }

  _createCityBlocks(network) {
    this.buildingGroup.clear();
    const bounds = network.bounds || {};
    const minX = bounds.minX || 0;
    const maxX = bounds.maxX || 0;
    const minY = bounds.minY || 0;
    const maxY = bounds.maxY || 0;
    const width = Math.max(20, maxX - minX);
    const height = Math.max(20, maxY - minY);
    const count = Math.max(70, Math.round((width * height) / 2500));
    const level1 = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial({ color: 0x172033, emissive: 0x0f172a, emissiveIntensity: 0.15 }), count);
    const level2 = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial({ color: 0x202a3a, emissive: 0x0f172a, emissiveIntensity: 0.2 }), count);
    const level3 = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial({ color: 0x273447, emissive: 0x111827, emissiveIntensity: 0.2 }), count);
    [level1, level2, level3].forEach((level) => {
      level.castShadow = true;
      level.receiveShadow = true;
      level.frustumCulled = true;
    });
    const lod = new THREE.LOD();
    lod.addLevel(level1, 0);
    lod.addLevel(level2, 120);
    lod.addLevel(level3, 220);
    this.buildingGroup.add(lod);
    const matrix = new THREE.Matrix4();
    for (let index = 0; index < count; index += 1) {
      const x = minX + (Math.random() * width) + 50;
      const z = minY + (Math.random() * height) + 50;
      const widthScale = 6 + Math.random() * 10;
      const depthScale = 6 + Math.random() * 10;
      const heightScale = 12 + Math.random() * 42;
      const translation = new THREE.Vector3(x, heightScale / 2, z);
      const scale = new THREE.Vector3(widthScale, heightScale, depthScale);
      matrix.compose(translation, new THREE.Quaternion(), scale);
      level1.setMatrixAt(index, matrix);
      level2.setMatrixAt(index, matrix);
      level3.setMatrixAt(index, matrix);
    }
  }

  _createTrees(network) {
    this.treeGroup.clear();
    const bounds = network.bounds || {};
    const minX = bounds.minX || 0;
    const maxX = bounds.maxX || 0;
    const minY = bounds.minY || 0;
    const maxY = bounds.maxY || 0;
    const width = Math.max(20, maxX - minX);
    const height = Math.max(20, maxY - minY);
    const count = Math.max(40, Math.round((width * height) / 6000));
    const trunkGeometry = new THREE.CylinderGeometry(0.4, 0.7, 4, 8);
    const trunkMaterial = new THREE.MeshStandardMaterial({ color: 0x4a2c18 });
    const canopyGeometry = new THREE.SphereGeometry(2.6, 8, 8);
    const canopyMaterial = new THREE.MeshStandardMaterial({ color: 0x1f7d3b });
    const instancedTrunk = new THREE.InstancedMesh(trunkGeometry, trunkMaterial, count);
    const instancedCanopy = new THREE.InstancedMesh(canopyGeometry, canopyMaterial, count);
    [instancedTrunk, instancedCanopy].forEach((mesh) => {
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.frustumCulled = true;
    });
    const matrix = new THREE.Matrix4();
    for (let index = 0; index < count; index += 1) {
      const x = minX + (Math.random() * width) + 20;
      const z = minY + (Math.random() * height) + 20;
      const trunk = new THREE.Matrix4().makeTranslation(x, 2, z);
      const canopy = new THREE.Matrix4().makeTranslation(x, 5.7, z);
      matrix.copy(trunk);
      instancedTrunk.setMatrixAt(index, matrix);
      matrix.copy(canopy);
      instancedCanopy.setMatrixAt(index, matrix);
    }
    this.treeGroup.add(instancedTrunk, instancedCanopy);
  }

  _createStreetLight(start, end, index) {
    const group = new THREE.Group();
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.25, 10, 8), new THREE.MeshStandardMaterial({ color: 0x334155 }));
    pole.position.y = 5;
    pole.castShadow = true;
    const light = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.4, 0.4), new THREE.MeshStandardMaterial({ color: 0xfef3c7, emissive: 0xfef3c7, emissiveIntensity: 0.8 }));
    light.position.set(0, 7.4, 0.5);
    const base = new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.2, 1.2), new THREE.MeshStandardMaterial({ color: 0x475569 }));
    base.position.y = 0.1;
    group.add(pole, light, base);
    const position = start.clone().lerp(end, 0.35 + (index % 3) * 0.1);
    position.y = 0;
    group.position.copy(position);
    group.lookAt(end);
    group.userData.animate = () => {
      light.material.emissiveIntensity = this.weather.night ? 0.95 : 0.25;
    };
    this.streetLightGroup.add(group);
  }

  _createRoadSign(start, end, index) {
    const sign = new THREE.Group();
    const post = new THREE.Mesh(new THREE.BoxGeometry(0.18, 4.2, 0.18), new THREE.MeshStandardMaterial({ color: 0x64748b }));
    post.position.y = 2.1;
    const panel = new THREE.Mesh(new THREE.BoxGeometry(2.2, 1.2, 0.15), new THREE.MeshStandardMaterial({ color: 0xf8fafc }));
    panel.position.set(0.8, 3.6, 0.1);
    const arrow = new THREE.Mesh(new THREE.ConeGeometry(0.45, 1.0, 8), new THREE.MeshStandardMaterial({ color: 0x38bdf8 }));
    arrow.position.set(0.8, 3.6, 0.2);
    arrow.rotation.z = Math.PI / 2;
    sign.add(post, panel, arrow);
    const position = start.clone().lerp(end, 0.2 + (index % 2) * 0.15);
    position.y = 0;
    sign.position.copy(position);
    sign.lookAt(end);
    this.signGroup.add(sign);
  }

  _createParkingLot(start, end) {
    const center = start.clone().lerp(end, 0.5);
    const lot = new THREE.Group();
    const base = new THREE.Mesh(new THREE.PlaneGeometry(18, 18), new THREE.MeshStandardMaterial({ color: 0x111827, roughness: 0.95, metalness: 0.02 }));
    base.rotation.x = -Math.PI / 2;
    base.position.y = 0.04;
    lot.add(base);
    for (let row = 0; row < 3; row += 1) {
      for (let col = 0; col < 3; col += 1) {
        const line = new THREE.Mesh(new THREE.BoxGeometry(3.2, 0.02, 0.1), new THREE.MeshStandardMaterial({ color: 0xf8fafc }));
        line.position.set(-3 + col * 3, 0.05, -4 + row * 4);
        line.rotation.x = Math.PI / 2;
        lot.add(line);
      }
    }
    lot.position.copy(center);
    lot.lookAt(end);
    lot.position.y = 0;
    this.parkingGroup.add(lot);
  }

  setVehicles(vehicles, trackedIds = []) {
    this.state.vehicles = vehicles || [];
    this.state.trackedIds = Array.isArray(trackedIds) ? trackedIds : [];
    const currentIds = new Set();
    this.state.vehicles.forEach((vehicle) => {
      const vehicleId = vehicle.id || vehicle.vehicle_id;
      if (!vehicleId) {
        console.error('[3D VEHICLES] missing authoritative vehicle ID', vehicle);
        return;
      }
      currentIds.add(vehicleId);
      const existing = this.state.vehicleNodes.get(vehicleId);
      if (existing) {
        existing.userData.updateVehicle(vehicle);
        return;
      }
      const mesh = this._createVehicle(vehicle);
      this.state.vehicleNodes.set(vehicleId, mesh);
      this.vehicleGroup.add(mesh);
    });
    for (const [vehicleId, node] of this.state.vehicleNodes) {
      if (currentIds.has(vehicleId)) continue;
      this.vehicleGroup.remove(node);
      node.traverse((object) => {
        if (object.geometry) object.geometry.dispose();
        if (object.material) {
          const materials = Array.isArray(object.material) ? object.material : [object.material];
          materials.forEach((material) => material.dispose());
        }
      });
      this.state.vehicleNodes.delete(vehicleId);
    }
  }

  _createVehicle(vehicle) {
    const group = new THREE.Group();
    const color = this._getVehicleColor(vehicle.id || vehicle.vehicle_id || 'ev');
    const body = new THREE.Mesh(new THREE.BoxGeometry(4.8, 2.4, 8.5), new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.28 }));
    body.position.y = 3.5;
    body.castShadow = true;
    const cabin = new THREE.Mesh(new THREE.BoxGeometry(3.2, 1.8, 4.5), new THREE.MeshStandardMaterial({ color: 0x0f172a, emissive: 0x020617, emissiveIntensity: 0.2 }));
    cabin.position.set(0, 4.8, 0.4);
    cabin.castShadow = true;
    const wheelMaterial = new THREE.MeshStandardMaterial({ color: 0x111111 });
    const wheelGeometry = new THREE.CylinderGeometry(1.1, 1.1, 0.8, 10);
    const leftWheel = new THREE.Mesh(wheelGeometry, wheelMaterial);
    leftWheel.rotation.z = Math.PI / 2;
    leftWheel.position.set(-1.8, 1.2, 3.1);
    const rightWheel = leftWheel.clone();
    rightWheel.position.set(1.8, 1.2, 3.1);
    const rearLeftWheel = leftWheel.clone();
    rearLeftWheel.position.set(-1.8, 1.2, -3.1);
    const rearRightWheel = rightWheel.clone();
    rearRightWheel.position.set(1.8, 1.2, -3.1);
    group.add(body, cabin, leftWheel, rightWheel, rearLeftWheel, rearRightWheel);

    const batteryBar = new THREE.Mesh(new THREE.BoxGeometry(3.5, 0.3, 0.3), new THREE.MeshStandardMaterial({ color: 0x22c55e }));
    batteryBar.position.set(0, 6.2, 0.5);
    if (vehicle.is_ev !== false) group.add(batteryBar);

    const trackedHalo = new THREE.Mesh(
      new THREE.TorusGeometry(5.5, 0.25, 12, 24),
      new THREE.MeshBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: 0.0 })
    );
    trackedHalo.rotation.x = Math.PI / 2;
    trackedHalo.position.y = 0.5;
    group.add(trackedHalo);

    const canvas = this._makeLabelCanvas(vehicle.id || vehicle.vehicle_id || 'EV', vehicle.battery_pct, vehicle.is_ev !== false);
    const texture = new THREE.CanvasTexture(canvas);
    const spriteMaterial = new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false });
    const sprite = new THREE.Sprite(spriteMaterial);
    sprite.scale.set(8, 3, 1);
    sprite.position.set(0, 8.8, 0);
    group.add(sprite);

    const startPos = this._resolvePosition(vehicle.current_position || vehicle.position || {});
    group.position.set(startPos.x, 0, startPos.z);

    group.userData = {
      vehicle,
      targetPosition: startPos.clone(),
      lastPosition: startPos.clone(),
      laneOffset: 0,
      turnPhase: 0,
      animate: (delta) => {
        const next = this._resolvePosition(vehicle.current_position || vehicle.position || {});
        const target = next.clone();
        const interpolation = Math.min(1, 0.18 + delta * 3.5);
        group.position.lerp(target, interpolation);
        const deltaVector = group.position.clone().sub(group.userData.lastPosition || group.position.clone());
        if (Number.isFinite(vehicle.heading_deg)) {
          group.rotation.y = THREE.MathUtils.degToRad(vehicle.heading_deg);
        } else if (deltaVector.lengthSq() > 0.0001) {
          group.rotation.y = Math.atan2(deltaVector.x, deltaVector.z) + Math.PI / 2;
          group.userData.turnPhase = Math.min(1, group.userData.turnPhase + delta * 2.2);
        } else {
          group.userData.turnPhase = Math.max(0, group.userData.turnPhase - delta * 2.2);
        }
        group.userData.lastPosition = group.position.clone();
        const selected = this.state.selectedVehicleId && this.state.selectedVehicleId === (vehicle.id || vehicle.vehicle_id);
        const isTracked = Boolean(vehicle.tracked) || this.state.trackedIds.includes(vehicle.id || vehicle.vehicle_id);
        body.material.emissive.setHex(selected ? 0x4ade80 : (isTracked ? 0x22d3ee : color));
        body.material.color.setHex(selected ? 0x4ade80 : (isTracked ? 0x38bdf8 : color));
        if (vehicle.is_ev !== false) {
          batteryBar.material.color.setHex(vehicle.battery_pct != null && vehicle.battery_pct < 20 ? 0xef4444 : 0x22c55e);
          batteryBar.scale.setX(Math.max(0.2, (vehicle.battery_pct || 0) / 100));
        }
        trackedHalo.material.opacity = isTracked ? 0.85 : 0.0;
        trackedHalo.rotation.z += isTracked ? 0.02 : 0.0;
        trackedHalo.scale.setScalar(isTracked ? (1 + 0.08 * Math.sin(performance.now() * 0.006)) : 1);
        if (vehicle.charging_status === 'waiting') {
          const queueGlow = new THREE.Color(0xf59e0b);
          body.material.emissive.setHex(queueGlow.getHex());
          body.scale.setScalar(1 + 0.02 * Math.sin(performance.now() * 0.01));
        } else if (vehicle.charging_target || vehicle.charging_status === 'charging') {
          const chargeGlow = new THREE.Color(0x2dd4bf);
          body.material.emissive.setHex(chargeGlow.getHex());
          body.scale.setScalar(1 + 0.03 * Math.sin(performance.now() * 0.008));
        } else {
          body.scale.setScalar(1);
        }
      },
    };
    group.userData.updateVehicle = (nextVehicle) => {
      vehicle = nextVehicle;
      group.userData.vehicle = nextVehicle;
    };

    return group;
  }

  _resolvePosition(position) {
    if (!position) return new THREE.Vector3(0, 0, 0);
    const x = position.lon ?? position.x ?? 0;
    const z = position.lat ?? position.z ?? 0;
    return new THREE.Vector3(x, 0, z);
  }

  _getVehicleColor(id) {
    const value = id.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
    const hue = (value * 37) % 360;
    return new THREE.Color(`hsl(${hue}, 70%, 55%)`).getHex();
  }

  _makeLabelCanvas(id, batteryPct, isEv = true) {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 128;
    const context = canvas.getContext('2d');
    context.fillStyle = 'rgba(2,6,23,0.85)';
    context.fillRect(0, 0, 256, 128);
    context.strokeStyle = '#38bdf8';
    context.strokeRect(8, 8, 240, 112);
    context.font = 'bold 24px Inter, sans-serif';
    context.fillStyle = '#f8fafc';
    context.fillText(String(id).slice(0, 12), 24, 44);
    context.font = '20px Inter, sans-serif';
    context.fillStyle = '#38bdf8';
    context.fillText(isEv ? `Battery ${Math.round(batteryPct ?? 0)}%` : 'TRAFFIC', 24, 84);
    return canvas;
  }

  setStations(stations) {
    this.state.stations = stations || [];
    this.stationGroup.clear();
    this.state.stations.forEach((station, index) => {
      const mesh = this._createStation(station, index);
      if (mesh) this.stationGroup.add(mesh);
    });
  }

  _createStation(station, index) {
    const group = new THREE.Group();
    const totalPorts = Math.max(1, Number(station.total_ports || 1));
    const occupiedPorts = Math.max(0, Number(station.occupied_ports || 0));
    const utilization = Math.min(1, occupiedPorts / totalPorts);
    const color = new THREE.Color().setHSL(0.35 - utilization * 0.35, 0.75, 0.4 + utilization * 0.15);
    const base = new THREE.Mesh(new THREE.BoxGeometry(10, 2.5, 10), new THREE.MeshStandardMaterial({ color: color.getHex(), emissive: color.getHex(), emissiveIntensity: 0.2 }));
    base.position.y = 1.25;
    base.castShadow = true;
    const charger = new THREE.Mesh(new THREE.CylinderGeometry(2.2, 2.2, 6, 18), new THREE.MeshStandardMaterial({ color: 0x7dd3fc, emissive: 0x38bdf8, emissiveIntensity: 0.8 }));
    charger.position.y = 4.2;

    const portGroup = new THREE.Group();
    for (let port = 0; port < Math.max(1, station.total_ports || 3); port += 1) {
      const portMesh = new THREE.Mesh(new THREE.BoxGeometry(1.2, 1.2, 0.5), new THREE.MeshStandardMaterial({ color: port < (station.occupied_ports || 0) ? 0xef4444 : 0x22c55e }));
      portMesh.position.set(-2.2 + port * 1.4, 2.6, 4.4);
      portGroup.add(portMesh);
    }
    const queueBar = new THREE.Mesh(new THREE.BoxGeometry(5, 0.3, 0.3), new THREE.MeshStandardMaterial({ color: 0xf59e0b }));
    queueBar.position.set(0, 5.8, 0.4);
    const queueIndicator = new THREE.Mesh(new THREE.BoxGeometry(4.5 * Math.max(0.2, Math.min(1, (station.queue_length || 0) / 6)), 0.3, 0.2), new THREE.MeshStandardMaterial({ color: 0xfacc15 }));
    queueIndicator.position.set(-0.25 + (2.25 * Math.max(0.2, Math.min(1, (station.queue_length || 0) / 6))), 5.8, 0.5);

    const halo = new THREE.Mesh(new THREE.TorusGeometry(6.5, 0.35, 16, 32), new THREE.MeshBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.55 }));
    halo.rotation.x = Math.PI / 2;
    halo.position.y = 1.25;
    const batteryIcon = new THREE.Mesh(new THREE.BoxGeometry(1.6, 1.2, 0.2), new THREE.MeshStandardMaterial({ color: 0xffffff }));
    batteryIcon.position.set(0, 3.2, 5.2);

    const canvas = this._makeStationLabelCanvas(station.station_id || `Station ${index + 1}`, station.available_ports || 0, station.occupied_ports || 0);
    const texture = new THREE.CanvasTexture(canvas);
    const spriteMaterial = new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false });
    const sprite = new THREE.Sprite(spriteMaterial);
    sprite.scale.set(12, 4, 1);
    sprite.position.set(0, 8.5, 0);

    group.add(base, charger, portGroup, queueBar, queueIndicator, halo, batteryIcon, sprite);
    group.userData.animate = () => {
      halo.rotation.z += 0.01;
      halo.scale.setScalar(1 + 0.06 * Math.sin(Date.now() * 0.002));
      const portPulse = Math.max(0.2, 1 - (station.occupied_ports || 0) / Math.max(1, station.total_ports || 1));
      charger.material.emissiveIntensity = 0.5 + portPulse * 0.4;
      if (station.is_recommended) {
        halo.material.opacity = 0.95;
        halo.material.color.setHex(0x4ade80);
      }
    };
    const x = station.position?.x ?? station.x;
    const z = station.position?.z ?? station.z;
    if (!Number.isFinite(x) || !Number.isFinite(z)) {
      console.warn('[3D STATIONS] malformed station position', station.station_id);
      return null;
    }
    group.position.set(x, 0, z);
    group.userData = { station, animate: () => {
      halo.rotation.z += 0.01;
      halo.scale.setScalar(1 + 0.06 * Math.sin(Date.now() * 0.002));
      if (station.is_recommended) {
        halo.material.opacity = 0.95;
        halo.material.color.setHex(0x4ade80);
      }
    }};
    return group;
  }

  _makeStationLabelCanvas(name, availablePorts, occupiedPorts) {
    const canvas = document.createElement('canvas');
    canvas.width = 320;
    canvas.height = 160;
    const context = canvas.getContext('2d');
    context.fillStyle = 'rgba(2,6,23,0.85)';
    context.fillRect(0, 0, 320, 160);
    context.strokeStyle = '#38bdf8';
    context.strokeRect(8, 8, 304, 144);
    context.font = 'bold 24px Inter, sans-serif';
    context.fillStyle = '#f8fafc';
    context.fillText(name, 22, 44);
    context.font = '20px Inter, sans-serif';
    context.fillStyle = '#38bdf8';
    context.fillText(`Ports ${availablePorts} / ${occupiedPorts}`, 22, 84);
    return canvas;
  }

  setTrafficLights(trafficLights) {
    this.state.trafficLights = trafficLights || [];
    this.trafficLightGroup.clear();
    this.state.trafficLights.forEach((light, index) => {
      const mesh = this._createTrafficLight(light, index);
      this.trafficLightGroup.add(mesh);
    });
  }

  _createTrafficLight(light, index) {
    const group = new THREE.Group();
    const pole = new THREE.Mesh(new THREE.BoxGeometry(0.5, 10, 0.5), new THREE.MeshStandardMaterial({ color: 0x111827 }));
    pole.position.y = 5;
    const box = new THREE.Mesh(new THREE.BoxGeometry(2.2, 2.6, 0.8), new THREE.MeshStandardMaterial({ color: 0x111827 }));
    box.position.y = 8.2;
    const red = new THREE.Mesh(new THREE.SphereGeometry(0.5, 16, 16), new THREE.MeshStandardMaterial({ color: 0xef4444, emissive: 0xef4444, emissiveIntensity: 0.2 }));
    red.position.set(0, 8.9, 0.6);
    const amber = new THREE.Mesh(new THREE.SphereGeometry(0.5, 16, 16), new THREE.MeshStandardMaterial({ color: 0xf59e0b, emissive: 0xf59e0b, emissiveIntensity: 0.2 }));
    amber.position.set(0, 8.2, 0.6);
    const green = new THREE.Mesh(new THREE.SphereGeometry(0.5, 16, 16), new THREE.MeshStandardMaterial({ color: 0x22c55e, emissive: 0x22c55e, emissiveIntensity: 0.2 }));
    green.position.set(0, 7.5, 0.6);
    group.add(pole, box, red, amber, green);
    group.position.set(light.x ?? index * 12, 0, light.z ?? index * 12);
    group.userData.animate = () => {
      const phase = (Date.now() * 0.001 + index) % 6;
      red.material.emissiveIntensity = phase < 1.4 ? 1 : 0.2;
      amber.material.emissiveIntensity = phase >= 1.4 && phase < 3 ? 1 : 0.2;
      green.material.emissiveIntensity = phase >= 3 ? 1 : 0.2;
    };
    return group;
  }

  setRoutes(routes) {
    this.routeGroup.clear();
    (routes || []).forEach((route) => {
      if (!route || !route.points || route.points.length < 2) return;
      const points = route.points.map((point) => new THREE.Vector3(point.x ?? point.lon ?? 0, 0.35, point.z ?? point.lat ?? 0));
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const material = new THREE.LineBasicMaterial({ color: route.color || 0x38bdf8, linewidth: 2 });
      const line = new THREE.Line(geometry, material);
      line.userData.animate = () => {
        line.material.opacity = 0.5 + 0.2 * Math.sin(Date.now() * 0.002);
      };
      this.routeGroup.add(line);
    });
  }

  setTrafficDensity(vehicles) {
    const density = new Map();
    (vehicles || []).forEach((vehicle) => {
      if (vehicle.current_edge) density.set(vehicle.current_edge, (density.get(vehicle.current_edge) || 0) + 1);
    });
    this.roadGroup.children.forEach((road) => {
      const edgeId = road.userData?.edgeId;
      if (!edgeId || !road.material?.color) return;
      const count = density.get(edgeId) || 0;
      const congestion = Math.min(1, count / 5);
      road.material.color.setHSL(0.58 - congestion * 0.58, 0.8, 0.25 + congestion * 0.18);
    });
  }

  highlightStation(stationId) {
    this.stationGroup.children.forEach((child) => {
      const station = child.userData?.station;
      if (!station) return;
      const stationKey = station.station_id || station.id;
      const isSelected = stationKey === stationId;
      child.traverse((object) => {
        if (object.isMesh && object.material) {
          if (Array.isArray(object.material)) {
            object.material.forEach((material) => {
              if (material.emissive) material.emissiveIntensity = isSelected ? 1.2 : 0.2;
            });
          } else if (object.material.emissive) {
            object.material.emissiveIntensity = isSelected ? 1.2 : 0.2;
          }
        }
      });
      if (child.children?.[0]) {
        const halo = child.children.find((entry) => entry.type === 'Mesh' && entry.geometry?.type === 'TorusGeometry');
        if (halo) {
          halo.material.opacity = isSelected ? 0.95 : 0.4;
          halo.scale.setScalar(isSelected ? 1.25 : 1);
        }
      }
    });
  }

  setSelectedVehicle(id) {
    this.state.selectedVehicleId = id || null;
  }

  setFollowVehicle(id) {
    this.state.followVehicleId = id || null;
  }
}
