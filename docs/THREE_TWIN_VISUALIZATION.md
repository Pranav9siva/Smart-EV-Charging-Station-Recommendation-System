# Three.js Digital Twin Visualization

## Overview

The new digital twin visualization is a standalone frontend experience that reuses the existing WebSocket stream and the controller-produced snapshot payload. It does not replace the PPO, SUMO, or monitoring logic. Instead, it consumes the live snapshot and renders a modern 3D city scene.

## Features

- Three.js-based 3D scene
- SUMO-inspired road rendering from the network XML
- Moving 3D vehicle meshes
- Glowing EV charging station meshes
- Animated traffic lights
- Orbit, zoom, pan, and follow-vehicle controls
- Dark modern responsive UI

## Entry Points

- Open /three in the visualization app to view the new interface.
- Open /live for the legacy dashboard.

## Frontend Structure

- src/visualization/frontend/three_twin_dashboard.html — page shell and UI
- src/visualization/frontend/assets/js/three-twin/scene.js — 3D scene construction
- src/visualization/frontend/assets/js/three-twin/camera-controller.js — camera controls
- src/visualization/frontend/assets/js/three-twin/renderer.js — state-to-scene bridge

## Notes

- The renderer is intentionally decoupled from PPO logic and only consumes the existing visualization snapshot format.
- The backend WebSocket remains the single source of simulation state.
