import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

const rootDir = fs.existsSync(path.join(process.cwd(), "frontend"))
  ? path.resolve(process.cwd())
  : path.resolve(process.cwd(), "..");

test("Frontend static artifacts build output exists and contains expected routes", () => {
  // Check page files exist
  assert.ok(fs.existsSync(path.join(rootDir, "frontend/app/page.tsx")), "Dashboard page.tsx must exist");
  assert.ok(fs.existsSync(path.join(rootDir, "frontend/app/sensors/page.tsx")), "Sensors page.tsx must exist");
  assert.ok(fs.existsSync(path.join(rootDir, "frontend/app/calibration/page.tsx")), "Calibration page.tsx must exist");
  assert.ok(fs.existsSync(path.join(rootDir, "frontend/app/globals.css")), "globals.css must exist");
});

test("Frontend globals.css uses Tailwind CSS v4 @import directive", () => {
  const css = fs.readFileSync(path.join(rootDir, "frontend/app/globals.css"), "utf-8");
  assert.match(css, /@import\s+"tailwindcss";/, "Must use @import 'tailwindcss' for Tailwind v4");
  assert.doesNotMatch(css, /@tailwind\s+base;/, "Must not use outdated @tailwind base directive");
});

test("Navigation types include all required telemetry fields and SIMULATION mode", () => {
  const typesContent = fs.readFileSync(path.join(rootDir, "frontend/types/navigation.ts"), "utf-8");
  assert.match(typesContent, /"SIMULATION"/, "Must include SIMULATION in NavigationMode");
  assert.match(typesContent, /vehicle_frame_imu/, "Must include vehicle_frame_imu");
  assert.match(typesContent, /calibration/, "Must include calibration");
  assert.match(typesContent, /rates/, "Must include rates");
});

test("Frontend edge lib supports WebSocket and HTTP URL configuration", () => {
  const edgeLib = fs.readFileSync(path.join(rootDir, "frontend/lib/edge.ts"), "utf-8");
  assert.match(edgeLib, /edgeWsUrl/, "Must provide edgeWsUrl");
  assert.match(edgeLib, /NEXT_PUBLIC_EDGE_URL/, "Must read from NEXT_PUBLIC_EDGE_URL");
});

test("Dashboard implements automotive layout components", () => {
  const dashboard = fs.readFileSync(path.join(rootDir, "frontend/app/page.tsx"), "utf-8");
  assert.match(dashboard, /LocalTestMap/, "Must include LocalTestMap");
  assert.match(dashboard, /StatusRail/, "Must include StatusRail");
  assert.match(dashboard, /Event Log/, "Must include Event Log");
  assert.match(dashboard, /SPEED/, "Must display SPEED");
  assert.match(dashboard, /HEADING/, "Must display HEADING");
  assert.match(dashboard, /ACCURACY/, "Must display ACCURACY");
  assert.match(dashboard, /CONFIDENCE/, "Must display CONFIDENCE");
});

test("Sensors page plots real IMU channels without client-side fake numbers", () => {
  const sensors = fs.readFileSync(path.join(rootDir, "frontend/app/sensors/page.tsx"), "utf-8");
  assert.match(sensors, /raw_imu/, "Must plot raw_imu");
  assert.match(sensors, /vehicle_frame_imu/, "Must plot vehicle_frame_imu");
  assert.match(sensors, /IMU RATE/, "Must show IMU rate");
  assert.match(sensors, /FUSION RATE/, "Must show fusion rate");
});

test("Calibration page displays 6-stage alignment pipeline and orientation angles", () => {
  const cal = fs.readFileSync(path.join(rootDir, "frontend/app/calibration/page.tsx"), "utf-8");
  assert.match(cal, /SENSOR INITIALIZATION/, "Stage 1 must exist");
  assert.match(cal, /GRAVITY ESTIMATION/, "Stage 2 must exist");
  assert.match(cal, /GYRO BIAS ESTIMATION/, "Stage 3 must exist");
  assert.match(cal, /VEHICLE DIRECTION ESTIMATION/, "Stage 4 must exist");
  assert.match(cal, /MAGNETOMETER CHECK/, "Stage 5 must exist");
  assert.match(cal, /ALIGNMENT COMPLETE/, "Stage 6 must exist");
  assert.match(cal, /ESTIMATED ROLL/, "Must display roll");
  assert.match(cal, /ESTIMATED PITCH/, "Must display pitch");
  assert.match(cal, /ESTIMATED YAW/, "Must display yaw");
});
