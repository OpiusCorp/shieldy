// End-to-end: start the relay, connect simulated phones over SSE, and drive the
// detection → vote → trigger → timeline flow through the real HTTP API.
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
import net from "node:net";

const SERVER = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "src", "server.js");
let BASE;
let child;

const freePort = () => new Promise((resolve) => {
  const srv = net.createServer();
  srv.listen(0, "127.0.0.1", () => { const { port } = srv.address(); srv.close(() => resolve(port)); });
});
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

before(async () => {
  const port = await freePort();
  BASE = `http://127.0.0.1:${port}`;
  child = spawn(process.execPath, [SERVER], {
    env: { ...process.env, PORT: String(port), HOST: "127.0.0.1", ECHOSHIELD_RELAY_API_KEY: "test-relay-key", GEMINI_API_KEY: "", GOOGLE_API_KEY: "" },
    stdio: ["ignore", "pipe", "pipe"]
  });
  for (let i = 0; i < 50; i++) {
    try { if ((await fetch(`${BASE}/v1/time`)).ok) return; } catch (_) { /* not up yet */ }
    await sleep(100);
  }
  throw new Error("relay did not start");
});

after(() => { child?.kill(); });

const devices = [];
after(() => devices.forEach((d) => d.ctrl.abort()));

function connect(deviceId, sensor = true) {
  return new Promise((resolve, reject) => {
    const ctrl = new AbortController();
    const dev = { deviceId, events: [], ctrl, token: null };
    devices.push(dev);
    fetch(`${BASE}/v1/mesh/stream?deviceId=${deviceId}&sensor=${sensor ? 1 : 0}&platform=test`, { signal: ctrl.signal })
      .then(async (res) => {
        const reader = res.body.getReader();
        const dec = new TextDecoder();
        let buf = "";
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += dec.decode(value, { stream: true });
          let i;
          while ((i = buf.indexOf("\n\n")) >= 0) {
            const frame = buf.slice(0, i);
            buf = buf.slice(i + 2);
            const ev = /event: (.*)/.exec(frame)?.[1];
            const data = /data: (.*)/.exec(frame)?.[1];
            if (!ev) continue;
            const parsed = JSON.parse(data);
            dev.events.push({ ev, data: parsed });
            if (ev === "hello") { dev.token = parsed.busToken; resolve(dev); }
          }
        }
      })
      .catch((e) => { if (e.name !== "AbortError") reject(e); });
  });
}

const post = (p, body, token) => fetch(BASE + p, {
  method: "POST",
  headers: { "content-type": "application/json", ...(token ? { authorization: `Bus ${token}` } : {}) },
  body: JSON.stringify(body)
}).then(async (r) => ({ status: r.status, body: await r.json().catch(() => null) }));

const LAT0 = 34.0689, LON0 = -118.4452;
const M_LAT = 111320, M_LON = 111320 * Math.cos((LAT0 * Math.PI) / 180);
const toLL = (p) => ({ latitude: LAT0 + p.y / M_LAT, longitude: LON0 + p.x / M_LON });
const PHONES = [[-25, 20], [28, 25], [30, -28], [-22, -26], [0, 5]].map(([x, y], i) => ({ id: `phone-${i}`, x, y }));

let devs;
let viewer;

test("pages and read endpoints respond", async () => {
  for (const p of ["/app", "/opius", "/dispatch", "/v1/time", "/v1/mesh/status", "/v1/incidents"]) {
    const r = await fetch(BASE + p, { headers: { authorization: "Bearer test-relay-key" } });
    assert.equal(r.status, 200, p);
  }
});

test("phones join the bus and receive per-device tokens", async () => {
  devs = [];
  for (const p of PHONES) devs.push(await connect(p.id));
  viewer = await connect("viewer-0", false);
  assert.ok(devs.every((d) => d.token));
});

test("mesh writes require the device's own token", async () => {
  assert.equal((await post("/v1/mesh/detections", { deviceId: "phone-0" })).status, 401);
  assert.equal((await post("/v1/mesh/detections", { deviceId: "phone-0" }, devs[1].token)).status, 401);
});

/** Raise a shot from phone 0 at `source`, then have the others vote. */
async function fireShot({ source, t0, zones = {}, raiserRevote = false }) {
  const arrival = (p) => t0 + (Math.hypot(p.x - source.x, p.y - source.y) / 343) * 1000;
  const acoustic = (p) => {
    const d = Math.hypot(p.x - source.x, p.y - source.y);
    return { peakDbfs: Math.max(-60, -20 * Math.log10(Math.max(d, 1) / 2)), clipFraction: d < 15 ? 0.02 : 0, brightness: Math.max(0.1, 1 - d / 60) };
  };
  const raiser = PHONES[0];
  const det = await post("/v1/mesh/detections", {
    deviceId: raiser.id, confidence: 0.4, ...toLL(raiser), detectedAtMs: arrival(raiser), timingUncertaintyMs: 5,
    zone: zones[raiser.id], ...acoustic(raiser)
  }, devs[0].token);
  assert.equal(det.status, 202);
  const sessionId = det.body.sessionId;
  if (raiserRevote) {
    const r = await post("/v1/mesh/votes", { sessionId, deviceId: raiser.id, isGunshot: true, confidence: 0.9 }, devs[0].token);
    assert.equal(r.body.triggered, false, "the raiser alone must never trigger");
  }
  const results = [];
  for (let i = 1; i < PHONES.length; i++) {
    const p = PHONES[i];
    results.push(await post("/v1/mesh/votes", {
      sessionId, deviceId: p.id, isGunshot: true, confidence: 0.7, ...toLL(p), altitude: 100,
      detectedAtMs: arrival(p), timingUncertaintyMs: 5, zone: zones[p.id], ...acoustic(p)
    }, devs[i].token));
  }
  await sleep(150);
  return { sessionId, det, results };
}

const eventsFor = (dev, ev, sessionId) => dev.events.filter((e) => e.ev === ev && (!sessionId || e.data.sessionId === sessionId));

test("quorum, wake, trigger, and a TDoA fix that keeps the raiser's timing", async () => {
  const source = { x: 12, y: -8 };
  const zones = { "phone-0": "Rm 101", "phone-1": "Rm 103", "phone-2": "Rm 102", "phone-3": "Rm 102", "phone-4": "Rm 102" };
  const { sessionId, det, results } = await fireShot({ source, t0: Date.now() - 2000, zones, raiserRevote: true });

  assert.equal(det.body.sensors, 5, "viewer without a mic is not a sensor");
  assert.equal(det.body.required, 2);
  assert.ok(devs.slice(1).every((d) => eventsFor(d, "wake_classify", sessionId).length));
  assert.equal(eventsFor(devs[0], "wake_classify", sessionId).length, 0);
  assert.equal(results[0].body.triggered, true, "first independent vote reaches quorum");

  const trig = eventsFor(viewer, "response_trigger", sessionId)[0];
  assert.ok(trig, "trigger reaches non-listening viewers");
  const refined = eventsFor(viewer, "location_refined", sessionId).map((e) => e.data.estimate);
  const tdoa = refined.find((e) => e.method === "tdoa_multilateration");
  assert.ok(tdoa, "refines to a TDoA fix");
  const final = refined.at(-1);
  assert.equal(final.contributingNodes, 5, "raiser's timed vote survived its re-vote");
  const err = Math.hypot((final.longitude - LON0) * M_LON - source.x, (final.latitude - LAT0) * M_LAT - source.y);
  assert.ok(err < 2, `fix error ${err.toFixed(2)} m`);

  const timelines = viewer.events.filter((e) => e.ev === "shot_timeline");
  const last = timelines.at(-1).data;
  assert.equal(last.zoneRanking.topZone, "Rm 102", "room containing the closest phones ranks first");
  assert.equal(last.timeline.shotCount, 1);
});

test("a second phone raising the same shot merges into one shot", async () => {
  const lastEvent = () => viewer.events.filter((e) => e.ev === "shot_timeline").at(-1).data.timeline;
  const before = lastEvent();
  // Same source, with arrivals 200 ms after the first session's: one physical shot.
  const source = { x: 12, y: -8 };
  const closest = Math.min(...PHONES.map((p) => Math.hypot(p.x - source.x, p.y - source.y)));
  await fireShot({ source, t0: before.lastShotAtMs + 200 - (closest / 343) * 1000 });
  assert.equal(lastEvent().shotCount, before.shotCount);
});

test("later shots build one incident timeline with direction of movement", async () => {
  // Two more shots, 20 s apart, walking north ~25 m each time.
  const firstShotMs = viewer.events.filter((e) => e.ev === "shot_timeline").at(-1).data.timeline.firstShotAtMs;
  await fireShot({ source: { x: 12, y: 17 }, t0: firstShotMs + 20_000 });
  await fireShot({ source: { x: 12, y: 42 }, t0: firstShotMs + 40_000 });
  const last = viewer.events.filter((e) => e.ev === "shot_timeline").at(-1).data;
  assert.equal(last.timeline.shotCount, 3);
  assert.equal(last.timeline.minShooters, 1);
  assert.equal(last.timeline.movement.heading, "N");
  assert.match(last.summary, /Moving N/);

  const incidents = await fetch(`${BASE}/v1/incidents`).then((r) => r.json());
  const mesh = (incidents.incidents || incidents).filter((i) => String(i.id).startsWith("incident-mesh-"));
  assert.equal(mesh.length, 1, "all shots belong to one incident");

  const latest = await fetch(`${BASE}/v1/incidents/latest`).then((r) => r.json());
  assert.equal(latest.incident.timeline.shotCount, 3);
  assert.match(latest.incident.timelineSummary, /3 shots/);
  assert.match(latest.incident.policeBrief, /3 shots/);
  assert.equal(latest.incident.shots, undefined, "raw shot records stay server-side");

  const page = await fetch(`${BASE}/dispatch`, { headers: { authorization: "Bearer test-relay-key" } }).then((r) => r.text());
  assert.match(page, /Shot timeline/);
  assert.match(page, /ACTIVE FIRE/);
});

test("votes after the window closes are rejected", async () => {
  const { sessionId } = await fireShot({ source: { x: 0, y: 0 }, t0: Date.now() });
  await sleep(6200);
  const r = await post("/v1/mesh/votes", { sessionId, deviceId: "phone-1", isGunshot: true }, devs[1].token);
  assert.equal(r.status, 409);
});
