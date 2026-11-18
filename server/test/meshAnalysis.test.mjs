import { test } from "node:test";
import assert from "node:assert/strict";
import {
  mergeVote,
  rankZones,
  triangulateTdoa,
  buildShotTimeline,
  describeTimeline,
  isSameShot
} from "../src/meshAnalysis.js";

const LAT0 = 34.0689;
const LON0 = -118.4452;
const M_LAT = 111320;
const M_LON = 111320 * Math.cos((LAT0 * Math.PI) / 180);
const at = (x, y) => ({ latitude: LAT0 + y / M_LAT, longitude: LON0 + x / M_LON });

test("mergeVote keeps earlier timing when a re-vote omits it", () => {
  const first = { deviceId: "a", isGunshot: true, detectedAtMs: 1000, timingUncertaintyMs: 5, zone: "204" };
  const revote = { deviceId: "a", isGunshot: true, confidence: 0.9, detectedAtMs: NaN, timingUncertaintyMs: NaN, zone: undefined };
  const merged = mergeVote(first, revote);
  assert.equal(merged.detectedAtMs, 1000);
  assert.equal(merged.timingUncertaintyMs, 5);
  assert.equal(merged.zone, "204");
  assert.equal(merged.confidence, 0.9);
});

test("mergeVote does not pair a new timestamp with the old uncertainty", () => {
  const merged = mergeVote({ detectedAtMs: 1000, timingUncertaintyMs: 5 }, { detectedAtMs: 2000 });
  assert.equal(merged.detectedAtMs, 2000);
  assert.ok(Number.isNaN(merged.timingUncertaintyMs));
});

// A shot in room 204: its phones clip hard and hear it first; the neighbours behind
// a wall are ~25 dB quieter, unclipped, duller and later; a far room barely hears it.
function classroomVotes({ sourceRoom = "204", t0 = 10_000 } = {}) {
  const rooms = {
    "204": { db: -1, clip: 0.03, bright: 0.9, delay: 0, heard: 1 },
    "206": { db: -26, clip: 0, bright: 0.35, delay: 18, heard: 1 },
    "210": { db: -40, clip: 0, bright: 0.2, delay: 45, heard: 0.5 }
  };
  const names = Object.keys(rooms);
  // Swap profiles so `sourceRoom` gets the loud one.
  if (sourceRoom !== "204") {
    const tmp = rooms[sourceRoom];
    rooms[sourceRoom] = rooms["204"];
    rooms["204"] = tmp;
  }
  const votes = [];
  for (const zone of names) {
    const p = rooms[zone];
    for (let i = 0; i < 30; i++) {
      // Some phones are in bags: quieter and duller than their room's median.
      const muffled = i % 5 === 0;
      const heard = i / 30 < p.heard;
      votes.push({
        deviceId: `${zone}-${i}`,
        zone,
        isGunshot: heard,
        peakDbfs: heard ? p.db - (muffled ? 15 : Math.random() * 3) : NaN,
        clipFraction: heard ? (muffled ? 0 : p.clip) : NaN,
        brightness: heard ? p.bright * (muffled ? 0.5 : 1) : NaN,
        detectedAtMs: heard ? t0 + p.delay + Math.random() * 8 : NaN,
        timingUncertaintyMs: heard ? 10 : NaN
      });
    }
  }
  return votes;
}

test("rankZones picks the source room with high confidence at classroom density", () => {
  const ranking = rankZones(classroomVotes());
  assert.equal(ranking.topZone, "204");
  assert.equal(ranking.confidence, "high");
  assert.deepEqual(ranking.zones.map((z) => z.zone), ["204", "206", "210"]);
});

test("rankZones follows the acoustics, not the room name", () => {
  const votes = classroomVotes({ sourceRoom: "206" });
  const ranking = rankZones(votes);
  assert.equal(ranking.topZone, "206");
});

test("rankZones ignores votes without a zone and returns null when none have one", () => {
  assert.equal(rankZones([{ isGunshot: true, peakDbfs: -1 }]), null);
});

test("rankZones with one room reports it but without high confidence", () => {
  const ranking = rankZones([
    { zone: "Library", isGunshot: true, peakDbfs: -3 },
    { zone: "Library", isGunshot: true, peakDbfs: -5 }
  ]);
  assert.equal(ranking.topZone, "Library");
  assert.notEqual(ranking.confidence, "high");
});

test("triangulateTdoa recovers the source with perfect clocks", () => {
  const source = { x: 12, y: -8 };
  const phones = [[-25, 20], [28, 25], [30, -28], [-22, -26], [0, 5]];
  const votes = phones.map(([x, y]) => ({
    ...at(x, y),
    detectedAtMs: 1_000_000 + (Math.hypot(x - source.x, y - source.y) / 343) * 1000,
    timingUncertaintyMs: 2
  }));
  const fix = triangulateTdoa(votes);
  assert.equal(fix.method, "tdoa_multilateration");
  const ex = (fix.longitude - LON0) * M_LON;
  const ey = (fix.latitude - LAT0) * M_LAT;
  assert.ok(Math.hypot(ex - source.x, ey - source.y) < 1);
});

test("triangulateTdoa refuses with fewer than three timed phones", () => {
  assert.equal(triangulateTdoa([{ ...at(0, 0), detectedAtMs: 1, timingUncertaintyMs: 1 }]), null);
});

const shot = (id, tSec, x, y, zone, method = "tdoa_multilateration") => ({
  shotId: id,
  atMs: 1_700_000_000_000 + tSec * 1000,
  estimate: { ...at(x, y), method, timingResidualMs: 3, spreadMeters: 20 },
  zoneRanking: zone ? { topZone: zone, confidence: "high" } : null,
  confirmedBy: ["a", "b", "c"]
});

test("timeline reports count, pace, direction and room sequence for one moving shooter", () => {
  // Walking north along a corridor at ~1.5 m/s, a shot every 20 s.
  const shots = [shot("s1", 0, 0, 0, "Rm 204"), shot("s2", 20, 0, 30, "Hall B"), shot("s3", 40, 0, 60, "Rm 210")];
  const now = shots[2].atMs + 5000;
  const t = buildShotTimeline(shots, now);
  assert.equal(t.shotCount, 3);
  assert.equal(t.status, "active_fire");
  assert.equal(t.secondsSinceLastShot, 5);
  assert.equal(t.shotsPerMinute, 3);
  assert.equal(t.movement.heading, "N");
  assert.deepEqual(t.zonePath, ["Rm 204", "Hall B", "Rm 210"]);
  assert.equal(t.minShooters, 1);
  assert.equal(t.latest.zone, "Rm 210");
  const text = describeTimeline(t);
  assert.match(text, /Rm 210/);
  assert.match(text, /Moving N/);
  assert.match(text, /single shooter/);
});

test("timeline requires two shooters when shots are too far apart for one person", () => {
  // 300 m apart, 3 s apart: nobody covers that on foot.
  const shots = [shot("s1", 0, 0, 0, "Gym"), shot("s2", 3, 300, 0, "Library")];
  const t = buildShotTimeline(shots, shots[1].atMs);
  assert.equal(t.minShooters, 2);
  assert.match(describeTimeline(t), /at least 2 shooters/);
});

test("timeline reports a lull after a minute without shots", () => {
  const shots = [shot("s1", 0, 0, 0, "Rm 204")];
  const t = buildShotTimeline(shots, shots[0].atMs + 90_000);
  assert.equal(t.status, "lull");
});

test("small position changes inside the error bound are not reported as movement", () => {
  const shots = [shot("s1", 0, 0, 0), shot("s2", 30, 3, 2)];
  const t = buildShotTimeline(shots, shots[1].atMs);
  assert.equal(t.movement.stationary, true);
  assert.equal(t.steps[0].significant, false);
});

test("isSameShot merges near-simultaneous nearby detections but not distant ones", () => {
  const a = shot("s1", 0, 0, 0);
  assert.equal(isSameShot(a, shot("x", 0.4, 5, 5)), true);
  assert.equal(isSameShot(a, shot("x", 0.4, 400, 0)), false);
  assert.equal(isSameShot(a, shot("x", 5, 0, 0)), false);
});

test("isSameShot keeps confidently different rooms apart when positions are unknown", () => {
  const a = { atMs: 0, zoneRanking: { topZone: "Gym", confidence: "high" } };
  const b = { atMs: 300, zoneRanking: { topZone: "Library", confidence: "high" } };
  assert.equal(isSameShot(a, b), false);
  assert.equal(isSameShot(a, { atMs: 300, zoneRanking: { topZone: "Gym", confidence: "high" } }), true);
});

test("triangulateTdoa rejects a fix far outside the listeners' footprint", () => {
  // A tight cluster of phones hearing a distant shot sees an almost flat wavefront:
  // the direction is known but the range is not, so the solve must not invent one.
  const source = { x: 120, y: 0 };
  const phones = [[-5, -5], [5, -5], [5, 5], [-5, 5]];
  const votes = phones.map(([x, y]) => ({
    ...at(x, y),
    detectedAtMs: 1_000_000 + (Math.hypot(x - source.x, y - source.y) / 343) * 1000 + (x === 5 && y === 5 ? 3 : 0),
    timingUncertaintyMs: 5
  }));
  assert.equal(triangulateTdoa(votes), null);
});
