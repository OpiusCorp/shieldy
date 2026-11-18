// Monte Carlo simulation of the two localization paths. Prints the tables quoted in
// docs/GUNSHOT_ACOUSTICS_INDOORS.md. Run with `npm run simulate`.
//
// This models the physics coarsely and is meant to show how accuracy scales with
// clock error and phone density — not to stand in for recordings of real shots.
import { triangulateTdoa, rankZones } from "../src/meshAnalysis.js";

let seed = 42;
const rand = () => ((seed = (seed * 1664525 + 1013904223) % 4294967296) / 4294967296);
const gauss = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());

const LAT0 = 34.0689, LON0 = -118.4452;
const M_LAT = 111320, M_LON = 111320 * Math.cos((LAT0 * Math.PI) / 180);
const toLL = (x, y) => ({ latitude: LAT0 + y / M_LAT, longitude: LON0 + x / M_LON });

// ── 1. Arrival-time multilateration vs clock error ──────────────────────────────
// Five phones spread across a 60 × 60 m floor plate, perfect positions, Gaussian
// clock error σ on each arrival, each phone honestly reporting σ as its uncertainty.
// A second run adds 5 m of position error, typical of good outdoor GPS.
function tdoaTrial(sigmaMs, positionErrorM) {
  const source = { x: (rand() - 0.5) * 40, y: (rand() - 0.5) * 40 };
  const phones = Array.from({ length: 5 }, () => ({ x: (rand() - 0.5) * 60, y: (rand() - 0.5) * 60 }));
  const votes = phones.map((p) => {
    const t = (Math.hypot(p.x - source.x, p.y - source.y) / 343) * 1000 + gauss() * sigmaMs;
    return {
      ...toLL(p.x + gauss() * positionErrorM, p.y + gauss() * positionErrorM),
      detectedAtMs: 1_000_000 + t,
      timingUncertaintyMs: Math.max(1, sigmaMs)
    };
  });
  const fix = triangulateTdoa(votes);
  if (!fix) return null;
  return Math.hypot((fix.longitude - LON0) * M_LON - source.x, (fix.latitude - LAT0) * M_LAT - source.y);
}

function tdoaTable(positionErrorM, trials = 200) {
  const rows = [];
  for (const sigma of [0, 5, 20, 50]) {
    const errors = [];
    for (let i = 0; i < trials; i++) {
      const e = tdoaTrial(sigma, positionErrorM);
      if (e !== null) errors.push(e);
    }
    errors.sort((a, b) => a - b);
    const med = errors.length ? errors[Math.floor(errors.length / 2)].toFixed(1) + " m" : "—";
    const p90 = errors.length ? errors[Math.floor(errors.length * 0.9)].toFixed(1) + " m" : "—";
    rows.push(`| ±${sigma} ms | ${Math.round((errors.length / trials) * 100)}% | ${med} | ${p90} |`);
  }
  return rows;
}

// ── 2. Room ranking vs phone density ────────────────────────────────────────────
// Rooms along a corridor, 9 m apart. Level falls with distance and loses 20–40 dB per
// wall crossed; each phone adds pocket/bag noise. A phone clips above −3 dBFS. Walls
// also strip high frequencies. Clocks carry ±20 ms of error. Each room holds 30
// people, of whom only some have listening mode on.
function roomTrial(listeningPerRoom, wallRange, roomCount = 6) {
  const sourceRoom = Math.floor(rand() * roomCount);
  const wallLoss = Array.from({ length: roomCount }, () => wallRange[0] + rand() * (wallRange[1] - wallRange[0]));
  const votes = [];
  for (let r = 0; r < roomCount; r++) {
    const walls = Math.abs(r - sourceRoom);
    let loss = 0;
    for (let w = Math.min(r, sourceRoom); w < Math.max(r, sourceRoom); w++) loss += wallLoss[w];
    for (let i = 0; i < listeningPerRoom; i++) {
      const distance = Math.max(2, walls * 9 + rand() * 8);
      const pocket = rand() < 0.4 ? 10 + rand() * 15 : 0;
      // ~+20 dBFS equivalent at 2 m for an unattenuated muzzle blast (clips hard).
      const level = 20 - 20 * Math.log10(distance / 2) - loss - pocket + gauss() * 4;
      const heard = level > -50;
      votes.push({
        deviceId: `${r}-${i}`,
        zone: `Room ${r}`,
        isGunshot: heard,
        peakDbfs: heard ? Math.min(0, level) : NaN,
        clipFraction: heard ? (level > -3 ? Math.min(0.05, (level + 3) / 400) : 0) : NaN,
        brightness: heard ? Math.max(0.05, 0.9 * Math.pow(0.5, walls) - pocket / 100 + gauss() * 0.05) : NaN,
        detectedAtMs: heard ? 1_000_000 + (distance / 343) * 1000 + walls * 5 + gauss() * 20 : NaN,
        timingUncertaintyMs: heard ? 20 : NaN
      });
    }
  }
  const ranking = rankZones(votes);
  return { correct: ranking.topZone === `Room ${sourceRoom}`, confidence: ranking.confidence };
}

function roomTable(wallRange, trials = 500) {
  const rows = [];
  for (const perRoom of [1, 2, 3, 5, 10]) {
    let correct = 0, high = 0, highCorrect = 0;
    for (let i = 0; i < trials; i++) {
      const r = roomTrial(perRoom, wallRange);
      if (r.correct) correct++;
      if (r.confidence === "high") { high++; if (r.correct) highCorrect++; }
    }
    const pct = (n, d) => (d ? Math.round((n / d) * 100) + "%" : "—");
    rows.push(`| ${perRoom} | ${Math.round((perRoom / 30) * 100)}% | ${pct(correct, trials)} | ${pct(high, trials)} | ${pct(highCorrect, high)} |`);
  }
  return rows;
}

console.log("Arrival-time fix, 5 phones, exact positions (indoor upper bound)\n");
console.log("| Clock error (σ) | Fix returned | Median error | 90th percentile |\n| --- | --- | --- | --- |");
console.log(tdoaTable(0).join("\n"));
console.log("\nSame, with 5 m position error per phone (good outdoor GPS)\n");
console.log("| Clock error (σ) | Fix returned | Median error | 90th percentile |\n| --- | --- | --- | --- |");
console.log(tdoaTable(5).join("\n"));
const roomHeader = "| Listening phones per room | Share of room | Top room correct | Rated high confidence | Correct when high |\n| --- | --- | --- | --- | --- |";
console.log("\nRoom ranking, 6 rooms on a corridor, 30 people per room, doors closed (20–40 dB per wall)\n");
console.log(roomHeader);
console.log(roomTable([20, 40]).join("\n"));
console.log("\nSame, doors open (6–15 dB per wall)\n");
console.log(roomHeader);
console.log(roomTable([6, 15]).join("\n"));
