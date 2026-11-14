// Pure analysis for the relay: localization, room ranking, and the shot timeline.
//
// Nothing in here touches the network or server state, so it can be unit-tested and
// driven by the simulation in `test/` directly. See docs/GUNSHOT_ACOUSTICS_INDOORS.md
// for the physics these choices rest on.

export const SPEED_OF_SOUND_MPS = 343;
export const MAX_TIMING_UNCERTAINTY_MS = 60;
export const MAX_ARRIVAL_SPREAD_MS = 1500;
export const MIN_BASELINE_METERS = 4;
/** Fixes further than this multiple of the listeners' radius from their centre are rejected. */
export const MAX_FIX_RADIUS_FACTOR = 1.5;

/** Detections closer together than this, and near each other, are one shot. */
export const SHOT_MERGE_MS = 1500;
/** Fastest a person plausibly moves between shots (a sprint), for shooter counting. */
export const MAX_SHOOTER_SPEED_MPS = 7;
/** A shot this recent means fire is ongoing; older than this is reported as a lull. */
export const ACTIVE_FIRE_WINDOW_MS = 60_000;

const M_PER_DEG_LAT = 111_320;

export function haversineMeters(lat1, lon1, lat2, lon2) {
  const R = 6371000;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((lat1 * Math.PI) / 180) * Math.cos((lat2 * Math.PI) / 180) * Math.sin(dLon / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

/** Initial compass bearing from point 1 to point 2, degrees clockwise from north. */
export function bearingDegrees(lat1, lon1, lat2, lon2) {
  const φ1 = (lat1 * Math.PI) / 180;
  const φ2 = (lat2 * Math.PI) / 180;
  const Δλ = ((lon2 - lon1) * Math.PI) / 180;
  const y = Math.sin(Δλ) * Math.cos(φ2);
  const x = Math.cos(φ1) * Math.sin(φ2) - Math.sin(φ1) * Math.cos(φ2) * Math.cos(Δλ);
  return ((Math.atan2(y, x) * 180) / Math.PI + 360) % 360;
}

const CARDINALS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
export function cardinal(bearing) {
  return CARDINALS[Math.round(bearing / 45) % 8];
}

const finite = (value) => typeof value === "number" && Number.isFinite(value);

/**
 * Fold a device's new vote into its previous one for the same session.
 *
 * A later vote must not erase what an earlier one established: a re-vote that omits
 * its arrival time would otherwise drop that device out of the triangulation. Fields
 * the new vote actually carries win; missing ones keep their previous value.
 */
export function mergeVote(previous, incoming) {
  if (!previous) return { ...incoming };
  const merged = { ...previous };
  for (const [key, value] of Object.entries(incoming)) {
    if (value === undefined || value === null) continue;
    if (typeof value === "number" && !Number.isFinite(value)) continue;
    if (key === "zone" && !String(value).trim()) continue;
    merged[key] = value;
  }
  // Timing travels as a pair: a new timestamp without its uncertainty is unusable,
  // and must not be paired with the old one's uncertainty.
  if (finite(incoming.detectedAtMs) && !finite(incoming.timingUncertaintyMs)) {
    merged.timingUncertaintyMs = NaN;
  }
  return merged;
}

/**
 * TDoA multilateration by grid search. Given confirming devices with positions and
 * arrival times on a shared clock, the true source is the point that makes
 * `arrival_i - distance_i / c` constant across devices. Returns null when the inputs
 * cannot support an honest fix.
 */
export function triangulateTdoa(located) {
  const timed = located.filter(
    (v) =>
      finite(v.detectedAtMs) &&
      finite(v.timingUncertaintyMs) &&
      v.timingUncertaintyMs <= MAX_TIMING_UNCERTAINTY_MS
  );
  if (timed.length < 3) return null;

  const arrivals = timed.map((v) => v.detectedAtMs);
  if (Math.max(...arrivals) - Math.min(...arrivals) > MAX_ARRIVAL_SPREAD_MS) return null;

  const lat0 = timed.reduce((s, v) => s + v.latitude, 0) / timed.length;
  const lon0 = timed.reduce((s, v) => s + v.longitude, 0) / timed.length;
  const mPerDegLon = M_PER_DEG_LAT * Math.cos((lat0 * Math.PI) / 180);
  const pts = timed.map((v) => ({
    x: (v.longitude - lon0) * mPerDegLon,
    y: (v.latitude - lat0) * M_PER_DEG_LAT,
    t: v.detectedAtMs / 1000
  }));

  const baseline = Math.max(
    ...pts.map((a) => Math.max(...pts.map((b) => Math.hypot(a.x - b.x, a.y - b.y))))
  );
  if (baseline < MIN_BASELINE_METERS) return null;

  const sweep = (cx, cy, range, step) => {
    let best = null;
    for (let gx = cx - range; gx <= cx + range; gx += step) {
      for (let gy = cy - range; gy <= cy + range; gy += step) {
        const offsets = pts.map((p) => p.t - Math.hypot(p.x - gx, p.y - gy) / SPEED_OF_SOUND_MPS);
        const mean = offsets.reduce((s, o) => s + o, 0) / offsets.length;
        const variance = offsets.reduce((s, o) => s + (o - mean) ** 2, 0) / offsets.length;
        if (!best || variance < best.variance) best = { gx, gy, variance };
      }
    }
    return best;
  };

  let best = sweep(0, 0, 150, 5);
  best = sweep(best.gx, best.gy, 5, 0.5) || best;

  const residualMs = Math.sqrt(best.variance) * 1000;
  const tolerance = Math.max(10, Math.max(...timed.map((v) => v.timingUncertaintyMs)) * 2);
  if (residualMs > tolerance) return null;

  // A source far outside the phones that heard it is almost always the solve sliding
  // along a poorly constrained direction to the edge of the grid, not a real fix —
  // the residual stays small because, along that line, every point fits about as
  // well. Phones that all heard one shot surround it or sit near it, so reject fixes
  // beyond the listeners' own footprint.
  const radius = Math.max(...pts.map((p) => Math.hypot(p.x, p.y)));
  if (Math.hypot(best.gx, best.gy) > Math.max(radius * MAX_FIX_RADIUS_FACTOR, radius + 15)) return null;

  return {
    latitude: lat0 + best.gy / M_PER_DEG_LAT,
    longitude: lon0 + best.gx / mPerDegLon,
    method: "tdoa_multilateration",
    timingResidualMs: Number(residualMs.toFixed(1))
  };
}

const median = (values) => {
  const sorted = values.filter(finite).sort((a, b) => a - b);
  if (!sorted.length) return NaN;
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

/**
 * Rank rooms (zones) by how likely each is to contain the shot.
 *
 * Indoors, arrival-time multilateration fails — phone positions are only known to
 * the access point, and sound reaches other rooms through doors rather than in
 * straight lines. What survives is a comparison between *groups* of phones: a wall
 * costs 20–40 dB, so the source room's phones are louder, clip harder, hear it first,
 * and keep more high-frequency energy. With many phones per room each feature is a
 * median over the group, so one phone in a pocket cannot swing the result.
 *
 * Every vote that names a zone counts, including "did not hear it" votes: a room
 * full of phones that heard nothing is evidence the shot was elsewhere.
 */
const FEATURE_WEIGHTS = { loudness: 0.3, clipping: 0.3, arrival: 0.25, brightness: 0.15 };

export function rankZones(votes) {
  const byZone = new Map();
  for (const vote of votes) {
    const zone = String(vote.zone || "").trim();
    if (!zone) continue;
    if (!byZone.has(zone)) byZone.set(zone, []);
    byZone.get(zone).push(vote);
  }
  if (byZone.size === 0) return null;

  const zones = [...byZone.entries()].map(([zone, list]) => {
    const heard = list.filter((v) => v.isGunshot);
    const timed = heard.filter(
      (v) => finite(v.detectedAtMs) && finite(v.timingUncertaintyMs) &&
        v.timingUncertaintyMs <= MAX_TIMING_UNCERTAINTY_MS
    );
    // Phones that heard nothing contribute silence, not a missing value.
    const levelOf = (v) => (v.isGunshot ? v.peakDbfs : -90);
    return {
      zone,
      phones: list.length,
      heard: heard.length,
      peakDbfs: median(list.map(levelOf)),
      clipFraction: median(list.map((v) => (v.isGunshot ? v.clipFraction ?? NaN : 0))),
      brightness: median(heard.map((v) => v.brightness)),
      earliestArrivalMs: timed.length ? Math.min(...timed.map((v) => v.detectedAtMs)) : NaN,
      arrivalUncertaintyMs: timed.length ? Math.min(...timed.map((v) => v.timingUncertaintyMs)) : NaN
    };
  });

  // Normalise each feature across zones to 0..1 (1 = most source-like). A feature is
  // only used when at least two zones report it, since a lone value ranks nothing.
  const normalise = (key, higherIsCloser, minSpread) => {
    const values = zones.map((z) => z[key]).filter(finite);
    if (values.length < 2) return null;
    const lo = Math.min(...values);
    const hi = Math.max(...values);
    const span = Math.max(hi - lo, minSpread);
    return (z) => {
      if (!finite(z[key])) return null;
      return higherIsCloser ? (z[key] - lo) / span : (hi - z[key]) / span;
    };
  };

  // An arrival lead smaller than the clocks' own uncertainty is not a lead.
  const worstClock = Math.max(0, ...zones.map((z) => z.arrivalUncertaintyMs).filter(finite));
  const scorers = {
    loudness: normalise("peakDbfs", true, 3),
    clipping: normalise("clipFraction", true, 0.002),
    arrival: normalise("earliestArrivalMs", false, Math.max(2 * worstClock, 5)),
    brightness: normalise("brightness", true, 0.05)
  };

  for (const z of zones) {
    let weighted = 0;
    let weightSum = 0;
    const evidence = {};
    for (const [feature, weight] of Object.entries(FEATURE_WEIGHTS)) {
      const score = scorers[feature] ? scorers[feature](z) : null;
      if (score === null) continue;
      evidence[feature] = Number(score.toFixed(2));
      weighted += weight * score;
      weightSum += weight;
    }
    // With no comparable feature at all, fall back to the share of phones that heard it.
    z.score = weightSum > 0 ? weighted / weightSum : z.heard / Math.max(1, z.phones);
    z.score = Number(z.score.toFixed(3));
    z.evidence = evidence;
  }

  zones.sort((a, b) => b.score - a.score || b.heard - a.heard);
  const top = zones[0];
  const margin = zones.length > 1 ? top.score - zones[1].score : 1;
  let confidence = "low";
  if (zones.length === 1) {
    confidence = top.heard >= 3 ? "medium" : "low";
  } else if (margin >= 0.25 && top.heard >= 3) {
    confidence = "high";
  } else if (margin >= 0.1 && top.heard >= 2) {
    confidence = "medium";
  }

  return {
    topZone: top.zone,
    confidence,
    margin: Number(margin.toFixed(3)),
    zones: zones.map((z) => ({
      zone: z.zone,
      score: z.score,
      phones: z.phones,
      heard: z.heard,
      peakDbfs: finite(z.peakDbfs) ? Number(z.peakDbfs.toFixed(1)) : null,
      clipFraction: finite(z.clipFraction) ? Number(z.clipFraction.toFixed(4)) : null,
      evidence: z.evidence
    }))
  };
}

/**
 * How far off a shot's position could plausibly be, in metres. An arrival-time fix
 * is bounded by its timing residual; a centroid only says "inside this cluster".
 */
export function shotUncertaintyMeters(estimate) {
  if (!estimate) return Infinity;
  if (estimate.method === "tdoa_multilateration") {
    const fromResidual = (estimate.timingResidualMs || 0) * (SPEED_OF_SOUND_MPS / 1000) * 2;
    return Math.max(5, fromResidual);
  }
  return Math.max(15, estimate.spreadMeters || 0);
}

const hasPosition = (shot) =>
  shot && shot.estimate && finite(shot.estimate.latitude) && finite(shot.estimate.longitude);

const distanceBetween = (a, b) =>
  haversineMeters(a.estimate.latitude, a.estimate.longitude, b.estimate.latitude, b.estimate.longitude);

/**
 * Whether a new detection is the same physical shot as an existing one. One shot is
 * heard by many phones, and several of them may each raise their own session, so
 * detections close in time are merged — unless they are too far apart for one gun,
 * which is exactly the two-shooter case and must stay separate.
 */
export function isSameShot(shot, candidate) {
  if (Math.abs(candidate.atMs - shot.atMs) > SHOT_MERGE_MS) return false;
  if (!hasPosition(shot) || !hasPosition(candidate)) {
    // Without positions, two confident top rooms that disagree are two sources.
    const a = shot.zoneRanking;
    const b = candidate.zoneRanking;
    if (a && b && a.confidence !== "low" && b.confidence !== "low" && a.topZone !== b.topZone) {
      return false;
    }
    return true;
  }
  const slack = shotUncertaintyMeters(shot.estimate) + shotUncertaintyMeters(candidate.estimate);
  return distanceBetween(shot, candidate) <= Math.max(30, slack);
}

/**
 * Turn the list of confirmed shots into what responders asked for: when it started,
 * how many shots and how fast, where the latest one was, which way it is moving, and
 * how many shooters the pattern requires.
 */
export function buildShotTimeline(shots, nowMs = Date.now()) {
  const ordered = [...(shots || [])].sort((a, b) => a.atMs - b.atMs);
  if (!ordered.length) return null;

  const first = ordered[0];
  const last = ordered[ordered.length - 1];
  const durationMs = last.atMs - first.atMs;
  const intervalsMs = ordered.slice(1).map((s, i) => s.atMs - ordered[i].atMs);

  // Movement between consecutive shots that both have positions. A step shorter than
  // the two positions' combined uncertainty is reported as "same area", not movement.
  const steps = [];
  for (let i = 1; i < ordered.length; i++) {
    const a = ordered[i - 1];
    const b = ordered[i];
    const fromZone = a.zoneRanking?.topZone || null;
    const toZone = b.zoneRanking?.topZone || null;
    const step = {
      fromShotId: a.shotId,
      toShotId: b.shotId,
      elapsedMs: b.atMs - a.atMs,
      fromZone,
      toZone,
      zoneChanged: Boolean(fromZone && toZone && fromZone !== toZone)
    };
    if (hasPosition(a) && hasPosition(b)) {
      const distance = distanceBetween(a, b);
      const uncertainty = shotUncertaintyMeters(a.estimate) + shotUncertaintyMeters(b.estimate);
      step.distanceMeters = Number(distance.toFixed(1));
      step.significant = distance > uncertainty;
      if (step.significant) {
        const bearing = bearingDegrees(a.estimate.latitude, a.estimate.longitude, b.estimate.latitude, b.estimate.longitude);
        step.bearingDegrees = Math.round(bearing);
        step.heading = cardinal(bearing);
        step.speedMps = step.elapsedMs > 0 ? Number((distance / (step.elapsedMs / 1000)).toFixed(2)) : null;
      }
    }
    steps.push(step);
  }

  // Overall direction, from the first to the latest located shot.
  const located = ordered.filter(hasPosition);
  let movement = null;
  if (located.length >= 2) {
    const a = located[0];
    const b = located[located.length - 1];
    const distance = distanceBetween(a, b);
    const uncertainty = shotUncertaintyMeters(a.estimate) + shotUncertaintyMeters(b.estimate);
    if (distance > uncertainty) {
      const bearing = bearingDegrees(a.estimate.latitude, a.estimate.longitude, b.estimate.latitude, b.estimate.longitude);
      movement = {
        heading: cardinal(bearing),
        bearingDegrees: Math.round(bearing),
        distanceMeters: Number(distance.toFixed(1)),
        overMs: b.atMs - a.atMs
      };
    } else {
      movement = { heading: null, distanceMeters: Number(distance.toFixed(1)), overMs: b.atMs - a.atMs, stationary: true };
    }
  }

  // Zone path: the sequence of distinct top rooms, for indoor movement.
  const zonePath = [];
  for (const shot of ordered) {
    const zone = shot.zoneRanking?.topZone;
    if (zone && zonePath[zonePath.length - 1] !== zone) zonePath.push(zone);
  }

  // Minimum shooter count: greedily assign each shot to a track one person could
  // have reached in time; a shot no track can reach needs another shooter. Greedy
  // assignment can only over-count when tracks overlap, so this is a lower bound in
  // practice and is labelled "at least".
  const tracks = [];
  for (const shot of ordered) {
    let assigned = false;
    if (hasPosition(shot)) {
      for (const track of tracks) {
        const prev = track[track.length - 1];
        if (!hasPosition(prev)) continue;
        const reach = MAX_SHOOTER_SPEED_MPS * ((shot.atMs - prev.atMs) / 1000);
        const slack = shotUncertaintyMeters(prev.estimate) + shotUncertaintyMeters(shot.estimate);
        if (distanceBetween(prev, shot) <= reach + slack) {
          track.push(shot);
          assigned = true;
          break;
        }
      }
    } else if (tracks.length) {
      tracks[tracks.length - 1].push(shot);
      assigned = true;
    }
    if (!assigned) tracks.push([shot]);
  }

  const sinceLastMs = nowMs - last.atMs;
  return {
    shotCount: ordered.length,
    firstShotAtMs: first.atMs,
    lastShotAtMs: last.atMs,
    durationMs,
    secondsSinceLastShot: Math.max(0, Math.round(sinceLastMs / 1000)),
    status: sinceLastMs <= ACTIVE_FIRE_WINDOW_MS ? "active_fire" : "lull",
    shotsPerMinute: ordered.length >= 2 && durationMs > 0
      ? Number(((ordered.length - 1) / (durationMs / 60000)).toFixed(1))
      : null,
    meanIntervalMs: intervalsMs.length
      ? Math.round(intervalsMs.reduce((s, v) => s + v, 0) / intervalsMs.length)
      : null,
    latest: {
      shotId: last.shotId,
      atMs: last.atMs,
      zone: last.zoneRanking?.topZone || null,
      zoneConfidence: last.zoneRanking?.confidence || null,
      estimate: last.estimate || null
    },
    movement,
    zonePath,
    steps,
    minShooters: Math.max(1, tracks.length),
    shots: ordered.map((s) => ({
      shotId: s.shotId,
      atMs: s.atMs,
      zone: s.zoneRanking?.topZone || null,
      zoneConfidence: s.zoneRanking?.confidence || null,
      method: s.estimate?.method || null,
      latitude: s.estimate?.latitude ?? null,
      longitude: s.estimate?.longitude ?? null,
      confirmedBy: s.confirmedBy?.length || 0
    }))
  };
}

// Server-local wall-clock time (set TZ for the deployment), matching the dispatch table.
const clockTime = (ms) =>
  new Date(ms).toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", second: "2-digit" });

/** One-paragraph responder summary of a timeline, most actionable fact first. */
export function describeTimeline(timeline) {
  if (!timeline) return "";
  const parts = [];
  const latest = timeline.latest;
  const where = latest.zone
    ? `${latest.zone} (${latest.zoneConfidence} confidence)`
    : latest.estimate
      ? `${latest.estimate.latitude.toFixed(5)}, ${latest.estimate.longitude.toFixed(5)}`
      : "location unresolved";
  parts.push(
    `Latest shot ${timeline.secondsSinceLastShot}s ago at ${where}` +
      (timeline.status === "lull" ? " — no shots in the last minute." : ".")
  );
  parts.push(
    `${timeline.shotCount} shot${timeline.shotCount === 1 ? "" : "s"} since ${clockTime(timeline.firstShotAtMs)}` +
      (timeline.shotsPerMinute ? ` (${timeline.shotsPerMinute}/min).` : ".")
  );
  if (timeline.zonePath.length >= 2) {
    parts.push(`Room sequence: ${timeline.zonePath.join(" → ")}.`);
  }
  if (timeline.movement && timeline.movement.heading) {
    parts.push(`Moving ${timeline.movement.heading}, about ${Math.round(timeline.movement.distanceMeters)} m since the first located shot.`);
  } else if (timeline.movement && timeline.movement.stationary) {
    parts.push("No significant movement between located shots.");
  }
  parts.push(
    timeline.minShooters > 1
      ? `Shot pattern requires at least ${timeline.minShooters} shooters.`
      : "Pattern consistent with a single shooter."
  );
  return parts.join(" ");
}
