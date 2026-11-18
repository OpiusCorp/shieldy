import { test, describe, before, after } from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
import net from "node:net";

const SERVER = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "src", "server.js");

const freePort = () => new Promise((resolve) => {
  const srv = net.createServer();
  srv.listen(0, "127.0.0.1", () => {
    const { port } = srv.address();
    srv.close(() => resolve(port));
  });
});

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

describe("Responder Console Security & Field Reports (Authenticated Mode)", () => {
  let port;
  let base;
  let child;
  const API_KEY = "test-secret-key-12345";

  before(async () => {
    port = await freePort();
    base = `http://127.0.0.1:${port}`;
    child = spawn(process.execPath, [SERVER], {
      env: {
        ...process.env,
        PORT: String(port),
        HOST: "127.0.0.1",
        ECHOSHIELD_RELAY_API_KEY: API_KEY,
        GEMINI_API_KEY: "",
        GOOGLE_API_KEY: ""
      },
      stdio: ["ignore", "pipe", "pipe"]
    });

    for (let i = 0; i < 50; i++) {
      try {
        const res = await fetch(`${base}/v1/time`);
        if (res.ok) return;
      } catch (_) {}
      await sleep(100);
    }
    throw new Error("Relay server did not start in time");
  });

  after(() => {
    child?.kill();
  });

  test("unauthenticated GET /dashboard and /dispatch return 401 login page without leaking API key", async () => {
    for (const path of ["/dashboard", "/dispatch"]) {
      const res = await fetch(`${base}${path}`);
      assert.equal(res.status, 401);
      const text = await res.text();
      assert.match(text, /EchoShield Console/);
      assert.match(text, /action="\/login"/);
      assert.ok(!text.includes(API_KEY), `Page ${path} must never leak the API_KEY secret`);
    }
  });

  test("POST /login with invalid key is rejected", async () => {
    const res = await fetch(`${base}/login`, {
      method: "POST",
      headers: { "content-type": "application/x-www-form-urlencoded" },
      body: "key=wrong-key&redirect=/dispatch"
    });
    assert.equal(res.status, 401);
    const text = await res.text();
    assert.match(text, /Invalid relay key/);
    assert.ok(!text.includes(API_KEY));
  });

  test("POST /login with valid key issues session cookie and redirects", async () => {
    const res = await fetch(`${base}/login`, {
      method: "POST",
      redirect: "manual",
      headers: { "content-type": "application/x-www-form-urlencoded" },
      body: `key=${encodeURIComponent(API_KEY)}&redirect=/dispatch`
    });
    assert.equal(res.status, 302);
    assert.equal(res.headers.get("location"), "/dispatch");
    const cookie = res.headers.get("set-cookie") || "";
    assert.ok(cookie.includes("echoshield_auth="));
    assert.ok(cookie.includes("HttpOnly"));
  });

  test("login redirects stay within the responder console", async () => {
    for (const redirect of ["//example.com", "/\\example.com", "https://example.com", 123, "/opius"]) {
      const res = await fetch(`${base}/login`, {
        method: "POST", redirect: "manual",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ key: API_KEY, redirect })
      });
      assert.equal(res.status, 302);
      assert.equal(res.headers.get("location"), "/dispatch");
    }
  });

  test("GET /dispatch and /dashboard with session cookie succeed and do NOT contain API key in HTML", async () => {
    const cookieHeader = `echoshield_auth=${encodeURIComponent(API_KEY)}`;
    for (const path of ["/dashboard", "/dispatch"]) {
      const res = await fetch(`${base}${path}`, {
        headers: { cookie: cookieHeader }
      });
      assert.equal(res.status, 200);
      const html = await res.text();
      assert.ok(!html.includes(API_KEY), `${path} HTML must not contain the raw API key`);
      assert.ok(html.includes("Log out"), `${path} should display Log out link when authenticated`);
    }
  });

  test("query parameter ?key= authenticates and redirects cleanly", async () => {
    const res = await fetch(`${base}/dispatch?key=${encodeURIComponent(API_KEY)}`, {
      redirect: "manual"
    });
    assert.equal(res.status, 302);
    assert.equal(res.headers.get("location"), "/dispatch");
    const cookie = res.headers.get("set-cookie") || "";
    assert.ok(cookie.includes("echoshield_auth="));
  });

  test("field reports appear in /dispatch console and cookie allows console actions", async () => {
    const cookieHeader = `echoshield_auth=${encodeURIComponent(API_KEY)}`;

    // 1. Create an incident via mesh alert
    const alertRes = await fetch(`${base}/v1/mesh/alerts`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "authorization": `Bearer ${API_KEY}`
      },
      body: JSON.stringify({
        protocolVersion: 1,
        deviceId: "EchoNode-A",
        messageId: "msg-101",
        alertType: "RESPONSE:TRIGGER",
        payload: "RESPONSE:TRIGGER|test-incident-session|37.7749|-122.4194|EchoNode-A,EchoNode-B|1710000000000",
        sourceNodeId: "EchoNode-A",
        observedAtMs: Date.now(),
        connectedPeerCount: 2,
        sessionId: "test-incident-session",
        latitude: 37.7749,
        longitude: -122.4194,
        confirmedByNodes: ["EchoNode-A", "EchoNode-B"]
      })
    });
    assert.equal(alertRes.status, 202);
    const alertData = await alertRes.json();
    const incidentId = alertData.incidentId;
    assert.ok(incidentId, "Expected incidentId to be returned");

    // 2. Submit a field report / note from a user
    const noteRes = await fetch(`${base}/v1/incidents/${encodeURIComponent(incidentId)}/notes`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "authorization": `Bearer ${API_KEY}`
      },
      body: JSON.stringify({
        roomNumber: "Room 304 (Math Wing)",
        safetyStatus: "INJURED",
        companionsCount: 5,
        injuredCount: 2,
        note: "Door barricaded with heavy desk, student has bleeding arm"
      })
    });
    assert.equal(noteRes.status, 202);

    // 3. Fetch /dispatch with the session cookie and inspect rendered field reports
    const dispatchRes = await fetch(`${base}/dispatch?incident=${encodeURIComponent(incidentId)}`, {
      headers: { cookie: cookieHeader }
    });
    assert.equal(dispatchRes.status, 200);
    const dispatchHtml = await dispatchRes.text();

    assert.ok(dispatchHtml.includes("Room 304 (Math Wing)"), "Field report room should be visible in dispatch");
    assert.ok(dispatchHtml.includes("INJURED"), "Safety status should be visible in dispatch");
    assert.ok(dispatchHtml.includes("People:"), "People count should be visible");
    assert.ok(dispatchHtml.includes("Injured:"), "Injured count should be visible");
    assert.ok(dispatchHtml.includes("Door barricaded with heavy desk"), "Note text should be visible");

    // 4. Send authority message using cookie (simulating dispatch UI form submit without bearer header)
    const msgRes = await fetch(`${base}/v1/incidents/${encodeURIComponent(incidentId)}/authority-messages`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "cookie": cookieHeader
      },
      body: JSON.stringify({
        sender: "Tactical Dispatch",
        role: "authority",
        message: "First responders are entering north stairwell. Keep doors secured."
      })
    });
    assert.equal(msgRes.status, 202);

    // 5. Clear notifications using cookie (simulating dispatch UI button click without bearer header)
    const clearRes = await fetch(`${base}/v1/incidents/${encodeURIComponent(incidentId)}/clear-notifications`, {
      method: "POST",
      headers: {
        "cookie": cookieHeader
      }
    });
    assert.equal(clearRes.status, 200);
  });
});

describe("Responder Console in Open Demo Mode (No API_KEY set)", () => {
  let port;
  let base;
  let child;

  before(async () => {
    port = await freePort();
    base = `http://127.0.0.1:${port}`;
    child = spawn(process.execPath, [SERVER], {
      env: {
        ...process.env,
        PORT: String(port),
        HOST: "127.0.0.1",
        ECHOSHIELD_RELAY_API_KEY: "",
        GEMINI_API_KEY: "",
        GOOGLE_API_KEY: ""
      },
      stdio: ["ignore", "pipe", "pipe"]
    });

    for (let i = 0; i < 50; i++) {
      try {
        const res = await fetch(`${base}/v1/time`);
        if (res.ok) return;
      } catch (_) {}
      await sleep(100);
    }
    throw new Error("Relay server did not start in time");
  });

  after(() => {
    child?.kill();
  });

  test("GET /dashboard and /dispatch work openly without credentials", async () => {
    for (const path of ["/dashboard", "/dispatch"]) {
      const res = await fetch(`${base}${path}`);
      assert.equal(res.status, 200);
      const text = await res.text();
      assert.ok(!text.includes("Log out"));
    }
  });
});
