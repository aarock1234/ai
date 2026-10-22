// Supervises Pi's RPC mode inside the sandbox for the Go Pi harness adapter.
//
// Sandbox processes expose no standard input, so the host writes each RPC
// record as a numbered file in the inbox directory and this relay feeds the
// records to Pi's standard input in order. Pi's standard output is appended
// to the outbox file, which the host follows from a byte offset, so that a
// later host process can reattach to a running Pi without losing records.
//
// Usage: node relay.mjs <state-dir> <pi-command> [pi-arguments...]
//
// State directory layout:
//   inbox/<sequence>.jsonl  one RPC record, consumed in sequence order
//   inbox/<sequence>.close  closes Pi's standard input (orderly shutdown)
//   outbox.jsonl            Pi's standard output, append-only
//   stderr.log              Pi's standard error, append-only
//   relay.pid               the relay's process ID
//   exit.json               written once Pi exited

import { spawn } from "node:child_process";
import {
  closeSync,
  existsSync,
  mkdirSync,
  openSync,
  readFileSync,
  renameSync,
  unlinkSync,
  watch,
  writeFileSync,
  writeSync,
} from "node:fs";
import path from "node:path";

const POLL_INTERVAL_MS = 50;
const SEQUENCE_WIDTH = 12;

const [stateDir, command, ...args] = process.argv.slice(2);

if (!stateDir || !command) {
  process.stderr.write("usage: node relay.mjs <state-dir> <pi-command> [pi-arguments...]\n");
  process.exit(2);
}

const inboxDir = path.join(stateDir, "inbox");
mkdirSync(inboxDir, { recursive: true });

const outbox = openSync(path.join(stateDir, "outbox.jsonl"), "a");
const stderrLog = openSync(path.join(stateDir, "stderr.log"), "a");

writeFileAtomically(path.join(stateDir, "relay.pid"), `${process.pid}\n`);

const pi = spawn(command, args, {
  stdio: ["pipe", "pipe", "pipe"],
  env: process.env,
});

let nextSequence = 1;
let stdinClosed = false;
let waitingForDrain = false;
let finished = false;

pi.stdout.on("data", (chunk) => {
  writeSync(outbox, chunk);
});

pi.stderr.on("data", (chunk) => {
  writeSync(stderrLog, chunk);
});

pi.stdin.on("error", () => {
  stdinClosed = true;
});

pi.on("error", (error) => {
  writeSync(stderrLog, `relay: failed to start pi: ${error.message}\n`);
  finish({ code: 127, signal: null });
});

pi.on("close", (code, signal) => {
  finish({ code, signal });
});

for (const signal of ["SIGTERM", "SIGINT"]) {
  process.on(signal, () => {
    pi.kill(signal);
  });
}

// The host may disconnect without stopping Pi; Pi keeps running for a later
// host process.
process.on("SIGHUP", () => {});

const watcher = watchInbox();
const poller = setInterval(pump, POLL_INTERVAL_MS);
pump();

// pump feeds every complete record that is next in sequence to Pi.
function pump() {
  while (!stdinClosed && !waitingForDrain) {
    const sequence = String(nextSequence).padStart(SEQUENCE_WIDTH, "0");
    const recordPath = path.join(inboxDir, `${sequence}.jsonl`);
    const closePath = path.join(inboxDir, `${sequence}.close`);

    if (existsSync(closePath)) {
      removeQuietly(closePath);
      stdinClosed = true;
      pi.stdin.end();

      return;
    }

    const record = readCompleteRecord(recordPath);
    if (record === undefined) {
      return;
    }

    removeQuietly(recordPath);
    nextSequence += 1;

    if (!pi.stdin.write(record)) {
      waitingForDrain = true;
      pi.stdin.once("drain", () => {
        waitingForDrain = false;
        pump();
      });
    }
  }
}

// readCompleteRecord returns a record once the host finished writing it.
// Records are single JSON objects whose only line feed terminates them.
function readCompleteRecord(recordPath) {
  let content;

  try {
    content = readFileSync(recordPath);
  } catch {
    return undefined;
  }

  if (content.length === 0 || content[content.length - 1] !== 0x0a) {
    return undefined;
  }

  return content;
}

function watchInbox() {
  try {
    return watch(inboxDir, () => pump());
  } catch {
    return undefined;
  }
}

function finish({ code, signal }) {
  if (finished) {
    return;
  }

  finished = true;
  clearInterval(poller);
  watcher?.close();

  writeFileAtomically(
    path.join(stateDir, "exit.json"),
    `${JSON.stringify({ code: code ?? null, signal: signal ?? null })}\n`,
  );

  closeSync(outbox);
  closeSync(stderrLog);
  process.exit(typeof code === "number" ? code : 1);
}

function writeFileAtomically(filePath, content) {
  const temporaryPath = `${filePath}.${process.pid}.tmp`;
  writeFileSync(temporaryPath, content);
  renameSync(temporaryPath, filePath);
}

function removeQuietly(filePath) {
  try {
    unlinkSync(filePath);
  } catch {
    // The record was already consumed.
  }
}
