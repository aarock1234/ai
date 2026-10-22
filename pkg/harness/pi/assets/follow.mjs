// Streams a file the relay appends to, such as Pi's outbox or its standard
// error log, to standard output for the Go Pi harness adapter. It starts at
// a byte offset and keeps following the file while Pi runs, and exits once
// the exit file exists and every byte of the file was written.
//
// Usage: node follow.mjs <file> <exit-file> <offset>

import { closeSync, existsSync, openSync, readSync, watch } from "node:fs";

const POLL_INTERVAL_MS = 50;
const CHUNK_SIZE = 64 * 1024;

const [filePath, exitPath, offsetArgument] = process.argv.slice(2);

if (!filePath || !exitPath) {
  process.stderr.write("usage: node follow.mjs <file> <exit-file> <offset>\n");
  process.exit(2);
}

let position = Number.parseInt(offsetArgument ?? "0", 10);
if (!Number.isSafeInteger(position) || position < 0) {
  process.stderr.write(`follow: invalid offset ${offsetArgument}\n`);
  process.exit(2);
}

const file = openSync(filePath, "a+");
const buffer = Buffer.allocUnsafe(CHUNK_SIZE);

let writing = false;
let finished = false;
let watcher;

try {
  watcher = watch(filePath, () => drain());
} catch {
  watcher = undefined;
}

const poller = setInterval(drain, POLL_INTERVAL_MS);
process.stdout.on("error", () => process.exit(0));
drain();

// drain writes every byte appended since the last read, honoring standard
// output backpressure. The exit file is checked before reading, so that the
// bytes Pi wrote before exiting are always written.
function drain() {
  if (writing || finished) {
    return;
  }

  for (;;) {
    const exited = existsSync(exitPath);
    const bytesRead = readSync(file, buffer, 0, CHUNK_SIZE, position);

    if (bytesRead === 0) {
      if (exited) {
        finish();
      }

      return;
    }

    position += bytesRead;

    if (!process.stdout.write(Buffer.from(buffer.subarray(0, bytesRead)))) {
      writing = true;
      process.stdout.once("drain", () => {
        writing = false;
        drain();
      });

      return;
    }
  }
}

function finish() {
  finished = true;
  clearInterval(poller);
  watcher?.close();
  closeSync(file);
  process.stdout.end(() => process.exit(0));
}
