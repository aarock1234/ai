// Installs the sandbox globals before the program runs and returns the
// helpers the host uses. The helpers are captured here so that the program
// cannot replace them.
//
// Host tool calls follow upstream's runtime: tools.name(input) serializes
// its argument list right away and returns a promise-like whose host call
// starts when the program first awaits it, so that calls are numbered in
// the same order in either implementation.
(function (bridge, serialize) {
  "use strict";

  const ErrorConstructor = Error;
  const TypeErrorConstructor = TypeError;
  const PromiseConstructor = Promise;
  const stringify = JSON.stringify;
  const parse = JSON.parse;
  const toText = String;
  const freeze = Object.freeze;
  const call = Function.prototype.call;
  const promiseThen = call.bind(PromiseConstructor.prototype.then);
  const promiseCatch = call.bind(PromiseConstructor.prototype.catch);
  const promiseFinally = call.bind(PromiseConstructor.prototype.finally);
  const resolvedPromise = () => PromiseConstructor.resolve();
  const toStringTag = Symbol.toStringTag;

  const bindings = Object.create(null);
  for (const name of Object.keys(bridge.bindings)) {
    bindings[name] = bridge.bindings[name];
  }

  const records = [];

  function formatConsoleValue(value) {
    if (typeof value === "string") {
      return value;
    }

    try {
      const encoded = stringify(value);

      return encoded === undefined ? toText(value) : encoded;
    } catch (error) {
      return toText(value);
    }
  }

  function consoleMethod(level) {
    return function (...values) {
      const parts = [];

      for (const value of values) {
        parts.push(formatConsoleValue(value));
      }

      bridge.writeConsole(level, parts.join(" "));
    };
  }

  function serializationError(error) {
    const message = error && error.message ? toText(error.message) : toText(error);
    const result = new TypeErrorConstructor("Host function arguments are not serializable: " + message);
    result.code = "RUN_SERIALIZATION_ERROR";

    return result;
  }

  // createBridgePromise returns a promise-like that starts the host call
  // the first time the program observes it.
  function createBridgePromise(toolName, start) {
    const record = {
      toolName: toolName,
      observed: false,
      status: "idle",
    };

    let promise;
    records.push(record);

    function getPromise() {
      record.observed = true;

      if (!promise) {
        record.status = "pending";
        promise = promiseThen(
          promiseThen(resolvedPromise(), start),
          function (value) {
            record.status = "fulfilled";

            return value;
          },
          function (error) {
            record.status = "rejected";

            throw error;
          },
        );
      }

      return promise;
    }

    return freeze({
      then(onFulfilled, onRejected) {
        return promiseThen(getPromise(), onFulfilled, onRejected);
      },
      catch(onRejected) {
        return promiseCatch(getPromise(), onRejected);
      },
      finally(onFinally) {
        return promiseFinally(getPromise(), onFinally);
      },
      get [toStringTag]() {
        return "Promise";
      },
    });
  }

  function callHost(bindingName, toolName, args) {
    let argumentsJson;

    try {
      argumentsJson = serialize(args);
    } catch (error) {
      throw serializationError(error);
    }

    return createBridgePromise(toolName, function () {
      return bridge.callTool(bindingName, argumentsJson);
    });
  }

  const tools = new Proxy(Object.create(null), {
    get(_target, name) {
      if (typeof name !== "string") {
        return undefined;
      }

      const binding = bindings[name];
      if (binding !== undefined) {
        return function (input) {
          return callHost(binding, name, [input]);
        };
      }

      return function (input) {
        return callHost(bridge.missingBinding, name, [name, input]);
      };
    },
  });

  const sandboxConsole = freeze({
    log: consoleMethod("log"),
    info: consoleMethod("info"),
    warn: consoleMethod("warn"),
    error: consoleMethod("error"),
    debug: consoleMethod("debug"),
  });

  Object.defineProperty(globalThis, "tools", {
    value: tools,
  });

  Object.defineProperty(globalThis, "console", {
    value: sandboxConsole,
  });

  function createError(message, name, code) {
    const error = new ErrorConstructor(message);
    error.name = name;

    if (code !== undefined) {
      error.code = code;
    }

    return error;
  }

  function describeError(reason) {
    try {
      if (reason === null || typeof reason !== "object") {
        return {
          name: "",
          message: toText(reason),
          stack: "",
        };
      }

      return {
        name: reason.name === undefined ? "" : toText(reason.name),
        message: reason.message === undefined ? "" : toText(reason.message),
        stack: reason.stack === undefined ? "" : toText(reason.stack),
      };
    } catch (error) {
      return {
        name: "",
        message: "uncaught exception",
        stack: "",
      };
    }
  }

  // detachedCall describes the first host tool call the program did not
  // await, or that was still running when it returned.
  function detachedCall() {
    for (const record of records) {
      if (!record.observed) {
        return {
          toolName: record.toolName,
          awaited: false,
        };
      }

      if (record.status === "pending") {
        return {
          toolName: record.toolName,
          awaited: true,
        };
      }
    }

    return null;
  }

  return {
    stringify,
    parse,
    createError,
    describeError,
    detachedCall,
  };
});
