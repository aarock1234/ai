// Returns the serializer of the run-js-v1 format that upstream's runtime
// uses for host function arguments: the devalue format with upstream's
// reducers for errors and objects with an own "__proto__" member. The
// encoding must match upstream byte for byte, because continuations compare
// the serialized arguments of every replayed host function call.
(function () {
  "use strict";

  const UNDEFINED = -1;
  const HOLE = -2;
  const NAN = -3;
  const POSITIVE_INFINITY = -4;
  const NEGATIVE_INFINITY = -5;
  const NEGATIVE_ZERO = -6;
  const SPARSE = -7;
  const MAX_ARRAY_INDEX = 2 ** 32 - 2;

  const ObjectPrototype = Object.prototype;
  const ErrorConstructor = Error;
  const AggregateErrorConstructor = typeof AggregateError === "function" ? AggregateError : undefined;
  const MapConstructor = Map;
  const toText = String;
  const isNotANumber = Number.isNaN;

  const getPrototypeOf = Object.getPrototypeOf;
  const getOwnPropertyNames = Object.getOwnPropertyNames;
  const getOwnPropertySymbols = Object.getOwnPropertySymbols;
  const getOwnPropertyDescriptor = Object.getOwnPropertyDescriptor;
  const keysOf = Object.keys;
  const isArray = Array.isArray;
  const call = Function.prototype.call;
  const hasOwn = call.bind(ObjectPrototype.hasOwnProperty);
  const typeTagOf = call.bind(ObjectPrototype.toString);
  const sliceText = call.bind(String.prototype.slice);
  const charCodeAt = call.bind(String.prototype.charCodeAt);
  const sortList = call.bind(Array.prototype.sort);
  const mapHas = call.bind(MapConstructor.prototype.has);
  const mapGet = call.bind(MapConstructor.prototype.get);
  const mapSet = call.bind(MapConstructor.prototype.set);

  const BASE64_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

  const objectPrototypeNames = joinList(sortList(getOwnPropertyNames(ObjectPrototype)), "\0");

  function joinList(list, separator) {
    let text = "";

    for (let index = 0; index < list.length; index += 1) {
      if (index > 0) {
        text += separator;
      }

      text += list[index];
    }

    return text;
  }

  function failure(message) {
    return new ErrorConstructor(message);
  }

  function isPrimitive(value) {
    return value === null || (typeof value !== "object" && typeof value !== "function");
  }

  function isPlainObject(value) {
    const prototype = getPrototypeOf(value);

    return (
      prototype === ObjectPrototype ||
      prototype === null ||
      getPrototypeOf(prototype) === null ||
      joinList(sortList(getOwnPropertyNames(prototype)), "\0") === objectPrototypeNames
    );
  }

  function hasEnumerableSymbols(value) {
    const symbols = getOwnPropertySymbols(value);

    for (let index = 0; index < symbols.length; index += 1) {
      if (getOwnPropertyDescriptor(value, symbols[index]).enumerable) {
        return true;
      }
    }

    return false;
  }

  function escapedCharacter(character) {
    switch (character) {
      case '"':
        return '\\"';
      case "<":
        return "\\u003C";
      case "\\":
        return "\\\\";
      case "\n":
        return "\\n";
      case "\r":
        return "\\r";
      case "\t":
        return "\\t";
      case "\b":
        return "\\b";
      case "\f":
        return "\\f";
      case "\u2028":
        return "\\u2028";
      case "\u2029":
        return "\\u2029";
      default:
        if (character < " ") {
          const hex = charCodeAt(character, 0).toString(16);

          return "\\u" + "0000".slice(hex.length) + hex;
        }

        return "";
    }
  }

  function stringifyString(text) {
    let result = "";
    let last = 0;

    for (let index = 0; index < text.length; index += 1) {
      const replacement = escapedCharacter(text[index]);

      if (replacement) {
        result += sliceText(text, last, index) + replacement;
        last = index + 1;
      }
    }

    return '"' + (last === 0 ? text : result + sliceText(text, last)) + '"';
  }

  function stringifyPrimitive(value) {
    if (typeof value === "string") {
      return stringifyString(value);
    }

    if (typeof value === "bigint") {
      return '["BigInt","' + value + '"]';
    }

    return toText(value);
  }

  function isValidArrayIndexString(text) {
    if (text.length === 0 || (text.length > 1 && charCodeAt(text, 0) === 48)) {
      return false;
    }

    for (let index = 0; index < text.length; index += 1) {
      const code = charCodeAt(text, index);

      if (code < 48 || code > 57) {
        return false;
      }
    }

    return +text <= MAX_ARRAY_INDEX;
  }

  function validArrayIndices(array) {
    const keys = keysOf(array);
    let last = keys.length - 1;

    while (last >= 0 && !isValidArrayIndexString(keys[last])) {
      last -= 1;
    }

    const indices = [];
    for (let index = 0; index <= last; index += 1) {
      indices[index] = keys[index];
    }

    return indices;
  }

  function encodeBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    let output = "";

    for (let index = 0; index < bytes.length; index += 3) {
      const hasSecond = index + 1 < bytes.length;
      const hasThird = index + 2 < bytes.length;
      const bits = (bytes[index] << 16) | ((hasSecond ? bytes[index + 1] : 0) << 8) | (hasThird ? bytes[index + 2] : 0);

      output += BASE64_ALPHABET[(bits >>> 18) & 63];
      output += BASE64_ALPHABET[(bits >>> 12) & 63];
      output += hasSecond ? BASE64_ALPHABET[(bits >>> 6) & 63] : "=";
      output += hasThird ? BASE64_ALPHABET[bits & 63] : "=";
    }

    return output;
  }

  // Upstream's reducers, applied before devalue's own encodings.
  function reduce(value) {
    if (value instanceof ErrorConstructor) {
      const isAggregate = AggregateErrorConstructor !== undefined && value instanceof AggregateErrorConstructor;
      const errors = [];

      if (isAggregate) {
        for (const item of value.errors) {
          errors[errors.length] = item;
        }
      }

      return {
        key: "RunErrorValue",
        value: {
          cause: value.cause,
          errors: isAggregate ? errors : undefined,
          hasCause: hasOwn(value, "cause"),
          message: toText(value.message),
          name: toText(value.name),
        },
      };
    }

    if (typeof value !== "object" || value === null || isArray(value)) {
      return undefined;
    }

    const prototype = getPrototypeOf(value);
    if ((prototype !== ObjectPrototype && prototype !== null) || !hasOwn(value, "__proto__")) {
      return undefined;
    }

    const entries = [];
    const keys = keysOf(value);

    for (let index = 0; index < keys.length; index += 1) {
      entries[index] = [keys[index], value[keys[index]]];
    }

    return {
      key: "RunOwnProtoObject",
      value: [prototype === null, entries],
    };
  }

  function serialize(root) {
    const slots = [];
    const indexes = new MapConstructor();
    let next = 0;

    function flatten(value) {
      if (value === undefined) {
        return UNDEFINED;
      }

      if (value !== value) {
        return NAN;
      }

      if (value === Infinity) {
        return POSITIVE_INFINITY;
      }

      if (value === -Infinity) {
        return NEGATIVE_INFINITY;
      }

      if (value === 0 && 1 / value < 0) {
        return NEGATIVE_ZERO;
      }

      if (mapHas(indexes, value)) {
        return mapGet(indexes, value);
      }

      const index = next;
      next += 1;
      mapSet(indexes, value, index);

      const reduced = reduce(value);
      if (reduced !== undefined) {
        slots[index] = '["' + reduced.key + '",' + flatten(reduced.value) + "]";

        return index;
      }

      if (typeof value === "function") {
        throw failure("Cannot stringify a function");
      }

      if (typeof value === "symbol") {
        throw failure("Cannot stringify a Symbol primitive");
      }

      if (isPrimitive(value)) {
        slots[index] = stringifyPrimitive(value);
      } else if (typeof value.then === "function") {
        throw failure("Cannot stringify a Promise or thenable \u2014 use stringifyAsync instead");
      } else {
        slots[index] = stringifyObject(value);
      }

      return index;
    }

    function stringifyObject(value) {
      const type = sliceText(typeTagOf(value), 8, -1);

      switch (type) {
        case "Number":
        case "String":
        case "Boolean":
        case "BigInt":
          return '["Object",' + flatten(value.valueOf()) + "]";

        case "Date": {
          const valid = !isNotANumber(value.getDate());

          return '["Date","' + (valid ? value.toISOString() : "") + '"]';
        }

        case "RegExp": {
          const source = value.source;
          const flags = value.flags;

          return flags ? '["RegExp",' + stringifyString(source) + ',"' + flags + '"]' : '["RegExp",' + stringifyString(source) + "]";
        }

        case "Array":
          return stringifyArray(value);

        case "Set": {
          let text = '["Set"';

          for (const item of value) {
            text += "," + flatten(item);
          }

          return text + "]";
        }

        case "Map": {
          let text = '["Map"';

          for (const [key, item] of value) {
            text += "," + flatten(key) + "," + flatten(item);
          }

          return text + "]";
        }

        case "Int8Array":
        case "Uint8Array":
        case "Uint8ClampedArray":
        case "Int16Array":
        case "Uint16Array":
        case "Float16Array":
        case "Int32Array":
        case "Uint32Array":
        case "Float32Array":
        case "Float64Array":
        case "BigInt64Array":
        case "BigUint64Array": {
          let text = '["' + type + '",' + flatten(value.buffer);

          if (value.byteLength !== value.buffer.byteLength) {
            text += "," + value.byteOffset + "," + value.length;
          }

          return text + "]";
        }

        case "DataView": {
          let text = '["' + type + '",' + flatten(value.buffer);

          if (value.byteLength !== value.buffer.byteLength) {
            text += "," + value.byteOffset + "," + value.byteLength;
          }

          return text + "]";
        }

        case "ArrayBuffer":
          return '["ArrayBuffer","' + encodeBase64(value) + '"]';

        default:
          return stringifyPlainObject(value);
      }
    }

    function stringifyArray(array) {
      let mostlyDense = false;
      let text = "[";

      for (let index = 0; index < array.length; index += 1) {
        if (index > 0) {
          text += ",";
        }

        if (hasOwn(array, index)) {
          text += flatten(array[index]);
        } else if (mostlyDense) {
          text += HOLE;
        } else {
          const populated = validArrayIndices(array);
          const digits = toText(array.length).length;
          const holeCost = (array.length - populated.length) * 3;
          const sparseCost = 4 + digits + populated.length * (digits + 1);

          if (holeCost > sparseCost) {
            text = "[" + SPARSE + "," + array.length;

            for (let position = 0; position < populated.length; position += 1) {
              const key = populated[position];

              text += "," + key + "," + flatten(array[key]);
            }

            break;
          }

          mostlyDense = true;
          text += HOLE;
        }
      }

      return text + "]";
    }

    function stringifyPlainObject(value) {
      if (!isPlainObject(value)) {
        throw failure("Cannot stringify arbitrary non-POJOs");
      }

      if (hasEnumerableSymbols(value)) {
        throw failure("Cannot stringify POJOs with symbolic keys");
      }

      const keys = keysOf(value);
      const nullPrototype = getPrototypeOf(value) === null;
      let text = nullPrototype ? '["null"' : "{";

      for (let index = 0; index < keys.length; index += 1) {
        const key = keys[index];

        if (key === "__proto__") {
          throw failure("Cannot stringify objects with __proto__ keys");
        }

        if (nullPrototype) {
          text += "," + stringifyString(key) + "," + flatten(value[key]);
        } else {
          text += (index > 0 ? "," : "") + stringifyString(key) + ":" + flatten(value[key]);
        }
      }

      return text + (nullPrototype ? "]" : "}");
    }

    const index = flatten(root);
    if (index < 0) {
      return "" + index;
    }

    return "[" + joinList(slots, ",") + "]";
  }

  return serialize;
});
