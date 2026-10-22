package policyopawasm

// OPA policies import their memory and host functions from the env module.
// wazero host modules cannot define memory, so each policy instance gets a
// small WebAssembly env module that owns the memory and re-exports the host
// functions of the shared opa_host module.
const (
	envModuleName  = "env"
	hostModuleName = "opa_host"
)

// envMemoryExport is the name under which the env module exports memory.
const envMemoryExport = "memory"

// WebAssembly binary format constants used to encode the env module.
const (
	wasmTypeSection   = 1
	wasmImportSection = 2
	wasmMemorySection = 5
	wasmExportSection = 7

	wasmFunctionKind = 0x00
	wasmMemoryKind   = 0x02

	wasmFunctionType = 0x60
	wasmI32          = 0x7f

	wasmLimitsMinimum        = 0x00
	wasmLimitsMinimumMaximum = 0x01
)

// wasmHeader is the magic number and version of a WebAssembly binary.
var wasmHeader = []byte{
	0x00, 0x61, 0x73, 0x6d,
	0x01, 0x00, 0x00, 0x00,
}

// memoryLimits are the page limits of the env memory.
type memoryLimits struct {
	initialPages uint32
	maxPages     uint32
	hasMax       bool
}

// encodeEnvModule returns a WebAssembly module that imports every host
// function from opa_host, defines a memory with limits and exports both
// under the names OPA policies import from env.
func encodeEnvModule(limits memoryLimits) []byte {
	module := append([]byte{}, wasmHeader...)
	module = appendSection(module, wasmTypeSection, encodeTypeSection())
	module = appendSection(module, wasmImportSection, encodeImportSection())
	module = appendSection(module, wasmMemorySection, encodeMemorySection(limits))
	module = appendSection(module, wasmExportSection, encodeExportSection())

	return module
}

// encodeTypeSection declares one function type per host function, in the
// order of hostFunctions.
func encodeTypeSection() []byte {
	section := appendU32(nil, uint32(len(hostFunctions)))

	for _, function := range hostFunctions {
		section = append(section, wasmFunctionType)

		section = appendU32(section, function.parameters)
		for range function.parameters {
			section = append(section, wasmI32)
		}

		if function.returnsAddress {
			section = append(section, 1, wasmI32)
		} else {
			section = append(section, 0)
		}
	}

	return section
}

// encodeImportSection imports every host function from opa_host.
func encodeImportSection() []byte {
	section := appendU32(nil, uint32(len(hostFunctions)))

	for index, function := range hostFunctions {
		section = appendName(section, hostModuleName)
		section = appendName(section, function.name)
		section = append(section, wasmFunctionKind)
		section = appendU32(section, uint32(index))
	}

	return section
}

// encodeMemorySection defines the single memory of the env module.
func encodeMemorySection(limits memoryLimits) []byte {
	section := appendU32(nil, 1)

	if !limits.hasMax {
		section = append(section, wasmLimitsMinimum)

		return appendU32(section, limits.initialPages)
	}

	section = append(section, wasmLimitsMinimumMaximum)
	section = appendU32(section, limits.initialPages)

	return appendU32(section, limits.maxPages)
}

// encodeExportSection exports the memory and every imported host function.
func encodeExportSection() []byte {
	section := appendU32(nil, uint32(len(hostFunctions)+1))

	section = appendName(section, envMemoryExport)
	section = append(section, wasmMemoryKind)
	section = appendU32(section, 0)

	for index, function := range hostFunctions {
		section = appendName(section, function.name)
		section = append(section, wasmFunctionKind)
		section = appendU32(section, uint32(index))
	}

	return section
}

// appendSection appends a section with its identifier and byte size.
func appendSection(module []byte, id byte, content []byte) []byte {
	module = append(module, id)
	module = appendU32(module, uint32(len(content)))

	return append(module, content...)
}

// appendName appends a length-prefixed UTF-8 name.
func appendName(buffer []byte, name string) []byte {
	buffer = appendU32(buffer, uint32(len(name)))

	return append(buffer, name...)
}

// appendU32 appends value as an unsigned LEB128 integer.
func appendU32(buffer []byte, value uint32) []byte {
	for value >= 0x80 {
		buffer = append(buffer, byte(value)|0x80)
		value >>= 7
	}

	return append(buffer, byte(value))
}
