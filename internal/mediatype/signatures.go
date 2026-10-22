package mediatype

type mediaSignature struct {
	mediaType string
	prefix    []int
}

// These ordered signatures mirror the upstream detect-media-type tables.
// Negative prefix entries are wildcards. Ordering is significant for containers
// such as MP4 and RIFF; a declared top-level type selects the matching table.
var mediaSignatures = []mediaSignature{
	{
		mediaType: "image/gif",
		prefix:    []int{0x47, 0x49, 0x46, 0x38, 0x37, 0x61},
	},
	{
		mediaType: "image/gif",
		prefix:    []int{0x47, 0x49, 0x46, 0x38, 0x39, 0x61},
	},
	{
		mediaType: "image/png",
		prefix:    []int{0x89, 0x50, 0x4e, 0x47},
	},
	{
		mediaType: "image/jpeg",
		prefix:    []int{0xff, 0xd8},
	},
	{
		mediaType: "image/webp",
		prefix:    []int{0x52, 0x49, 0x46, 0x46, -1, -1, -1, -1, 0x57, 0x45, 0x42, 0x50},
	},
	{
		mediaType: "image/bmp",
		prefix:    []int{0x42, 0x4d, -1, -1, -1, -1, 0, 0, 0, 0},
	},
	{
		mediaType: "image/tiff",
		prefix:    []int{0x49, 0x49, 0x2a, 0},
	},
	{
		mediaType: "image/tiff",
		prefix:    []int{0x4d, 0x4d, 0, 0x2a},
	},
	{
		mediaType: "image/avif",
		prefix:    []int{0, 0, 0, -1, 0x66, 0x74, 0x79, 0x70, 0x61, 0x76, 0x69, 0x66},
	},
	{
		mediaType: "image/heic",
		prefix:    []int{0, 0, 0, -1, 0x66, 0x74, 0x79, 0x70, 0x68, 0x65, 0x69, 0x63},
	},
	{
		mediaType: "application/pdf",
		prefix:    []int{0x25, 0x50, 0x44, 0x46},
	},
	{
		mediaType: "audio/aac",
		prefix:    []int{0xff, 0xf0},
	},
	{
		mediaType: "audio/aac",
		prefix:    []int{0xff, 0xf1},
	},
	{
		mediaType: "audio/aac",
		prefix:    []int{0xff, 0xf8},
	},
	{
		mediaType: "audio/aac",
		prefix:    []int{0xff, 0xf9},
	},
	{
		mediaType: "audio/mpeg",
		prefix:    []int{0xff, 0xfb},
	},
	{
		mediaType: "audio/mpeg",
		prefix:    []int{0xff, 0xfa},
	},
	{
		mediaType: "audio/mpeg",
		prefix:    []int{0xff, 0xf3},
	},
	{
		mediaType: "audio/mpeg",
		prefix:    []int{0xff, 0xf2},
	},
	{
		mediaType: "audio/mpeg",
		prefix:    []int{0xff, 0xe3},
	},
	{
		mediaType: "audio/mpeg",
		prefix:    []int{0xff, 0xe2},
	},
	{
		mediaType: "audio/wav",
		prefix:    []int{0x52, 0x49, 0x46, 0x46, -1, -1, -1, -1, 0x57, 0x41, 0x56, 0x45},
	},
	{
		mediaType: "audio/ogg",
		prefix:    []int{0x4f, 0x67, 0x67, 0x53},
	},
	{
		mediaType: "audio/flac",
		prefix:    []int{0x66, 0x4c, 0x61, 0x43},
	},
	{
		mediaType: "audio/aac",
		prefix:    []int{0x40, 0x15, 0, 0},
	},
	{
		mediaType: "audio/webm",
		prefix:    []int{0x1a, 0x45, 0xdf, 0xa3},
	},
	{
		mediaType: "audio/mp4",
		prefix:    []int{0, 0, 0, -1, 0x66, 0x74, 0x79, 0x70},
	},
	{
		mediaType: "video/mp4",
		prefix:    []int{0, 0, 0, -1, 0x66, 0x74, 0x79, 0x70},
	},
	{
		mediaType: "video/webm",
		prefix:    []int{0x1a, 0x45, 0xdf, 0xa3},
	},
	{
		mediaType: "video/quicktime",
		prefix:    []int{0, 0, 0, 0x14, 0x66, 0x74, 0x79, 0x70, 0x71, 0x74},
	},
	{
		mediaType: "video/x-msvideo",
		prefix:    []int{0x52, 0x49, 0x46, 0x46},
	},
}
