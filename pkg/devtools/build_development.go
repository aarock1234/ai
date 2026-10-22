//go:build !production

package devtools

// isProductionBuild is false in binaries built without the production build
// tag.
const isProductionBuild = false
