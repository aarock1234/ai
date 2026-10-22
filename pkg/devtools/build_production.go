//go:build production

package devtools

// isProductionBuild is true in binaries built with the production build
// tag, where NewTelemetry and NewMiddleware return ErrProduction.
const isProductionBuild = true
