package devtools

import (
	"bytes"
	"embed"
	"io/fs"
	"net/http"
	"os"
	"path"
	"strings"
	"time"
)

// The viewer is the built React client of the published @ai-sdk/devtools
// 1.0.49 npm package, whose src directory matches the pinned upstream
// reference. The files in the client directory are the package's
// dist/client files, embedded verbatim. They are Copyright 2023 Vercel,
// Inc. and licensed under the Apache License, Version 2.0; the license text
// is client/LICENSE, which is embedded with them.

// clientFiles holds the viewer's static assets and their license.
//
//go:embed client/assets client/LICENSE
var clientFiles embed.FS

// viewerPage is client/index.html, the page served for every path outside
// the API and the assets.
//
//go:embed client/index.html
var viewerPage []byte

// clientDirectory is the directory of clientFiles that holds the assets.
const clientDirectory = "client"

// assetsPathPrefix is the URL path prefix of the viewer's static assets.
const assetsPathPrefix = "/assets/"

// developmentPage is served instead of the viewer in development mode, when
// the client runs on upstream's Vite development server.
//
//go:embed development.html
var developmentPage []byte

// developmentEnvironmentVariable enables development mode, in which the
// server only serves the API.
const developmentEnvironmentVariable = "AI_SDK_DEVTOOLS_DEV"

// assetContentTypes are the content types of the embedded assets, as
// upstream's static file handler sends them.
var assetContentTypes = map[string]string{
	".css": "text/css; charset=utf-8",
	".js":  "text/javascript; charset=utf-8",
}

// handleClient serves the viewer outside the API: assets under /assets/
// for any method, like upstream's static file middleware, and the viewer
// page for every other GET request.
func handleClient(w http.ResponseWriter, r *http.Request) {
	if strings.HasPrefix(r.URL.Path, assetsPathPrefix) {
		handleAsset(w, r)

		return
	}

	serveViewerFallback(w, r)
}

// handleAsset serves an embedded asset. Paths without an asset fall back to
// the viewer page, so that the client handles them.
func handleAsset(w http.ResponseWriter, r *http.Request) {
	content, ok := readAsset(r.URL.Path)
	if !ok {
		serveViewerFallback(w, r)

		return
	}

	w.Header().Set("Content-Type", assetContentType(r.URL.Path))
	http.ServeContent(w, r, "", time.Time{}, bytes.NewReader(content))
}

// serveViewerFallback serves the viewer page for GET and HEAD requests and
// answers other methods with 404, like upstream's catch-all GET route.
func serveViewerFallback(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		http.NotFound(w, r)

		return
	}

	handleViewer(w, r)
}

// handleViewer serves the viewer page, or the development page in
// development mode.
func handleViewer(w http.ResponseWriter, _ *http.Request) {
	page := developmentPage
	if !isDevelopmentMode() {
		page = viewerPage
	}

	w.Header().Set("Content-Type", "text/html; charset=UTF-8")
	_, _ = w.Write(page)
}

// readAsset returns the embedded file of an asset URL path. Directories and
// missing files are not assets.
func readAsset(urlPath string) ([]byte, bool) {
	name := path.Join(clientDirectory, strings.TrimPrefix(path.Clean(urlPath), "/"))
	if !strings.HasPrefix(name, clientDirectory+assetsPathPrefix) {
		return nil, false
	}

	info, err := fs.Stat(clientFiles, name)
	if err != nil || !info.Mode().IsRegular() {
		return nil, false
	}

	content, err := clientFiles.ReadFile(name)
	if err != nil {
		return nil, false
	}

	return content, true
}

func assetContentType(urlPath string) string {
	contentType, ok := assetContentTypes[path.Ext(urlPath)]
	if !ok {
		return "application/octet-stream"
	}

	return contentType
}

// isDevelopmentMode reports whether AI_SDK_DEVTOOLS_DEV is set to a value
// other than "false" and "0".
func isDevelopmentMode() bool {
	value, ok := os.LookupEnv(developmentEnvironmentVariable)

	return ok && value != "false" && value != "0"
}
