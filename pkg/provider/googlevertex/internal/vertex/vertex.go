// Package vertex holds the project, location and host rules shared by the
// Google Vertex AI providers. Settings read from options or the environment
// become part of request hosts and paths, so they are validated and escaped
// here.
package vertex

import (
	"fmt"
	"os"
	"regexp"
	"strings"

	"github.com/aarock1234/ai/internal/urlpath"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// ProjectVariable holds the Google Cloud project.
	ProjectVariable = "GOOGLE_VERTEX_PROJECT"

	// LocationVariable holds the Google Cloud location.
	LocationVariable = "GOOGLE_VERTEX_LOCATION"
)

// hostnameLabelPattern matches a single DNS label.
var hostnameLabelPattern = regexp.MustCompile(`(?i)^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$`)

// IsHostnameLabel reports whether value is a single DNS label, which a
// value must be to form part of a request host.
func IsHostnameLabel(value string) bool {
	return hostnameLabelPattern.MatchString(value)
}

// Host returns the Vertex AI host of a location: the global host, a
// multi-region host or a regional host.
func Host(location string) string {
	switch location {
	case "global":
		return "aiplatform.googleapis.com"
	case "eu", "us":
		return "aiplatform." + location + ".rep.googleapis.com"
	default:
		return location + "-aiplatform.googleapis.com"
	}
}

// LoadLocation returns the configured location, GOOGLE_VERTEX_LOCATION or
// fallback. An empty fallback makes a missing location an error. The
// location becomes part of the request host, so only a single DNS label is
// accepted.
func LoadLocation(configured string, fallback string) (string, error) {
	location := configured
	if location == "" {
		location = os.Getenv(LocationVariable)
	}

	if location == "" {
		location = fallback
	}

	if location == "" {
		return "", fmt.Errorf(
			"%w: Google Vertex location is missing; set %s or use WithLocation",
			model.ErrInvalidArgument,
			LocationVariable,
		)
	}

	if !IsHostnameLabel(location) {
		return "", fmt.Errorf(
			"%w: invalid Google Vertex location; expected a single DNS label (letters, digits, and hyphens); use WithBaseURL for custom endpoints",
			model.ErrInvalidArgument,
		)
	}

	return location, nil
}

// LoadProject returns the configured project or GOOGLE_VERTEX_PROJECT,
// escaped as a URL path segment.
func LoadProject(configured string) (string, error) {
	project := configured
	if project == "" {
		project = os.Getenv(ProjectVariable)
	}

	if strings.TrimSpace(project) == "" {
		return "", fmt.Errorf(
			"%w: Google Vertex project is missing; set %s or use WithProject",
			model.ErrInvalidArgument,
			ProjectVariable,
		)
	}

	return urlpath.Escape(project), nil
}

// ModelResource returns the resource name of a publisher model, such as
// "projects/p/locations/us-central1/publishers/google/models/gemini-3".
// project and location must come from LoadProject and LoadLocation.
func ModelResource(project string, location string, publisher string, modelID string) string {
	return "projects/" + project +
		"/locations/" + location +
		"/publishers/" + urlpath.Escape(publisher) +
		"/models/" + urlpath.Escape(modelID)
}
