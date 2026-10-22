// Package origin compares URL origins the way upstream isSameOrigin does with
// the WHATWG URL origin: scheme, host and port, with the scheme and host
// lower case and the scheme's default port implied. Providers use it to send
// credentials only to their own origin when a response names another URL.
package origin

import (
	"net/url"
	"strings"
)

// defaultPorts are the ports implied by URLs without an explicit port.
var defaultPorts = map[string]string{
	"http":  "80",
	"https": "443",
	"ws":    "80",
	"wss":   "443",
}

// Of returns the serialized origin of target, such as "https://example.com"
// or "http://[::1]:8080". The default port of the scheme is omitted.
func Of(target *url.URL) string {
	scheme := strings.ToLower(target.Scheme)
	host := strings.ToLower(target.Hostname())

	if strings.Contains(host, ":") {
		host = "[" + host + "]"
	}

	if port := target.Port(); port != "" && port != defaultPorts[scheme] {
		host += ":" + port
	}

	return scheme + "://" + host
}

// SameURL reports whether two parsed URLs share an origin.
func SameURL(first *url.URL, second *url.URL) bool {
	return Of(first) == Of(second)
}

// Same reports whether two URLs share an origin. URLs that cannot be parsed
// or are not absolute never match.
func Same(first string, second string) bool {
	firstURL, ok := parseAbsolute(first)
	if !ok {
		return false
	}

	secondURL, ok := parseAbsolute(second)
	if !ok {
		return false
	}

	return SameURL(firstURL, secondURL)
}

func parseAbsolute(location string) (*url.URL, bool) {
	parsed, err := url.Parse(location)
	if err != nil {
		return nil, false
	}

	isAbsolute := parsed.Scheme != "" && parsed.Host != ""

	return parsed, isAbsolute
}
