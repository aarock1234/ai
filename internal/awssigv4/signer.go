// Package awssigv4 signs HTTP requests with AWS Signature Version 4 and
// loads the AWS credentials and region that signing needs.
//
// The signer follows the aws4fetch library that the upstream providers use:
// it signs every header with an ASCII value except a fixed set of
// unsignable ones, hashes the payload and URI-encodes the already escaped
// path a second time, as AWS services other than S3 require.
package awssigv4

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/url"
	"regexp"
	"slices"
	"strings"
	"time"
)

const (
	algorithm      = "AWS4-HMAC-SHA256"
	dateTimeFormat = "20060102T150405Z"
	dateFormat     = "20060102"
	scopeTerminal  = "aws4_request"
)

// unsignableHeaders are left out of the signature, as aws4fetch does.
var unsignableHeaders = map[string]bool{
	"authorization":     true,
	"content-type":      true,
	"content-length":    true,
	"user-agent":        true,
	"presigned-expires": true,
	"expect":            true,
	"x-amzn-trace-id":   true,
	"range":             true,
	"connection":        true,
}

var (
	repeatedSlashes = regexp.MustCompile(`/+`)
	whitespaceRun   = regexp.MustCompile(`\s+`)
)

// Credentials are the AWS keys that sign requests.
type Credentials struct {
	AccessKeyID     string
	SecretAccessKey string

	// SessionToken is sent as X-Amz-Security-Token for temporary
	// credentials.
	SessionToken string
}

// Signer signs requests for one AWS service and region.
type Signer struct {
	Credentials Credentials
	Region      string
	Service     string

	// Now returns the signing time. Nil uses time.Now.
	Now func() time.Time
}

// Sign sets the X-Amz-Date, X-Amz-Security-Token and Authorization headers
// of request, whose payload is body.
func (s Signer) Sign(request *http.Request, body []byte) {
	signedAt := s.now().UTC()
	dateTime := signedAt.Format(dateTimeFormat)

	request.Header.Set("X-Amz-Date", dateTime)

	// A token left from earlier temporary credentials must not be signed
	// with keys it does not belong to.
	if s.Credentials.SessionToken == "" {
		request.Header.Del("X-Amz-Security-Token")
	} else {
		request.Header.Set("X-Amz-Security-Token", s.Credentials.SessionToken)
	}

	canonical, signedHeaders := canonicalRequest(request, body)
	scope := strings.Join(
		[]string{
			signedAt.Format(dateFormat),
			s.Region,
			s.Service,
			scopeTerminal,
		},
		"/",
	)

	stringToSign := strings.Join(
		[]string{
			algorithm,
			dateTime,
			scope,
			hashHex([]byte(canonical)),
		},
		"\n",
	)

	key := s.signingKey(signedAt.Format(dateFormat))
	signature := hex.EncodeToString(hmacSHA256(key, []byte(stringToSign)))

	request.Header.Set("Authorization", fmt.Sprintf(
		"%s Credential=%s/%s, SignedHeaders=%s, Signature=%s",
		algorithm,
		s.Credentials.AccessKeyID,
		scope,
		signedHeaders,
		signature,
	))
}

func (s Signer) now() time.Time {
	if s.Now == nil {
		return time.Now()
	}

	return s.Now()
}

// signingKey derives the key of one day, region and service.
func (s Signer) signingKey(date string) []byte {
	key := hmacSHA256([]byte("AWS4"+s.Credentials.SecretAccessKey), []byte(date))
	key = hmacSHA256(key, []byte(s.Region))
	key = hmacSHA256(key, []byte(s.Service))

	return hmacSHA256(key, []byte(scopeTerminal))
}

// canonicalRequest returns the canonical form of request and the list of
// signed header names.
func canonicalRequest(request *http.Request, body []byte) (string, string) {
	headerLines, signedHeaders := canonicalHeaders(request)

	canonical := strings.Join(
		[]string{
			strings.ToUpper(request.Method),
			canonicalPath(request.URL),
			canonicalQuery(request.URL),
			headerLines + "\n",
			signedHeaders,
			hashHex(body),
		},
		"\n",
	)

	return canonical, signedHeaders
}

// canonicalPath encodes the escaped path again, keeping slashes.
func canonicalPath(location *url.URL) string {
	path := repeatedSlashes.ReplaceAllString(location.EscapedPath(), "/")
	if path == "" {
		return "/"
	}

	return encodeURIComponent(path, true)
}

// queryParameter is one encoded query parameter.
type queryParameter struct {
	key   string
	value string
}

// canonicalQuery encodes every parameter and sorts by the encoded key, then
// the encoded value, as SigV4 requires.
func canonicalQuery(location *url.URL) string {
	var parameters []queryParameter

	for key, values := range location.Query() {
		for _, value := range values {
			parameters = append(parameters, queryParameter{
				key:   encodeURIComponent(key, false),
				value: encodeURIComponent(value, false),
			})
		}
	}

	slices.SortFunc(parameters, compareQueryParameters)

	pairs := make([]string, 0, len(parameters))
	for _, parameter := range parameters {
		pairs = append(pairs, parameter.key+"="+parameter.value)
	}

	return strings.Join(pairs, "&")
}

func compareQueryParameters(a queryParameter, b queryParameter) int {
	if order := strings.Compare(a.key, b.key); order != 0 {
		return order
	}

	return strings.Compare(a.value, b.value)
}

// canonicalHeaders returns the sorted "name:value" lines of the signed
// headers, including host, and their semicolon-separated names.
func canonicalHeaders(request *http.Request) (string, string) {
	values := map[string]string{
		"host": requestHost(request),
	}

	for name, headerValues := range request.Header {
		lowered := strings.ToLower(name)
		if unsignableHeaders[lowered] {
			continue
		}

		value := strings.Join(headerValues, ",")
		if !isASCII(value) {
			continue
		}

		values[lowered] = strings.TrimSpace(whitespaceRun.ReplaceAllString(value, " "))
	}

	names := make([]string, 0, len(values))
	for name := range values {
		names = append(names, name)
	}

	slices.Sort(names)

	lines := make([]string, 0, len(names))
	for _, name := range names {
		lines = append(lines, name+":"+values[name])
	}

	return strings.Join(lines, "\n"), strings.Join(names, ";")
}

func requestHost(request *http.Request) string {
	if request.Host != "" {
		return request.Host
	}

	return request.URL.Host
}

// encodeURIComponent percent-encodes every byte except the RFC 3986
// unreserved characters, and slashes when keepSlashes is set.
func encodeURIComponent(value string, keepSlashes bool) string {
	var encoded strings.Builder

	for index := range len(value) {
		character := value[index]

		if isUnreserved(character) || (keepSlashes && character == '/') {
			encoded.WriteByte(character)

			continue
		}

		fmt.Fprintf(&encoded, "%%%02X", character)
	}

	return encoded.String()
}

func isUnreserved(character byte) bool {
	switch {
	case 'A' <= character && character <= 'Z':
		return true
	case 'a' <= character && character <= 'z':
		return true
	case '0' <= character && character <= '9':
		return true
	default:
		return character == '-' || character == '_' || character == '.' || character == '~'
	}
}

func isASCII(value string) bool {
	for index := range len(value) {
		if value[index] >= 0x80 {
			return false
		}
	}

	return true
}

func hashHex(data []byte) string {
	sum := sha256.Sum256(data)

	return hex.EncodeToString(sum[:])
}

func hmacSHA256(key []byte, data []byte) []byte {
	mac := hmac.New(sha256.New, key)
	mac.Write(data)

	return mac.Sum(nil)
}
