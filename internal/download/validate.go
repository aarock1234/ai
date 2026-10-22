package download

import (
	"fmt"
	"net/netip"
	"net/url"
	"strings"
	"syscall"
)

var (
	privateIPv4Prefixes = mustParsePrefixes(
		"0.0.0.0/8",
		"10.0.0.0/8",
		"100.64.0.0/10",
		"127.0.0.0/8",
		"169.254.0.0/16",
		"172.16.0.0/12",
		"192.0.0.0/24",
		"192.0.2.0/24",
		"192.168.0.0/16",
		"198.18.0.0/15",
		"198.51.100.0/24",
		"203.0.113.0/24",
		"224.0.0.0/4",
		"240.0.0.0/4",
	)

	privateIPv6Prefixes = mustParsePrefixes(
		"::/128",
		"::1/128",
		"fc00::/7",
		"fe80::/10",
		"fec0::/10",
		"ff00::/8",
		"2001:db8::/32",
		"3fff::/20",
	)

	// embeddedIPv4Prefixes are IPv6 ranges whose last 32 bits carry an IPv4
	// address that must itself be public.
	embeddedIPv4Prefixes = mustParsePrefixes(
		"::/96",
		"::ffff:0:0/96",
		"::ffff:0:0:0/96",
		"64:ff9b::/96",
		"64:ff9b:1::/48",
	)

	// sixToFourPrefix is the 6to4 range, whose bits 16 to 47 carry the IPv4
	// address of the relay.
	sixToFourPrefix = netip.MustParsePrefix("2002::/16")

	// teredoPrefix is the Teredo range, which carries the server IPv4
	// address in bits 32 to 63 and the client IPv4 address, inverted, in
	// the last 32 bits.
	teredoPrefix = netip.MustParsePrefix("2001::/32")
)

// ValidateURL reports a *Error when location could reach the local machine
// or a private network: schemes other than http, https and data, empty or
// local hostnames, and private, loopback, link-local, reserved, multicast or
// documentation IP addresses. It checks only the URL; Fetch also validates
// the addresses that hostnames resolve to.
func ValidateURL(location string) error {
	parsed, err := url.Parse(location)
	if err != nil {
		return &Error{
			URL:     location,
			Message: fmt.Sprintf("invalid URL: %s", location),
			Cause:   err,
		}
	}

	return validateURL(parsed)
}

func validateURL(target *url.URL) error {
	location := target.String()
	scheme := strings.ToLower(target.Scheme)

	if scheme == "data" {
		return nil
	}

	if scheme != "http" && scheme != "https" {
		return &Error{
			URL:     location,
			Message: fmt.Sprintf("URL scheme must be http, https, or data, got %s:", target.Scheme),
		}
	}

	hostname := strings.TrimRight(strings.ToLower(target.Hostname()), ".")
	if hostname == "" {
		return &Error{
			URL:     location,
			Message: "URL must have a hostname",
		}
	}

	if isLocalHostname(hostname) {
		return &Error{
			URL:     location,
			Message: fmt.Sprintf("URL with hostname %s is not allowed", hostname),
		}
	}

	if !isDisallowedAddressLiteral(hostname) {
		return nil
	}

	return &Error{
		URL:     location,
		Message: fmt.Sprintf("URL with IP address %s is not allowed", hostname),
	}
}

func isLocalHostname(hostname string) bool {
	return hostname == "localhost" ||
		strings.HasSuffix(hostname, ".local") ||
		strings.HasSuffix(hostname, ".localhost")
}

// isDisallowedAddressLiteral reports whether hostname is a private IP
// address. An IPv6 literal that cannot be parsed fails closed.
func isDisallowedAddressLiteral(hostname string) bool {
	address, err := netip.ParseAddr(hostname)
	if err != nil {
		return strings.Contains(hostname, ":")
	}

	return isPrivateAddress(address)
}

// validateDialAddress checks the resolved address of every connection, so
// that a public hostname cannot resolve to a private address. It is the
// dialer's Control function.
func validateDialAddress(network, address string, _ syscall.RawConn) error {
	addressPort, err := netip.ParseAddrPort(address)
	if err != nil {
		return &Error{
			URL:     address,
			Message: fmt.Sprintf("cannot validate %s address %s", network, address),
			Cause:   err,
		}
	}

	if isPrivateAddress(addressPort.Addr()) {
		return &Error{
			URL:     address,
			Message: fmt.Sprintf("connection to disallowed IP address %s was blocked", addressPort.Addr()),
		}
	}

	return nil
}

// isPrivateAddress reports whether address is loopback, private, link-local,
// reserved, multicast or used for documentation, including IPv4 addresses
// embedded in IPv6 by translation, 6to4 and Teredo.
func isPrivateAddress(address netip.Addr) bool {
	address = address.WithZone("")

	if address.Is4() {
		return containsAddress(privateIPv4Prefixes, address)
	}

	if containsAddress(privateIPv6Prefixes, address) {
		return true
	}

	for _, embedded := range embeddedIPv4Addresses(address) {
		if containsAddress(privateIPv4Prefixes, embedded) {
			return true
		}
	}

	return false
}

// embeddedIPv4Addresses returns the IPv4 addresses that an IPv6 address
// routes to: the last 32 bits of mapped and translated addresses, the relay
// of a 6to4 address, and the server and client of a Teredo address.
func embeddedIPv4Addresses(address netip.Addr) []netip.Addr {
	octets := address.As16()

	switch {
	case containsAddress(embeddedIPv4Prefixes, address):
		return []netip.Addr{
			netip.AddrFrom4([4]byte(octets[12:16])),
		}

	case sixToFourPrefix.Contains(address):
		return []netip.Addr{
			netip.AddrFrom4([4]byte(octets[2:6])),
		}

	case teredoPrefix.Contains(address):
		client := [4]byte(octets[12:16])
		for index := range client {
			client[index] = ^client[index]
		}

		return []netip.Addr{
			netip.AddrFrom4([4]byte(octets[4:8])),
			netip.AddrFrom4(client),
		}

	default:
		return nil
	}
}

func containsAddress(prefixes []netip.Prefix, address netip.Addr) bool {
	for _, prefix := range prefixes {
		if prefix.Contains(address) {
			return true
		}
	}

	return false
}

func mustParsePrefixes(values ...string) []netip.Prefix {
	prefixes := make([]netip.Prefix, len(values))
	for index, value := range values {
		prefixes[index] = netip.MustParsePrefix(value)
	}

	return prefixes
}
