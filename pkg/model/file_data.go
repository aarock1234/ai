package model

import "net/url"

// FileData is the payload of a file: FileBytes, FileURL, FileReference or
// FileText. Generated files and reasoning files use only FileBytes or FileURL.
type FileData interface {
	isFileData()
}

// FileBytes is inline file content.
type FileBytes struct {
	Data []byte
}

// FileURL is file content that the provider or the caller downloads.
type FileURL struct {
	URL *url.URL
}

// FileReference identifies a file already uploaded to one or more providers.
// It maps a provider namespace to that provider's file ID.
type FileReference struct {
	IDs map[string]string
}

// FileText is inline text file content, such as a text/plain document.
type FileText struct {
	Text string
}

func (FileBytes) isFileData()     {}
func (FileURL) isFileData()       {}
func (FileReference) isFileData() {}
func (FileText) isFileData()      {}
