package download

import (
	"fmt"

	"github.com/aarock1234/ai/internal/dataurl"
)

// decodeDataURL decodes a data URL of the form
// "data:[<media type>][;base64],<data>", failing when the decoded content
// exceeds maxBytes.
func decodeDataURL(location string, maxBytes int64) (*File, error) {
	mediaType, data, err := dataurl.Parse(location)
	if err != nil {
		return nil, &Error{
			URL:   "data URL",
			Cause: err,
		}
	}

	if int64(len(data)) > maxBytes {
		return nil, &Error{
			URL:     "data URL",
			Message: fmt.Sprintf("data URL exceeded maximum size of %d bytes", maxBytes),
		}
	}

	return &File{
		Data:      data,
		MediaType: mediaType,
	}, nil
}
