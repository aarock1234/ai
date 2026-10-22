package alibaba

import (
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// maxCacheBreakpoints is the number of cache markers Alibaba honors per
// request.
const maxCacheBreakpoints = 4

// cacheControls reads cache controls from provider options and counts the
// breakpoints of one request, warning when there are too many.
type cacheControls struct {
	breakpoints int
	warnings    []model.Warning
}

// read returns the cache control of the options, or nil when there is none.
// Each cache control found counts as a breakpoint.
func (c *cacheControls) read(options model.ProviderOptions) (*CacheControl, error) {
	values, _, err := model.ProviderValue[cacheControlOptions](options, Namespace)
	if err != nil {
		return nil, err
	}

	cacheControl := values.CacheControl
	if cacheControl == nil {
		cacheControl = values.SnakeCacheControl
	}

	if cacheControl == nil {
		return nil, nil
	}

	c.breakpoints++
	if c.breakpoints > maxCacheBreakpoints {
		c.warnings = append(c.warnings, model.Warning{
			Type:    model.WarningOther,
			Message: fmt.Sprintf("Max breakpoint limit exceeded. Only the last %d cache markers will take effect.", maxCacheBreakpoints),
		})
	}

	return cacheControl, nil
}
