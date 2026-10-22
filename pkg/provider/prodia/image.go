package prodia

import (
	"context"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// imageAccept asks for the job record and the image as multipart parts.
const imageAccept = "multipart/form-data; image/png"

// maxLoras is the LoRA limit of one job.
const maxLoras = 3

// StylePreset is a visual theme for generated images.
type StylePreset string

// Style presets.
const (
	Style3DModel      StylePreset = "3d-model"
	StyleAnalogFilm   StylePreset = "analog-film"
	StyleAnime        StylePreset = "anime"
	StyleCinematic    StylePreset = "cinematic"
	StyleComicBook    StylePreset = "comic-book"
	StyleDigitalArt   StylePreset = "digital-art"
	StyleEnhance      StylePreset = "enhance"
	StyleFantasyArt   StylePreset = "fantasy-art"
	StyleIsometric    StylePreset = "isometric"
	StyleLineArt      StylePreset = "line-art"
	StyleLowPoly      StylePreset = "low-poly"
	StyleNeonPunk     StylePreset = "neon-punk"
	StyleOrigami      StylePreset = "origami"
	StylePhotographic StylePreset = "photographic"
	StylePixelArt     StylePreset = "pixel-art"
	StyleTexture      StylePreset = "texture"
	StyleCraftClay    StylePreset = "craft-clay"
)

// stylePresets are the accepted style presets.
var stylePresets = []StylePreset{
	Style3DModel,
	StyleAnalogFilm,
	StyleAnime,
	StyleCinematic,
	StyleComicBook,
	StyleDigitalArt,
	StyleEnhance,
	StyleFantasyArt,
	StyleIsometric,
	StyleLineArt,
	StyleLowPoly,
	StyleNeonPunk,
	StyleOrigami,
	StylePhotographic,
	StylePixelArt,
	StyleTexture,
	StyleCraftClay,
}

// ImageOptions are the provider options of image models, read from the
// "prodia" namespace.
type ImageOptions struct {
	// Steps is the number of iterations, from 1 to 4.
	Steps *int `json:"steps,omitempty"`

	// Width and Height are the output dimensions, from 256 to 1920 pixels.
	// They take precedence over the size call option.
	Width  *int `json:"width,omitempty"`
	Height *int `json:"height,omitempty"`

	// StylePreset applies a visual theme.
	StylePreset *StylePreset `json:"stylePreset,omitempty"`

	// Loras augments the output with up to three LoRA models.
	Loras []string `json:"loras,omitempty"`

	// Progressive returns a progressive JPEG.
	Progressive *bool `json:"progressive,omitempty"`
}

// ImageMetadata is the Prodia metadata of an image response, stored under
// ProviderMetadata["prodia"].
type ImageMetadata struct {
	// Images has one entry for the generated image.
	Images []JobMetadata `json:"images"`
}

// ImageModel is a Prodia image model. It generates one image per call and
// does not accept input images.
type ImageModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "prodia.image" by default.
func (m *ImageModel) Provider() string {
	return m.provider.name + ".image"
}

// ModelID returns the model identifier.
func (m *ImageModel) ModelID() string {
	return m.modelID
}

// MaxImagesPerCall returns 1.
func (m *ImageModel) MaxImagesPerCall() int {
	return 1
}

type imageJobRequest struct {
	Type   string         `json:"type"`
	Config imageJobConfig `json:"config"`
}

type imageJobConfig struct {
	Prompt      string       `json:"prompt"`
	Width       *int         `json:"width,omitempty"`
	Height      *int         `json:"height,omitempty"`
	Seed        *int         `json:"seed,omitempty"`
	Steps       *int         `json:"steps,omitempty"`
	StylePreset *StylePreset `json:"style_preset,omitempty"`
	Loras       []string     `json:"loras,omitempty"`
	Progressive *bool        `json:"progressive,omitempty"`
}

// Generate runs an image job and returns its image.
func (m *ImageModel) Generate(ctx context.Context, options model.ImageOptions) (*model.ImageResult, error) {
	timestamp := time.Now()

	imageOptions, err := readImageOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	config, warnings := imageConfig(options, imageOptions)

	request, err := m.provider.jobRequest(imageAccept, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, imageJobRequest{
		Type:   m.modelID,
		Config: config,
	})
	if err != nil {
		return nil, err
	}

	record, image, err := imageJobResult(response)
	if err != nil {
		return nil, err
	}

	return &model.ImageResult{
		Images: [][]byte{
			image,
		},
		Warnings: warnings,
		ProviderMetadata: model.ProviderMetadata{
			Namespace: ImageMetadata{
				Images: []JobMetadata{
					jobMetadata(record),
				},
			},
		},
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				Timestamp: timestamp,
				ModelID:   m.modelID,
			},
			Headers: response.Headers,
		},
	}, nil
}

func readImageOptions(options model.ProviderOptions) (ImageOptions, error) {
	imageOptions, _, err := model.ProviderValue[ImageOptions](options, Namespace)
	if err != nil {
		return ImageOptions{}, err
	}

	if err := imageOptions.validate(); err != nil {
		return ImageOptions{}, err
	}

	return imageOptions, nil
}

func (o ImageOptions) validate() error {
	if o.Steps != nil && (*o.Steps < 1 || *o.Steps > 4) {
		return fmt.Errorf("%w: prodia steps must be between 1 and 4", model.ErrInvalidArgument)
	}

	if o.Width != nil && (*o.Width < 256 || *o.Width > 1920) {
		return fmt.Errorf("%w: prodia width must be between 256 and 1920", model.ErrInvalidArgument)
	}

	if o.Height != nil && (*o.Height < 256 || *o.Height > 1920) {
		return fmt.Errorf("%w: prodia height must be between 256 and 1920", model.ErrInvalidArgument)
	}

	if o.StylePreset != nil && !slices.Contains(stylePresets, *o.StylePreset) {
		return fmt.Errorf("%w: prodia stylePreset %q", model.ErrInvalidArgument, *o.StylePreset)
	}

	if len(o.Loras) > maxLoras {
		return fmt.Errorf("%w: prodia accepts at most %d loras", model.ErrInvalidArgument, maxLoras)
	}

	return nil
}

// imageConfig builds the job configuration. Provider dimensions take
// precedence over the size, and an invalid size is ignored with a warning.
func imageConfig(options model.ImageOptions, imageOptions ImageOptions) (imageJobConfig, []model.Warning) {
	var warnings []model.Warning

	config := imageJobConfig{
		Prompt:      options.Prompt,
		Width:       imageOptions.Width,
		Height:      imageOptions.Height,
		Seed:        options.Seed,
		Steps:       imageOptions.Steps,
		StylePreset: imageOptions.StylePreset,
		Loras:       imageOptions.Loras,
		Progressive: imageOptions.Progressive,
	}

	if options.Size == "" {
		return config, warnings
	}

	width, height, ok := parseSize(options.Size)
	if !ok {
		warnings = append(warnings, model.UnsupportedWarning(
			"size",
			fmt.Sprintf("Invalid size format: %s. Expected format: WIDTHxHEIGHT (e.g., 1024x1024)", options.Size),
		))

		return config, warnings
	}

	if config.Width == nil {
		config.Width = new(width)
	}

	if config.Height == nil {
		config.Height = new(height)
	}

	return config, warnings
}

// parseSize reads a "WIDTHxHEIGHT" size.
func parseSize(size string) (int, int, bool) {
	widthText, heightText, found := strings.Cut(size, "x")
	if !found {
		return 0, 0, false
	}

	width, widthErr := strconv.Atoi(widthText)
	height, heightErr := strconv.Atoi(heightText)

	return width, height, widthErr == nil && heightErr == nil
}

// imageJobResult reads the job record and the image from the multipart
// response.
func imageJobResult(response *apicall.Response) (job, []byte, error) {
	parts, err := readJobParts(response.Headers, response.Body)
	if err != nil {
		return job{}, nil, err
	}

	var record *job
	var image []byte

	for _, part := range parts {
		switch {
		case part.name == "job":
			decoded, err := decodeJob(part)
			if err != nil {
				return job{}, nil, err
			}

			record = &decoded

		case part.name == "output", isImageOutput(part):
			image = part.body
		}
	}

	if record == nil {
		return job{}, nil, fmt.Errorf("%w: Prodia multipart response missing job part", model.ErrInvalidResponse)
	}

	if image == nil {
		return job{}, nil, fmt.Errorf("%w: Prodia multipart response missing output image", model.ErrInvalidResponse)
	}

	return *record, image, nil
}

var _ model.ImageModel = (*ImageModel)(nil)
