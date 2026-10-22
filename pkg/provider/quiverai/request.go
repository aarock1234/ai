package quiverai

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"unicode/utf16"
	"unicode/utf8"

	"github.com/aarock1234/ai/pkg/model"
)

const (
	// maxArrow2OutputTokens is the output token limit of Arrow 2 models.
	maxArrow2OutputTokens = 65536

	// maxAnimationSourceBase64Length bounds the encoded source SVG of an
	// animation.
	maxAnimationSourceBase64Length = 1_066_668

	// maxEditSVGBytes bounds the source SVG of an edit.
	maxEditSVGBytes = 200_000

	// maxEditPromptLength bounds the edit instruction, in UTF-16 code units.
	maxEditPromptLength = 4000
)

// arrow2Models support animation, editing and the lower output token limit.
var arrow2Models = []string{
	"arrow-2",
	"arrow-2-telos",
}

// arrow1Models accept fewer generation references.
var arrow1Models = []string{
	"arrow-1",
	"arrow-1.0",
	"arrow-1.1",
}

// samplingOptions are the sampling settings shared by generation and
// vectorization.
type samplingOptions struct {
	Temperature     *float64         `json:"temperature,omitempty"`
	TopP            *float64         `json:"top_p,omitempty"`
	PresencePenalty *float64         `json:"presence_penalty,omitempty"`
	MaxOutputTokens *int             `json:"max_output_tokens,omitempty"`
	ReasoningEffort *ReasoningEffort `json:"reasoning_effort,omitempty"`
	Attributes      *Attributes      `json:"attributes,omitempty"`
	Stream          bool             `json:"stream"`
}

type generateRequest struct {
	Model  string `json:"model"`
	N      int    `json:"n,omitempty"`
	Prompt string `json:"prompt"`

	samplingOptions

	Instructions *string          `json:"instructions,omitempty"`
	References   []ImageReference `json:"references,omitempty"`
}

type vectorizeRequest struct {
	Model string         `json:"model"`
	Image ImageReference `json:"image"`

	samplingOptions

	AutoCrop   *bool `json:"auto_crop,omitempty"`
	TargetSize *int  `json:"target_size,omitempty"`
}

type animateRequest struct {
	Model           string           `json:"model"`
	SVGSource       ImageReference   `json:"svg_source"`
	Prompt          string           `json:"prompt,omitempty"`
	Temperature     *float64         `json:"temperature,omitempty"`
	MaxOutputTokens *int             `json:"max_output_tokens,omitempty"`
	ReasoningEffort *ReasoningEffort `json:"reasoning_effort,omitempty"`
	Stream          bool             `json:"stream"`
}

type editRequest struct {
	Model           string           `json:"model"`
	Prompt          string           `json:"prompt"`
	SVGSource       ImageReference   `json:"svg_source"`
	ReferenceImages []ImageReference `json:"reference_images,omitempty"`
	MaxReviewSteps  *int             `json:"max_review_steps,omitempty"`
	ReasoningEffort *ReasoningEffort `json:"reasoning_effort,omitempty"`
	Settings        *editSettings    `json:"settings,omitempty"`
	Stream          bool             `json:"stream"`
}

type editSettings struct {
	MaxOutputTokens             *int     `json:"max_output_tokens,omitempty"`
	OrchestratorMaxOutputTokens *int     `json:"orchestrator_max_output_tokens,omitempty"`
	ShallowMaxOutputTokens      *int     `json:"shallow_max_output_tokens,omitempty"`
	Temperature                 *float64 `json:"temperature,omitempty"`
}

// namedOption is a provider option and whether the call set it.
type namedOption struct {
	name string
	set  bool
}

// operationPath returns the endpoint of an operation.
func operationPath(operation Operation) string {
	switch operation {
	case OperationVectorize:
		return "/svgs/vectorizations"
	case OperationEdit:
		return "/svgs/edits"
	case OperationAnimate:
		return "/svgs/animations"
	default:
		return "/svgs/generations"
	}
}

// requestBody builds the encoded request of the operation.
func requestBody(modelID string, options model.ImageOptions, imageOptions ImageOptions, operation Operation) (json.RawMessage, error) {
	if slices.Contains(arrow2Models, modelID) &&
		imageOptions.MaxOutputTokens != nil &&
		*imageOptions.MaxOutputTokens > maxArrow2OutputTokens {
		return nil, fmt.Errorf(
			"%w: maxOutputTokens: QuiverAI model %q supports at most %d output tokens",
			model.ErrInvalidArgument,
			modelID,
			maxArrow2OutputTokens,
		)
	}

	if operation != OperationEdit {
		if err := rejectEditOnlyOptions(operation, imageOptions); err != nil {
			return nil, err
		}
	}

	switch operation {
	case OperationEdit:
		return encodeBody(editBody(modelID, options, imageOptions))
	case OperationAnimate:
		return encodeBody(animateBody(modelID, options, imageOptions))
	case OperationVectorize:
		return encodeBody(vectorizeBody(modelID, options, imageOptions))
	default:
		return encodeBody(generateBody(modelID, options, imageOptions))
	}
}

// encodeBody encodes a request built without error.
func encodeBody[Request any](body Request, err error) (json.RawMessage, error) {
	if err != nil {
		return nil, err
	}

	encoded, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("encode request body: %w", err)
	}

	return encoded, nil
}

func sharedSamplingOptions(imageOptions ImageOptions) samplingOptions {
	return samplingOptions{
		Temperature:     imageOptions.Temperature,
		TopP:            imageOptions.TopP,
		PresencePenalty: imageOptions.PresencePenalty,
		MaxOutputTokens: imageOptions.MaxOutputTokens,
		ReasoningEffort: imageOptions.ReasoningEffort,
		Attributes:      imageOptions.Attributes,
	}
}

func generateBody(modelID string, options model.ImageOptions, imageOptions ImageOptions) (generateRequest, error) {
	if strings.TrimSpace(options.Prompt) == "" {
		return generateRequest{}, fmt.Errorf(
			"%w: prompt: QuiverAI image generation requires a non-empty prompt",
			model.ErrInvalidArgument,
		)
	}

	maxReferences := 16
	if slices.Contains(arrow1Models, modelID) {
		maxReferences = 4
	}

	if len(options.Files) > maxReferences {
		return generateRequest{}, fmt.Errorf(
			"%w: files: QuiverAI generate supports up to %d reference images for model %q",
			model.ErrInvalidArgument,
			maxReferences,
			modelID,
		)
	}

	references := make([]ImageReference, 0, len(options.Files))
	for _, file := range options.Files {
		reference, err := imageReference(file)
		if err != nil {
			return generateRequest{}, err
		}

		references = append(references, reference)
	}

	return generateRequest{
		Model:           modelID,
		N:               options.N,
		Prompt:          options.Prompt,
		samplingOptions: sharedSamplingOptions(imageOptions),
		Instructions:    imageOptions.Instructions,
		References:      references,
	}, nil
}

func vectorizeBody(modelID string, options model.ImageOptions, imageOptions ImageOptions) (vectorizeRequest, error) {
	if len(options.Files) == 0 {
		return vectorizeRequest{}, fmt.Errorf(
			"%w: files: QuiverAI vectorize requires an input image. Pass an input image and set the quiverai operation option to \"vectorize\"",
			model.ErrInvalidArgument,
		)
	}

	if len(options.Files) > 1 {
		return vectorizeRequest{}, fmt.Errorf("%w: files: QuiverAI vectorize accepts a single input image", model.ErrInvalidArgument)
	}

	if err := requireSingleImage(options.N, "vectorize returns one SVG per request"); err != nil {
		return vectorizeRequest{}, err
	}

	image, err := imageReference(options.Files[0])
	if err != nil {
		return vectorizeRequest{}, err
	}

	return vectorizeRequest{
		Model:           modelID,
		Image:           image,
		samplingOptions: sharedSamplingOptions(imageOptions),
		AutoCrop:        imageOptions.AutoCrop,
		TargetSize:      imageOptions.TargetSize,
	}, nil
}

func animateBody(modelID string, options model.ImageOptions, imageOptions ImageOptions) (animateRequest, error) {
	if err := validateAnimateCall(modelID, options); err != nil {
		return animateRequest{}, err
	}

	unsupported := setOptionNames([]namedOption{
		{
			name: "instructions",
			set:  imageOptions.Instructions != nil,
		},
		{
			name: "topP",
			set:  imageOptions.TopP != nil,
		},
		{
			name: "presencePenalty",
			set:  imageOptions.PresencePenalty != nil,
		},
		{
			name: "attributes",
			set:  imageOptions.Attributes != nil,
		},
		{
			name: "autoCrop",
			set:  imageOptions.AutoCrop != nil,
		},
		{
			name: "targetSize",
			set:  imageOptions.TargetSize != nil,
		},
	})

	if len(unsupported) > 0 {
		return animateRequest{}, fmt.Errorf(
			"%w: providerOptions.quiverai.%s: QuiverAI animate does not support providerOptions.quiverai.%s",
			model.ErrInvalidArgument,
			unsupported[0],
			unsupported[0],
		)
	}

	source, err := animationSource(options.Files[0])
	if err != nil {
		return animateRequest{}, err
	}

	return animateRequest{
		Model:           modelID,
		SVGSource:       source,
		Prompt:          options.Prompt,
		Temperature:     imageOptions.Temperature,
		MaxOutputTokens: imageOptions.MaxOutputTokens,
		ReasoningEffort: imageOptions.ReasoningEffort,
	}, nil
}

// validateAnimateCall checks the model, input file, image count, mask and
// prompt of an animation.
func validateAnimateCall(modelID string, options model.ImageOptions) error {
	if !slices.Contains(arrow2Models, modelID) {
		return fmt.Errorf(
			"%w: modelId: QuiverAI animate is supported by the \"arrow-2\" and \"arrow-2-telos\" models",
			model.ErrInvalidArgument,
		)
	}

	if len(options.Files) != 1 {
		return fmt.Errorf(
			"%w: files: QuiverAI animate requires exactly one source SVG input image",
			model.ErrInvalidArgument,
		)
	}

	if err := requireSingleImage(options.N, "animate returns one SVG per request"); err != nil {
		return err
	}

	if options.Mask != nil {
		return fmt.Errorf("%w: mask: QuiverAI animate does not support masks", model.ErrInvalidArgument)
	}

	isBlankPrompt := options.Prompt != "" && strings.TrimSpace(options.Prompt) == ""
	if isBlankPrompt {
		return fmt.Errorf(
			"%w: prompt: QuiverAI animate requires a non-empty prompt when an animation instruction is provided",
			model.ErrInvalidArgument,
		)
	}

	return nil
}

func editBody(modelID string, options model.ImageOptions, imageOptions ImageOptions) (editRequest, error) {
	if err := validateEditCall(modelID, options); err != nil {
		return editRequest{}, err
	}

	unsupported := setOptionNames([]namedOption{
		{
			name: "instructions",
			set:  imageOptions.Instructions != nil,
		},
		{
			name: "attributes",
			set:  imageOptions.Attributes != nil,
		},
		{
			name: "topP",
			set:  imageOptions.TopP != nil,
		},
		{
			name: "presencePenalty",
			set:  imageOptions.PresencePenalty != nil,
		},
		{
			name: "autoCrop",
			set:  imageOptions.AutoCrop != nil,
		},
		{
			name: "targetSize",
			set:  imageOptions.TargetSize != nil,
		},
	})

	if len(unsupported) > 0 {
		return editRequest{}, fmt.Errorf(
			"%w: providerOptions: QuiverAI SVG editing does not support these provider options: %s",
			model.ErrInvalidArgument,
			strings.Join(unsupported, ", "),
		)
	}

	references, err := editReferences(imageOptions.ReferenceImages)
	if err != nil {
		return editRequest{}, err
	}

	source, err := editSource(options.Files[0])
	if err != nil {
		return editRequest{}, err
	}

	body := editRequest{
		Model:           modelID,
		Prompt:          options.Prompt,
		SVGSource:       source,
		ReferenceImages: references,
		MaxReviewSteps:  imageOptions.MaxReviewSteps,
		ReasoningEffort: imageOptions.ReasoningEffort,
	}

	settings := editSettings{
		MaxOutputTokens:             imageOptions.MaxOutputTokens,
		OrchestratorMaxOutputTokens: imageOptions.OrchestratorMaxOutputTokens,
		ShallowMaxOutputTokens:      imageOptions.ShallowMaxOutputTokens,
		Temperature:                 imageOptions.Temperature,
	}

	if settings != (editSettings{}) {
		body.Settings = &settings
	}

	return body, nil
}

// validateEditCall checks the model, prompt, input file, image count and
// mask of an edit.
func validateEditCall(modelID string, options model.ImageOptions) error {
	if !slices.Contains(arrow2Models, modelID) {
		return fmt.Errorf(
			"%w: modelId: QuiverAI SVG editing is supported by the \"arrow-2\" and \"arrow-2-telos\" models",
			model.ErrInvalidArgument,
		)
	}

	if strings.TrimSpace(options.Prompt) == "" {
		return fmt.Errorf("%w: prompt: QuiverAI SVG editing requires a non-empty instruction in the prompt", model.ErrInvalidArgument)
	}

	if utf16Length(options.Prompt) > maxEditPromptLength {
		return fmt.Errorf(
			"%w: prompt: QuiverAI SVG editing instructions must contain at most %d characters",
			model.ErrInvalidArgument,
			maxEditPromptLength,
		)
	}

	if len(options.Files) != 1 {
		return fmt.Errorf("%w: files: QuiverAI SVG editing requires exactly one source SVG input image", model.ErrInvalidArgument)
	}

	if err := requireSingleImage(options.N, "SVG editing returns exactly one SVG per request"); err != nil {
		return err
	}

	if options.Mask != nil {
		return fmt.Errorf("%w: mask: QuiverAI SVG editing does not support masks", model.ErrInvalidArgument)
	}

	return nil
}

// rejectEditOnlyOptions rejects edit options in other operations.
func rejectEditOnlyOptions(operation Operation, imageOptions ImageOptions) error {
	editOnly := setOptionNames([]namedOption{
		{
			name: "referenceImages",
			set:  imageOptions.ReferenceImages != nil,
		},
		{
			name: "maxReviewSteps",
			set:  imageOptions.MaxReviewSteps != nil,
		},
		{
			name: "orchestratorMaxOutputTokens",
			set:  imageOptions.OrchestratorMaxOutputTokens != nil,
		},
		{
			name: "shallowMaxOutputTokens",
			set:  imageOptions.ShallowMaxOutputTokens != nil,
		},
	})

	if len(editOnly) == 0 {
		return nil
	}

	return fmt.Errorf(
		"%w: providerOptions: QuiverAI %s does not support these edit-only provider options: %s",
		model.ErrInvalidArgument,
		operation,
		strings.Join(editOnly, ", "),
	)
}

func setOptionNames(options []namedOption) []string {
	var names []string
	for _, option := range options {
		if option.set {
			names = append(names, option.name)
		}
	}

	return names
}

// requireSingleImage rejects calls for more than one image. Zero means the
// count was not set.
func requireSingleImage(n int, operation string) error {
	if n == 0 || n == 1 {
		return nil
	}

	return fmt.Errorf(
		"%w: n: QuiverAI %s. Set the maximum images per call to 1 to run it multiple times",
		model.ErrInvalidArgument,
		operation,
	)
}

// imageReference returns an input file as a URL or base64 reference.
func imageReference(file model.ImageFile) (ImageReference, error) {
	switch file := file.(type) {
	case model.ImageURL:
		return ImageReference{
			URL: file.URL,
		}, nil

	case model.ImageBytes:
		return ImageReference{
			Base64: base64.StdEncoding.EncodeToString(file.Data),
		}, nil

	default:
		return ImageReference{}, fmt.Errorf("%w: image file %T", model.ErrUnsupported, file)
	}
}

// animationSource returns the source SVG of an animation: an http or https
// URL, or SVG data.
func animationSource(file model.ImageFile) (ImageReference, error) {
	switch file := file.(type) {
	case model.ImageURL:
		if _, err := validateImageURL(file.URL); err != nil {
			return ImageReference{}, fmt.Errorf("%w: files: QuiverAI animate requires an HTTP or HTTPS SVG URL", model.ErrInvalidArgument)
		}

		return ImageReference{
			URL: file.URL,
		}, nil

	case model.ImageBytes:
		if !hasSVGHead(file.Data) {
			return ImageReference{}, fmt.Errorf("%w: files: QuiverAI animate requires the input file to contain SVG data", model.ErrInvalidArgument)
		}

		encoded := base64.StdEncoding.EncodeToString(file.Data)
		if len(encoded) > maxAnimationSourceBase64Length {
			return ImageReference{}, fmt.Errorf(
				"%w: files: QuiverAI animate accepts at most %d base64 characters for the source SVG",
				model.ErrInvalidArgument,
				maxAnimationSourceBase64Length,
			)
		}

		return ImageReference{
			Base64: encoded,
		}, nil

	default:
		return ImageReference{}, fmt.Errorf("%w: image file %T", model.ErrUnsupported, file)
	}
}

// editSource returns the source SVG of an edit: a normalized http or https
// URL, or a complete SVG document of at most 200 kB.
func editSource(file model.ImageFile) (ImageReference, error) {
	switch file := file.(type) {
	case model.ImageURL:
		validated, err := validateImageURL(file.URL)
		if err != nil {
			return ImageReference{}, err
		}

		return ImageReference{
			URL: validated,
		}, nil

	case model.ImageBytes:
		if err := validateEditSVG(file.Data); err != nil {
			return ImageReference{}, err
		}

		return ImageReference{
			Base64: base64.StdEncoding.EncodeToString(file.Data),
		}, nil

	default:
		return ImageReference{}, fmt.Errorf("%w: image file %T", model.ErrUnsupported, file)
	}
}

func validateEditSVG(data []byte) error {
	if len(data) == 0 || len(data) > maxEditSVGBytes {
		return fmt.Errorf("%w: files: QuiverAI SVG source data must contain 1-%d bytes", model.ErrInvalidArgument, maxEditSVGBytes)
	}

	if !utf8.Valid(data) {
		return fmt.Errorf("%w: files: QuiverAI SVG source data must be valid UTF-8", model.ErrInvalidArgument)
	}

	svg := string(data)
	if utf16Length(svg) > maxEditSVGBytes || !isSVGMarkup(svg) {
		return fmt.Errorf("%w: files: QuiverAI SVG source data must contain a complete SVG document", model.ErrInvalidArgument)
	}

	return nil
}

// editReferences validates the reference images of an edit.
func editReferences(references []ImageReference) ([]ImageReference, error) {
	if references == nil {
		return nil, nil
	}

	validated := make([]ImageReference, 0, len(references))

	for index, reference := range references {
		if reference.URL != "" {
			location, err := validateImageURL(reference.URL)
			if err != nil {
				return nil, err
			}

			validated = append(validated, ImageReference{
				URL: location,
			})

			continue
		}

		argument := fmt.Sprintf("providerOptions.quiverai.referenceImages[%d]", index)
		if err := validateReferenceBase64(reference.Base64, argument); err != nil {
			return nil, err
		}

		validated = append(validated, ImageReference{
			Base64: reference.Base64,
		})
	}

	return validated, nil
}

// utf16Length returns the length of text in UTF-16 code units, as
// JavaScript string lengths are measured.
func utf16Length(text string) int {
	return len(utf16.Encode([]rune(text)))
}
