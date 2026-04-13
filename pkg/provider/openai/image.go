package openai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"path/filepath"

	"github.com/aarock1234/ai/internal/apierror"
	"github.com/aarock1234/ai/internal/provideropts"
	"github.com/aarock1234/ai/pkg/model"
)

// Compile-time check that ImageModel implements model.ImageModel.
var _ model.ImageModel = (*ImageModel)(nil)

// ImageModel is an OpenAI image generation model.
type ImageModel struct {
	modelID      string
	clientConfig *ClientConfig
}

// Provider returns "openai".
func (m *ImageModel) Provider() string { return "openai" }

// ModelID returns the model identifier.
func (m *ImageModel) ModelID() string { return m.modelID }

type imageGenerationRequest struct {
	Model             string `json:"model"`
	Prompt            string `json:"prompt"`
	Size              string `json:"size,omitempty"`
	Quality           string `json:"quality,omitempty"`
	Background        string `json:"background,omitempty"`
	OutputFormat      string `json:"output_format,omitempty"`
	OutputCompression *int   `json:"output_compression,omitempty"`
	N                 int    `json:"n,omitempty"`
}

type imageGenerationResponse struct {
	Data  []imageResult `json:"data"`
	Usage *imageUsage   `json:"usage,omitempty"`
}

type imageResult struct {
	URL           string `json:"url,omitempty"`
	B64JSON       string `json:"b64_json,omitempty"`
	RevisedPrompt string `json:"revised_prompt,omitempty"`
}

type imageUsage struct {
	InputTokens  int `json:"input_tokens,omitempty"`
	OutputTokens int `json:"output_tokens,omitempty"`
}

// DoGenerateImage performs an OpenAI image generation or edit request.
func (m *ImageModel) DoGenerateImage(ctx context.Context, req *model.ImageRequest) (*model.ImageResponse, error) {
	if req == nil {
		req = &model.ImageRequest{}
	}

	if len(req.Images) == 0 {
		return m.generateImage(ctx, req)
	}

	return m.editImage(ctx, req)
}

func (m *ImageModel) generateImage(ctx context.Context, req *model.ImageRequest) (*model.ImageResponse, error) {
	body, err := provideropts.JSONBody(imageGenerationRequest{
		Model:             m.modelID,
		Prompt:            req.Prompt,
		Size:              req.Size,
		Quality:           req.Quality,
		Background:        req.Background,
		OutputFormat:      req.OutputFormat,
		OutputCompression: req.OutputCompress,
		N:                 req.Count,
	}, "openai", req.ProviderOptions,
		"model",
		"prompt",
		"size",
		"quality",
		"background",
		"output_format",
		"output_compression",
		"n",
	)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, m.clientConfig.BaseURL+"/images/generations", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}

	setHeaders(httpReq, m.clientConfig, req.Headers)

	resp, err := m.clientConfig.httpClient().Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("executing request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	return decodeImageResponse(resp, req.OutputFormat)
}

func (m *ImageModel) editImage(ctx context.Context, req *model.ImageRequest) (*model.ImageResponse, error) {
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)

	if err := writer.WriteField("model", m.modelID); err != nil {
		return nil, fmt.Errorf("writing model field: %w", err)
	}
	if req.Prompt != "" {
		if err := writer.WriteField("prompt", req.Prompt); err != nil {
			return nil, fmt.Errorf("writing prompt field: %w", err)
		}
	}
	if req.Size != "" {
		if err := writer.WriteField("size", req.Size); err != nil {
			return nil, fmt.Errorf("writing size field: %w", err)
		}
	}
	if req.Quality != "" {
		if err := writer.WriteField("quality", req.Quality); err != nil {
			return nil, fmt.Errorf("writing quality field: %w", err)
		}
	}
	if req.Background != "" {
		if err := writer.WriteField("background", req.Background); err != nil {
			return nil, fmt.Errorf("writing background field: %w", err)
		}
	}
	if req.OutputFormat != "" {
		if err := writer.WriteField("output_format", req.OutputFormat); err != nil {
			return nil, fmt.Errorf("writing output_format field: %w", err)
		}
	}
	if req.OutputCompress != nil {
		if err := writer.WriteField("output_compression", fmt.Sprintf("%d", *req.OutputCompress)); err != nil {
			return nil, fmt.Errorf("writing output_compression field: %w", err)
		}
	}
	if req.Count > 0 {
		if err := writer.WriteField("n", fmt.Sprintf("%d", req.Count)); err != nil {
			return nil, fmt.Errorf("writing n field: %w", err)
		}
	}

	for i, image := range req.Images {
		if len(image.Data) == 0 {
			return nil, fmt.Errorf("image edits require inline image bytes")
		}

		filename := image.Filename
		if filename == "" {
			filename = fmt.Sprintf("image-%d%s", i+1, extensionForMIMEType(image.MIMEType))
		}

		part, err := writer.CreateFormFile("image", filename)
		if err != nil {
			return nil, fmt.Errorf("creating image form file: %w", err)
		}

		if _, err := part.Write(image.Data); err != nil {
			return nil, fmt.Errorf("writing image data: %w", err)
		}
	}

	if err := provideropts.WriteMultipartFields(writer, "openai", req.ProviderOptions,
		"model",
		"prompt",
		"size",
		"quality",
		"background",
		"output_format",
		"output_compression",
		"n",
		"image",
	); err != nil {
		return nil, err
	}

	if err := writer.Close(); err != nil {
		return nil, fmt.Errorf("closing multipart writer: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, m.clientConfig.BaseURL+"/images/edits", bytes.NewReader(body.Bytes()))
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}

	httpReq.Header.Set("Content-Type", writer.FormDataContentType())
	if m.clientConfig.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+m.clientConfig.APIKey)
	}
	for key, values := range m.clientConfig.ExtraHeaders {
		for _, value := range values {
			httpReq.Header.Set(key, value)
		}
	}
	for key, values := range req.Headers {
		for _, value := range values {
			httpReq.Header.Set(key, value)
		}
	}

	resp, err := m.clientConfig.httpClient().Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("executing request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	return decodeImageResponse(resp, req.OutputFormat)
}

func decodeImageResponse(resp *http.Response, outputFormat string) (*model.ImageResponse, error) {
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, apierror.New(resp.StatusCode, string(body))
	}

	var result imageGenerationResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}

	images := make([]model.GeneratedImage, 0, len(result.Data))
	for _, item := range result.Data {
		image := model.GeneratedImage{
			URL:           item.URL,
			RevisedPrompt: item.RevisedPrompt,
		}

		if item.B64JSON != "" {
			data, err := base64.StdEncoding.DecodeString(item.B64JSON)
			if err != nil {
				return nil, fmt.Errorf("decoding image data: %w", err)
			}
			image.Data = data
			image.MIMEType = mimeTypeForOutputFormat(outputFormat)
		}

		images = append(images, image)
	}

	response := &model.ImageResponse{
		Images: images,
	}
	if result.Usage != nil {
		response.Usage = model.Usage{
			InputTokens:  result.Usage.InputTokens,
			OutputTokens: result.Usage.OutputTokens,
		}
	}

	return response, nil
}

func extensionForMIMEType(mimeType string) string {
	switch mimeType {
	case "image/png":
		return ".png"
	case "image/jpeg":
		return ".jpg"
	case "image/webp":
		return ".webp"
	default:
		ext := filepath.Ext(mimeType)
		if ext != "" {
			return ext
		}

		return ".png"
	}
}

func mimeTypeForOutputFormat(outputFormat string) string {
	switch outputFormat {
	case "jpeg", "jpg":
		return "image/jpeg"
	case "webp":
		return "image/webp"
	default:
		return "image/png"
	}
}
