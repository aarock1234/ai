package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openrouter"
)

// check is one feature exercised against a model. It returns a short report
// of what the model produced.
type check struct {
	name string
	run  func(ctx context.Context, languageModel model.LanguageModel) (string, error)
}

// checks run in order against every model.
var checks = []check{
	{
		name: "text",
		run:  checkText,
	},
	{
		name: "object output",
		run:  checkObject,
	},
	{
		name: "array output",
		run:  checkArray,
	},
	{
		name: "agent",
		run:  checkAgent,
	},
}

func checkText(ctx context.Context, languageModel model.LanguageModel) (string, error) {
	result, err := ai.GenerateText(ctx, ai.GenerateTextOptions{
		Model: languageModel,
		Prompt: ai.Prompt{
			System: "Answer in one short sentence.",
			Prompt: "Why do Go programs return errors instead of throwing them?",
		},
		ProviderOptions: routingOptions(),
	})
	if err != nil {
		return "", err
	}

	return report(result, result.Text), nil
}

// Release is the object output. It nests objects and arrays, restricts
// strings to enums and bounds, and has nullable and optional fields.
type Release struct {
	Product  string   `json:"product" jsonschema:"minLength=1,description=The product name"`
	Channel  string   `json:"channel" jsonschema:"enum=stable|beta|nightly"`
	Version  Version  `json:"version"`
	Codename *string  `json:"codename" jsonschema:"description=The release codename, or null when it has none"`
	Notes    string   `json:"notes,omitempty" jsonschema:"maxLength=200,description=Optional notes, left empty when there are none"`
	Changes  []Change `json:"changes" jsonschema:"minItems=1,maxItems=5"`
}

// Version is a semantic version.
type Version struct {
	Major int `json:"major" jsonschema:"minimum=0"`
	Minor int `json:"minor" jsonschema:"minimum=0"`
	Patch int `json:"patch" jsonschema:"minimum=0"`
}

// Change is one entry of a release's changelog.
type Change struct {
	Kind    string `json:"kind" jsonschema:"enum=added|changed|fixed"`
	Summary string `json:"summary" jsonschema:"description=One sentence, in the past tense"`
}

func checkObject(ctx context.Context, languageModel model.LanguageModel) (string, error) {
	result, err := ai.GenerateText(ctx, ai.GenerateTextOptions{
		Model: languageModel,
		Prompt: ai.Prompt{
			Prompt: "Describe version 2.4.1 of the Lumen photo editor, a beta release without a codename. " +
				"It added a crop tool, changed the export dialog and fixed a crash when opening large files.",
		},
		Output: ai.ObjectOutput[Release](ai.ObjectOutputOptions{
			Name:        "release",
			Description: "A software release and its changelog",
		}),
		ProviderOptions: routingOptions(),
	})
	if err != nil {
		return "", err
	}

	release, err := ai.OutputAs[Release](result.Output)
	if err != nil {
		return "", err
	}

	if release.Codename != nil {
		return "", fmt.Errorf("codename is %q, not null", *release.Codename)
	}

	summary := fmt.Sprintf(
		"%s %d.%d.%d (%s), %d changes, first %s: %s",
		release.Product,
		release.Version.Major,
		release.Version.Minor,
		release.Version.Patch,
		release.Channel,
		len(release.Changes),
		release.Changes[0].Kind,
		release.Changes[0].Summary,
	)

	return report(result, summary), nil
}

// Request is one element of the array output.
type Request struct {
	Title       string  `json:"title" jsonschema:"description=The request as a short imperative sentence"`
	Priority    string  `json:"priority" jsonschema:"enum=low|medium|high"`
	DuplicateOf *string `json:"duplicateOf" jsonschema:"description=The title of an earlier request this repeats, or null"`
}

func checkArray(ctx context.Context, languageModel model.LanguageModel) (string, error) {
	result, err := ai.GenerateText(ctx, ai.GenerateTextOptions{
		Model: languageModel,
		Prompt: ai.Prompt{
			Prompt: "List the feature requests in these messages, one element each, in order:\n" +
				"1. Please add a dark mode, my eyes hurt at night.\n" +
				"2. Could exports keep the original file name?\n" +
				"3. Dark theme please!!",
		},
		Output: ai.ArrayOutput[Request](ai.ArrayOutputOptions{
			MinItems: new(1),
			MaxItems: new(5),
			Name:     "requests",
		}),
		ProviderOptions: routingOptions(),
	})
	if err != nil {
		return "", err
	}

	requests, err := ai.OutputAs[[]Request](result.Output)
	if err != nil {
		return "", err
	}

	titles := make([]string, 0, len(requests))

	for _, request := range requests {
		title := fmt.Sprintf("%s [%s]", request.Title, request.Priority)
		if request.DuplicateOf != nil {
			title += " duplicates " + *request.DuplicateOf
		}

		titles = append(titles, title)
	}

	return report(result, strings.Join(titles, "; ")), nil
}

// report formats a result's summary with its steps, tokens and cost.
func report(result *ai.GenerateTextResult, summary string) string {
	cost, err := callCost(result.Steps)
	if err != nil {
		return fmt.Sprintf("  %s\n  cost unavailable: %v", summary, err)
	}

	return fmt.Sprintf(
		"  %s\n  steps %d, input tokens %s, output tokens %s, cost $%.6f",
		summary,
		len(result.Steps),
		tokenCount(result.Usage.InputTokens),
		tokenCount(result.Usage.OutputTokens),
		cost,
	)
}

// callCost adds the cost OpenRouter reported for each step.
func callCost(steps []ai.StepResult) (float64, error) {
	total := 0.0

	for _, step := range steps {
		metadata, found, err := model.ProviderValue[openrouter.ChatMetadata](step.ProviderMetadata, openrouter.Namespace)
		if err != nil {
			return 0, err
		}

		if !found || metadata.Usage == nil || metadata.Usage.Cost == nil {
			return 0, fmt.Errorf("step %d has no cost", step.StepNumber)
		}

		total += *metadata.Usage.Cost
	}

	return total, nil
}

func tokenCount(count *int) string {
	if count == nil {
		return "unknown"
	}

	return fmt.Sprint(*count)
}
