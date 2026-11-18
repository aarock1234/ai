// Command decide classifies customer messages with Decide and OpenAI's
// Decisions API: a choice question picks each message's kind, and a
// boolean question estimates whether it asks for something new. It prints
// each answer with its probabilities and the call's usage.
//
// Usage:
//
//	OPENAI_API_KEY=... go run ./examples/decide gpt-6-luna
package main

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"slices"
	"strings"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openai"
)

// messages are the customer messages to classify.
var messages = []string{
	"Please add a dark mode, my eyes hurt at night.",
	"The app crashes every time I import proxies.",
	"Is there a way to export my tasks?",
	"gm everyone, hit two today!",
}

// kinds are the choices of the kind question.
var kinds = []model.DecisionChoice{
	{
		Name:        "feature_request",
		Description: model.DecisionText("Asks for something new, or a change to how something works."),
	},
	{
		Name:        "bug",
		Description: model.DecisionText("Something that should work doesn't."),
	},
	{
		Name:        "question",
		Description: model.DecisionText("Asks how to do something."),
	},
	{
		Name:        "none",
		Description: model.DecisionText("Asks nothing of the product: greetings, success posts, chatter."),
	},
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	if len(os.Args) != 2 {
		return errors.New("name one OpenAI decision model, such as gpt-6-luna")
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	decisionModel := openai.New().Decision(os.Args[1])

	for _, message := range messages {
		if err := classify(ctx, decisionModel, message); err != nil {
			return err
		}
	}

	return nil
}

// classify asks both questions about one message and prints the answers.
func classify(ctx context.Context, decisionModel model.DecisionModel, message string) error {
	result, err := ai.Decide(ctx, ai.DecideOptions{
		Model: decisionModel,
		State: []model.DecisionStatePart{
			model.TextPart{
				Text: message,
			},
		},
		Questions: []model.DecisionQuestion{
			model.ChoiceQuestion{
				ID:           "kind",
				Instructions: model.DecisionText("What does this message from a customer of a desktop app ask of the product?"),
				Choices:      kinds,
			},
			model.BooleanQuestion{
				ID:           "new",
				Instructions: model.DecisionText("The message asks for something the product should add or change."),
			},
		},
	})
	if err != nil {
		return fmt.Errorf("decide %q: %w", message, err)
	}

	kind, ok := result.Answers["kind"].(model.ChoiceAnswer)
	if !ok {
		return fmt.Errorf("decide %q: the kind answer is %T", message, result.Answers["kind"])
	}

	isNew, ok := result.Answers["new"].(model.BooleanAnswer)
	if !ok {
		return fmt.Errorf("decide %q: the new answer is %T", message, result.Answers["new"])
	}

	fmt.Printf("%q\n  kind %s %s, asks for something new %.2f, %s\n", message, kind.Choice, probabilities(kind), isNew.Probability, tokens(result.Usage))

	return nil
}

// probabilities lists a choice answer's distribution, most likely first.
func probabilities(answer model.ChoiceAnswer) string {
	names := make([]string, 0, len(answer.Probabilities))
	for name := range answer.Probabilities {
		names = append(names, name)
	}

	slices.SortFunc(names, func(a, b string) int {
		return cmp.Compare(answer.Probabilities[b], answer.Probabilities[a])
	})

	listed := make([]string, 0, len(names))
	for _, name := range names {
		listed = append(listed, fmt.Sprintf("%s %.2f", name, answer.Probabilities[name]))
	}

	return "(" + strings.Join(listed, ", ") + ")"
}

// tokens reads as "input 120, output 4 tokens", with "?" for an unknown
// count.
func tokens(usage ai.DecisionUsage) string {
	return fmt.Sprintf("input %s, output %s tokens", count(usage.InputTokens), count(usage.OutputTokens))
}

func count(value *int) string {
	if value == nil {
		return "?"
	}

	return fmt.Sprint(*value)
}
