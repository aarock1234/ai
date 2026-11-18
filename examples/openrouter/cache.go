package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openrouter"
)

// cacheSessionID names the prompt cache check's conversation, so both of
// its calls reach the same upstream provider.
const cacheSessionID = "ai-sdk-go-examples-cache"

// handbook is a long, unchanging system prompt: the prefix both calls of
// the cache check share. It is longer than every model's minimum cacheable
// prompt.
var handbook = "You answer questions about the store's handbook below, in one sentence.\n\n" +
	strings.Repeat("Returns are accepted within 30 days with a receipt. Store credit is issued without one. Gift cards never expire. ", 120)

// checkCache holds a two-turn conversation in one session about the
// handbook, with automatic prompt caching. The second turn extends the
// first, so it should read the first turn's prompt from the cache.
func checkCache(ctx context.Context, languageModel model.LanguageModel) (string, error) {
	var conversation []model.Message
	var last *ai.GenerateTextResult

	for _, question := range []string{"How long do I have to return something?", "Do gift cards expire?"} {
		conversation = append(conversation, model.UserMessage{
			Content: []model.UserPart{
				model.TextPart{
					Text: question,
				},
			},
		})

		result, err := ai.GenerateText(ctx, ai.GenerateTextOptions{
			Model: languageModel,
			Prompt: ai.Prompt{
				System:   handbook,
				Messages: conversation,
			},
			ProviderOptions: cachedOptions(),
		})
		if err != nil {
			return "", err
		}

		conversation = append(conversation, result.ResponseMessages...)
		last = result
	}

	read := last.Usage.InputTokenDetails.CacheReadTokens
	if read == nil || *read == 0 {
		return "", fmt.Errorf("the second call read no cached tokens of %s", tokenCount(last.Usage.InputTokens))
	}

	summary := fmt.Sprintf("second call read %d of %s input tokens from the cache", *read, tokenCount(last.Usage.InputTokens))

	return report(last, summary), nil
}

// cachedOptions are routingOptions with automatic prompt caching in one
// session.
func cachedOptions() model.ProviderOptions {
	chatOptions := routingChatOptions()
	chatOptions.CacheControl = &openrouter.CacheControl{
		Type: openrouter.CacheEphemeral,
	}
	chatOptions.SessionID = cacheSessionID

	return model.ProviderOptions{
		openrouter.Namespace: chatOptions,
	}
}
