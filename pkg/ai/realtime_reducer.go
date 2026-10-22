package ai

import (
	"encoding/json"
	"fmt"
	"slices"

	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/pkg/model"
)

// realtimeEffect is work a server event asks the session to do:
// realtimeToolCallEffect or realtimeErrorEffect.
type realtimeEffect interface {
	isRealtimeEffect()
}

// realtimeToolCallEffect asks the session to run a function call.
type realtimeToolCallEffect struct {
	call RealtimeToolCall
}

// realtimeErrorEffect reports a non-fatal error.
type realtimeErrorEffect struct {
	err error
}

func (realtimeToolCallEffect) isRealtimeEffect() {}
func (realtimeErrorEffect) isRealtimeEffect()    {}

// realtimePartLocation locates a streaming text part.
type realtimePartLocation struct {
	messageID string
	partIndex int
}

// realtimeReducer rebuilds the conversation of a realtime session from its
// server events. The session guards it with its lock.
type realtimeReducer struct {
	maxEvents int

	messages []RealtimeMessage
	events   []model.RealtimeServerEvent

	// currentAssistantMessageID is the message that receives the output of
	// the current response, or empty before its first output.
	currentAssistantMessageID string

	textByItem          map[string]string
	textLocationByItem  map[string]realtimePartLocation
	argumentsByCall     map[string]string
	messageIDByCall     map[string]string
	toolNameByCall      map[string]string
	insertIndexByItemID map[string]int
}

func newRealtimeReducer(maxEvents int) *realtimeReducer {
	return &realtimeReducer{
		maxEvents:           maxEvents,
		textByItem:          make(map[string]string),
		textLocationByItem:  make(map[string]realtimePartLocation),
		argumentsByCall:     make(map[string]string),
		messageIDByCall:     make(map[string]string),
		toolNameByCall:      make(map[string]string),
		insertIndexByItemID: make(map[string]int),
	}
}

// reduce applies one server event and returns the work it asks for.
func (r *realtimeReducer) reduce(event model.RealtimeServerEvent) []realtimeEffect {
	r.events = lastItems(append(r.events, event), r.maxEvents)

	switch typed := event.(type) {
	case model.RealtimeAudioCommitted:
		if typed.ItemID != "" {
			r.insertIndexByItemID[typed.ItemID] = len(r.messages)
		}

	case model.RealtimeAudioTranscriptDelta:
		r.appendText(typed.ItemID, typed.Delta)

	case model.RealtimeTextDelta:
		r.appendText(typed.ItemID, typed.Delta)

	case model.RealtimeAudioTranscriptDone:
		r.finalizeText(typed.ItemID, typed.Transcript)

	case model.RealtimeTextDone:
		r.finalizeText(typed.ItemID, typed.Text)

	case model.RealtimeInputTranscriptionCompleted:
		r.addInputTranscription(typed.ItemID, typed.Transcript)

	case model.RealtimeResponseCreated, model.RealtimeResponseDone, model.RealtimeSpeechStarted:
		r.currentAssistantMessageID = ""

	case model.RealtimeFunctionCallArgumentsDelta:
		r.appendArguments(typed)

	case model.RealtimeFunctionCallArgumentsDone:
		return r.completeFunctionCall(typed)

	case model.RealtimeError:
		return []realtimeEffect{
			realtimeErrorEffect{
				err: &RealtimeProviderError{
					Message:       typed.Message,
					Code:          typed.Code,
					ClientEventID: typed.ClientEventID,
				},
			},
		}
	}

	return nil
}

// addUserText records a text message the client sends and returns its ID.
func (r *realtimeReducer) addUserText(text string) string {
	messageID := id.New("user")

	r.messages = append(r.messages, RealtimeMessage{
		ID:   messageID,
		Role: RealtimeUser,
		Parts: []RealtimeMessagePart{
			RealtimeTextPart{
				Text: text,
			},
		},
	})

	return messageID
}

// removeMessage drops a recorded message, such as a user text message that
// could not be sent, and shifts the insert positions after it.
func (r *realtimeReducer) removeMessage(messageID string) {
	index := slices.IndexFunc(r.messages, func(message RealtimeMessage) bool {
		return message.ID == messageID
	})
	if index < 0 {
		return
	}

	r.messages = slices.Delete(r.messages, index, index+1)

	for itemID, insertIndex := range r.insertIndexByItemID {
		if insertIndex > index {
			r.insertIndexByItemID[itemID] = insertIndex - 1
		}
	}
}

// toolName returns the function name of a call.
func (r *realtimeReducer) toolName(callID string) string {
	return r.toolNameByCall[callID]
}

// addToolOutput records the result of a function call.
func (r *realtimeReducer) addToolOutput(callID string, output any) {
	r.updateToolPart(callID, func(part *RealtimeToolPart) {
		part.State = RealtimeToolOutputReady
		part.Output = output
	})
}

// assistantMessage returns the message of the current response, creating
// it on the first output.
func (r *realtimeReducer) assistantMessage() string {
	if r.currentAssistantMessageID != "" {
		return r.currentAssistantMessageID
	}

	r.currentAssistantMessageID = id.New("assistant")
	r.messages = append(r.messages, RealtimeMessage{
		ID:   r.currentAssistantMessageID,
		Role: RealtimeAssistant,
	})

	return r.currentAssistantMessageID
}

func (r *realtimeReducer) appendText(itemID, delta string) {
	messageID := r.assistantMessage()

	text := r.textByItem[itemID] + delta
	r.textByItem[itemID] = text

	part := RealtimeTextPart{
		Text:      text,
		Streaming: true,
	}

	if location, ok := r.textLocationByItem[itemID]; ok {
		r.setPart(location, part)

		return
	}

	message := r.message(messageID)
	if message == nil {
		return
	}

	r.textLocationByItem[itemID] = realtimePartLocation{
		messageID: messageID,
		partIndex: len(message.Parts),
	}

	message.Parts = append(message.Parts, part)
}

// finalizeText completes the text of an item with the final text, or with
// the accumulated text when the provider did not repeat it.
func (r *realtimeReducer) finalizeText(itemID string, finalText *string) {
	text := r.textByItem[itemID]
	if finalText != nil {
		text = *finalText
	}

	delete(r.textByItem, itemID)

	location, ok := r.textLocationByItem[itemID]
	if !ok {
		return
	}

	delete(r.textLocationByItem, itemID)

	r.setPart(location, RealtimeTextPart{
		Text: text,
	})
}

// addInputTranscription records the transcript of the user's audio before
// the messages that followed its commit.
func (r *realtimeReducer) addInputTranscription(itemID, transcript string) {
	messageID := "user-" + itemID
	parts := []RealtimeMessagePart{
		RealtimeTextPart{
			Text: transcript,
		},
	}

	if existing := r.message(messageID); existing != nil {
		existing.Parts = parts

		return
	}

	insertIndex, ok := r.insertIndexByItemID[itemID]
	if !ok || insertIndex > len(r.messages) {
		insertIndex = len(r.messages)
	}

	r.messages = slices.Insert(r.messages, insertIndex, RealtimeMessage{
		ID:    messageID,
		Role:  RealtimeUser,
		Parts: parts,
	})
}

func (r *realtimeReducer) appendArguments(event model.RealtimeFunctionCallArgumentsDelta) {
	messageID := r.assistantMessage()
	r.messageIDByCall[event.CallID] = messageID
	r.argumentsByCall[event.CallID] += event.Delta

	message := r.message(messageID)
	if message == nil || r.hasToolPart(message, event.CallID) {
		return
	}

	message.Parts = append(message.Parts, RealtimeToolPart{
		ToolCallID: event.CallID,
		State:      RealtimeToolInputStreaming,
	})
}

// completeFunctionCall marks the input available and asks the session to
// run the call. Arguments that are not JSON are reported instead.
func (r *realtimeReducer) completeFunctionCall(event model.RealtimeFunctionCallArgumentsDone) []realtimeEffect {
	delete(r.argumentsByCall, event.CallID)
	r.toolNameByCall[event.CallID] = event.Name

	isValid := json.Valid([]byte(event.Arguments))

	input := json.RawMessage(event.Arguments)
	if !isValid {
		input = json.RawMessage(`{}`)
	}

	r.updateToolPart(event.CallID, func(part *RealtimeToolPart) {
		part.ToolName = event.Name
		part.State = RealtimeToolInputAvailable
		part.Input = input
	})

	if !isValid {
		return []realtimeEffect{
			realtimeErrorEffect{
				err: fmt.Errorf("%w: failed to parse tool arguments: %s", model.ErrInvalidResponse, event.Arguments),
			},
		}
	}

	return []realtimeEffect{
		realtimeToolCallEffect{
			call: RealtimeToolCall{
				ToolCallID: event.CallID,
				ToolName:   event.Name,
				Input:      input,
			},
		},
	}
}

func (r *realtimeReducer) updateToolPart(callID string, update func(part *RealtimeToolPart)) {
	message := r.message(r.messageIDByCall[callID])
	if message == nil {
		return
	}

	for index, part := range message.Parts {
		toolPart, ok := part.(RealtimeToolPart)
		if !ok || toolPart.ToolCallID != callID {
			continue
		}

		update(&toolPart)
		message.Parts[index] = toolPart
	}
}

func (r *realtimeReducer) hasToolPart(message *RealtimeMessage, callID string) bool {
	return slices.ContainsFunc(message.Parts, func(part RealtimeMessagePart) bool {
		toolPart, ok := part.(RealtimeToolPart)

		return ok && toolPart.ToolCallID == callID
	})
}

func (r *realtimeReducer) setPart(location realtimePartLocation, part RealtimeMessagePart) {
	message := r.message(location.messageID)
	if message == nil || location.partIndex >= len(message.Parts) {
		return
	}

	message.Parts[location.partIndex] = part
}

// message returns the message with the ID, or nil.
func (r *realtimeReducer) message(messageID string) *RealtimeMessage {
	if messageID == "" {
		return nil
	}

	for index := range r.messages {
		if r.messages[index].ID == messageID {
			return &r.messages[index]
		}
	}

	return nil
}

// snapshot copies the conversation and the events.
func (r *realtimeReducer) snapshot() ([]RealtimeMessage, []model.RealtimeServerEvent) {
	messages := make([]RealtimeMessage, len(r.messages))
	for index, message := range r.messages {
		message.Parts = slices.Clone(message.Parts)
		messages[index] = message
	}

	return messages, slices.Clone(r.events)
}
