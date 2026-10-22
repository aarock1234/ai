package devtools

import (
	"encoding/json"
	"net/http"
	"slices"
)

// firstMessageLength is the number of characters of the user message that
// the run list shows.
const firstMessageLength = 60

// runSummary is a root run in the run list.
type runSummary struct {
	run

	StepCount    int       `json:"stepCount"`
	FirstMessage string    `json:"firstMessage"`
	HasError     bool      `json:"hasError"`
	IsInProgress bool      `json:"isInProgress"`
	Type         *stepType `json:"type,omitempty"`
}

// runStatus is a run with its progress.
type runStatus struct {
	run

	IsInProgress bool `json:"isInProgress"`
}

// runDetail is a run with its steps and descendant runs.
type runDetail struct {
	Run       runStatus   `json:"run"`
	Steps     []step      `json:"steps"`
	ChildRuns []runDetail `json:"childRuns"`
}

func (s *Server) handleRuns(w http.ResponseWriter, _ *http.Request) {
	db := s.readViewedDatabase()
	summaries := []runSummary{}

	for _, candidate := range db.runsByStartDescending() {
		if candidate.ParentRunID != nil {
			continue
		}

		summaries = append(summaries, summarizeRun(candidate, db.stepsForRun(candidate.ID)))
	}

	writeJSON(w, http.StatusOK, summaries)
}

func (s *Server) handleRun(w http.ResponseWriter, r *http.Request) {
	db := s.readViewedDatabase()

	found, ok := db.findRun(r.PathValue("id"))
	if !ok {
		writeJSON(w, http.StatusNotFound, errorResponse{
			Error: "Run not found",
		})

		return
	}

	writeJSON(w, http.StatusOK, describeRun(db, found))
}

// summarizeRun describes a root run for the run list.
func summarizeRun(summarized run, steps []step) runSummary {
	summary := runSummary{
		run:          summarized,
		StepCount:    len(steps),
		FirstMessage: "No user message",
	}

	if len(steps) == 0 {
		return summary
	}

	if message, ok := lastUserMessageText(steps[0].Input); ok {
		summary.FirstMessage = truncateMessage(message)
	}

	summary.HasError = slices.ContainsFunc(steps, func(candidate step) bool {
		return candidate.Error != nil
	})
	summary.IsInProgress = hasStepInProgress(steps)
	summary.Type = &steps[0].Type

	return summary
}

// describeRun returns a run with its steps and, recursively, its child
// runs.
func describeRun(db *database, described run) runDetail {
	steps := db.stepsForRun(described.ID)
	if steps == nil {
		steps = []step{}
	}

	childRuns := []runDetail{}
	for _, candidate := range db.runsByStartDescending() {
		if candidate.ParentRunID != nil && *candidate.ParentRunID == described.ID {
			childRuns = append(childRuns, describeRun(db, candidate))
		}
	}

	return runDetail{
		Run: runStatus{
			run:          described,
			IsInProgress: hasStepInProgress(steps),
		},
		Steps:     steps,
		ChildRuns: childRuns,
	}
}

func hasStepInProgress(steps []step) bool {
	return slices.ContainsFunc(steps, step.isInProgress)
}

// recordedPrompt is the part of a recorded step input that the run list
// reads.
type recordedPrompt struct {
	Prompt []struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	} `json:"prompt"`
}

// lastUserMessageText returns the text of the last user message of a
// recorded step input: its content when it is a string, and otherwise the
// text of its first part.
func lastUserMessageText(input string) (string, bool) {
	var recorded recordedPrompt
	if err := json.Unmarshal([]byte(input), &recorded); err != nil {
		return "", false
	}

	for _, message := range slices.Backward(recorded.Prompt) {
		if message.Role != "user" {
			continue
		}

		return messageContentText(message.Content), true
	}

	return "", false
}

func messageContentText(content json.RawMessage) string {
	var text string
	if err := json.Unmarshal(content, &text); err == nil {
		return text
	}

	var parts []struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(content, &parts); err != nil || len(parts) == 0 {
		return ""
	}

	return parts[0].Text
}

func truncateMessage(message string) string {
	characters := []rune(message)
	if len(characters) <= firstMessageLength {
		return message
	}

	return string(characters[:firstMessageLength]) + "..."
}
