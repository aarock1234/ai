package sandbox

import "context"

// sessionContextKey keys the sandbox session stored in a context.
type sessionContextKey struct{}

// WithSession returns a context carrying session. It is the Go replacement
// for the upstream experimental_sandbox tool execution option: callers that
// run tools inside a sandbox, such as the harness agent, attach the session
// so that tools can retrieve it with SessionFromContext.
func WithSession(ctx context.Context, session Session) context.Context {
	return context.WithValue(ctx, sessionContextKey{}, session)
}

// SessionFromContext returns the sandbox session attached with WithSession,
// and false when there is none.
func SessionFromContext(ctx context.Context) (Session, bool) {
	session, ok := ctx.Value(sessionContextKey{}).(Session)

	return session, ok && session != nil
}
