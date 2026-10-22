package sandboxshell

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"mvdan.cc/sh/v3/pattern"
)

// walkDecision tells a walk how to continue after visiting a node.
type walkDecision int

// Walk decisions.
const (
	continueWalk walkDecision = iota
	skipDirectory
)

// walkFunc visits one node of a walk; depth is zero for the root.
type walkFunc func(name string, info fs.FileInfo, depth int) walkDecision

// walk visits root and everything below it in preorder, children in name
// order. Names extend the root operand as written.
func (c *commandCall) walk(ctx context.Context, root string, visit walkFunc) error {
	info, err := c.fs().Stat(c.resolve(root))
	if err != nil {
		return err
	}

	return c.walkNode(ctx, root, info, 0, visit)
}

// walkNode visits one node and, unless skipped, its children.
func (c *commandCall) walkNode(ctx context.Context, name string, info fs.FileInfo, depth int, visit walkFunc) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	if visit(name, info, depth) == skipDirectory || !info.IsDir() {
		return nil
	}

	entries, err := c.fs().ReadDir(c.resolve(name))
	if err != nil {
		return err
	}

	for _, entry := range entries {
		childInfo, err := entry.Info()
		if err != nil {
			return err
		}

		if err := c.walkNode(ctx, joinWalkName(name, entry.Name()), childInfo, depth+1, visit); err != nil {
			return err
		}
	}

	return nil
}

// joinWalkName appends a child name to a walked name.
func joinWalkName(name string, child string) string {
	if strings.HasSuffix(name, "/") {
		return name + child
	}

	return name + "/" + child
}

// findEntry is one node find evaluates.
type findEntry struct {
	name  string
	info  fs.FileInfo
	depth int
}

// findExpression is a node of find's expression tree.
type findExpression interface {
	evaluate(ctx context.Context, run *findRun, entry findEntry) bool
}

// findRun is the state of one find invocation.
type findRun struct {
	call     *commandCall
	minDepth int
	maxDepth int
	ok       bool

	// batches collects the names of "-exec ... {} +" actions.
	batches map[*findExec][]string
}

// runFind carries out "find [path...] [expression]".
func runFind(ctx context.Context, call *commandCall) int {
	roots, expressionArgs := splitFindArgs(call.args)

	run := &findRun{
		call:     call,
		maxDepth: -1,
		ok:       true,
		batches:  make(map[*findExec][]string),
	}

	parser := findParser{
		run:  run,
		args: expressionArgs,
	}

	expression, err := parser.parse()
	if err != nil {
		return call.fail("%v", err)
	}

	if !parser.hasAction {
		expression = findAnd{
			left:  expression,
			right: findPrint{},
		}
	}

	entries := run.collect(ctx, roots)
	if parser.deletes {
		slices.Reverse(entries)
	}

	for _, entry := range entries {
		if ctx.Err() != nil {
			return exitInterrupted
		}

		if entry.depth >= run.minDepth {
			expression.evaluate(ctx, run, entry)
		}
	}

	run.flushBatches(ctx)

	return statusOf(run.ok)
}

// splitFindArgs splits find's arguments into start paths and the
// expression.
func splitFindArgs(args []string) ([]string, []string) {
	index := 0
	for index < len(args) && !isFindExpressionStart(args[index]) {
		index++
	}

	roots := args[:index]
	if len(roots) == 0 {
		roots = []string{
			".",
		}
	}

	return roots, args[index:]
}

// isFindExpressionStart reports whether an argument starts the expression.
func isFindExpressionStart(arg string) bool {
	return strings.HasPrefix(arg, "-") && arg != "-" || arg == "(" || arg == "!"
}

// collect walks the start paths within the depth limits.
func (r *findRun) collect(ctx context.Context, roots []string) []findEntry {
	var entries []findEntry

	for _, root := range roots {
		err := r.call.walk(ctx, root, func(name string, info fs.FileInfo, depth int) walkDecision {
			entries = append(entries, findEntry{
				name:  name,
				info:  info,
				depth: depth,
			})

			if r.maxDepth >= 0 && depth >= r.maxDepth {
				return skipDirectory
			}

			return continueWalk
		})
		if err != nil && !errors.Is(err, context.Canceled) {
			r.call.report("'%s': %s", root, describeError(err))
			r.ok = false
		}
	}

	return entries
}

// flushBatches runs the "-exec ... {} +" actions with their collected
// names.
func (r *findRun) flushBatches(ctx context.Context) {
	for exec, names := range r.batches {
		if len(names) == 0 {
			continue
		}

		if exec.runWith(ctx, r, names) != 0 {
			r.ok = false
		}
	}
}

// findParser parses find's expression.
type findParser struct {
	run       *findRun
	args      []string
	position  int
	hasAction bool
	deletes   bool
}

// parse parses the whole expression; an empty expression is true.
func (p *findParser) parse() (findExpression, error) {
	if len(p.args) == 0 {
		return findConstant(true), nil
	}

	expression, err := p.parseOr()
	if err != nil {
		return nil, err
	}

	if p.position < len(p.args) {
		return nil, fmt.Errorf("unexpected argument '%s'", p.args[p.position])
	}

	return expression, nil
}

// parseOr parses alternatives joined by -o.
func (p *findParser) parseOr() (findExpression, error) {
	left, err := p.parseAnd()
	if err != nil {
		return nil, err
	}

	for p.accept("-o", "-or") {
		right, err := p.parseAnd()
		if err != nil {
			return nil, err
		}

		left = findOr{
			left:  left,
			right: right,
		}
	}

	return left, nil
}

// parseAnd parses terms joined by -a or juxtaposition.
func (p *findParser) parseAnd() (findExpression, error) {
	left, err := p.parseUnary()
	if err != nil {
		return nil, err
	}

	for p.position < len(p.args) && !p.peekIs("-o", "-or", ")") {
		p.accept("-a", "-and")

		right, err := p.parseUnary()
		if err != nil {
			return nil, err
		}

		left = findAnd{
			left:  left,
			right: right,
		}
	}

	return left, nil
}

// parseUnary parses a negation, a group or a primary.
func (p *findParser) parseUnary() (findExpression, error) {
	switch {
	case p.accept("!", "-not"):
		operand, err := p.parseUnary()
		if err != nil {
			return nil, err
		}

		return findNot{
			operand: operand,
		}, nil
	case p.accept("("):
		grouped, err := p.parseOr()
		if err != nil {
			return nil, err
		}

		if !p.accept(")") {
			return nil, errors.New("missing ')'")
		}

		return grouped, nil
	default:
		return p.parsePrimary()
	}
}

// parsePrimary parses a test or an action.
func (p *findParser) parsePrimary() (findExpression, error) {
	if p.position >= len(p.args) {
		return nil, errors.New("expected an expression")
	}

	name := p.args[p.position]
	p.position++

	switch name {
	case "-print":
		p.hasAction = true

		return findPrint{}, nil
	case "-print0":
		p.hasAction = true

		return findPrint{
			terminator: "\x00",
		}, nil
	case "-delete":
		p.hasAction = true
		p.deletes = true

		return findDelete{}, nil
	case "-exec", "-execdir":
		p.hasAction = true

		return p.parseExec()
	case "-true":
		return findConstant(true), nil
	case "-false":
		return findConstant(false), nil
	case "-empty":
		return findEmpty{}, nil
	case "-prune":
		return findConstant(true), nil
	}

	value, err := p.value(name)
	if err != nil {
		return nil, err
	}

	return p.parseValuePrimary(name, value)
}

// parseValuePrimary parses a primary that takes a value.
func (p *findParser) parseValuePrimary(name string, value string) (findExpression, error) {
	switch name {
	case "-name", "-iname":
		return newFindName(value, name == "-iname", false)
	case "-path", "-ipath", "-wholename", "-iwholename":
		return newFindName(value, strings.HasPrefix(name, "-i"), true)
	case "-type":
		if value != "f" && value != "d" {
			return nil, fmt.Errorf("unsupported file type '%s'", value)
		}

		return findType{
			directory: value == "d",
		}, nil
	case "-maxdepth", "-mindepth":
		return p.parseDepth(name, value)
	case "-size":
		return parseFindSize(value)
	default:
		return nil, fmt.Errorf("unknown predicate '%s'", name)
	}
}

// parseDepth applies -maxdepth or -mindepth, which are options rather than
// tests.
func (p *findParser) parseDepth(name string, value string) (findExpression, error) {
	depth, err := strconv.Atoi(value)
	if err != nil || depth < 0 {
		return nil, fmt.Errorf("invalid argument '%s' to '%s'", value, name)
	}

	if name == "-maxdepth" {
		p.run.maxDepth = depth
	} else {
		p.run.minDepth = depth
	}

	return findConstant(true), nil
}

// parseExec parses "-exec command... ;" or "-exec command... {} +".
func (p *findParser) parseExec() (findExpression, error) {
	var command []string

	for p.position < len(p.args) {
		arg := p.args[p.position]
		p.position++

		batched := arg == "+" && len(command) > 0 && command[len(command)-1] == "{}"
		if arg == ";" || batched {
			if len(command) == 0 {
				return nil, errors.New("missing command for '-exec'")
			}

			return &findExec{
				command: command,
				batched: batched,
			}, nil
		}

		command = append(command, arg)
	}

	return nil, errors.New("missing argument to '-exec'")
}

// value consumes the value of a primary.
func (p *findParser) value(name string) (string, error) {
	if p.position >= len(p.args) {
		return "", fmt.Errorf("missing argument to '%s'", name)
	}

	value := p.args[p.position]
	p.position++

	return value, nil
}

// accept consumes the next argument when it is one of the words.
func (p *findParser) accept(words ...string) bool {
	if !p.peekIs(words...) {
		return false
	}

	p.position++

	return true
}

// peekIs reports whether the next argument is one of the words.
func (p *findParser) peekIs(words ...string) bool {
	return p.position < len(p.args) && slices.Contains(words, p.args[p.position])
}

// findAnd is "left -a right".
type findAnd struct {
	left  findExpression
	right findExpression
}

// evaluate reports whether the entry satisfies the expression.
func (e findAnd) evaluate(ctx context.Context, run *findRun, entry findEntry) bool {
	return e.left.evaluate(ctx, run, entry) && e.right.evaluate(ctx, run, entry)
}

// findOr is "left -o right".
type findOr struct {
	left  findExpression
	right findExpression
}

// evaluate reports whether the entry satisfies the expression.
func (e findOr) evaluate(ctx context.Context, run *findRun, entry findEntry) bool {
	return e.left.evaluate(ctx, run, entry) || e.right.evaluate(ctx, run, entry)
}

// findNot is "! operand".
type findNot struct {
	operand findExpression
}

// evaluate reports whether the entry satisfies the expression.
func (e findNot) evaluate(ctx context.Context, run *findRun, entry findEntry) bool {
	return !e.operand.evaluate(ctx, run, entry)
}

// findConstant is -true, -false and the options that are always true.
type findConstant bool

// evaluate reports whether the entry satisfies the expression.
func (e findConstant) evaluate(context.Context, *findRun, findEntry) bool {
	return bool(e)
}

// findName is -name, -iname, -path and -ipath. Unlike in pathname
// expansion, wildcards also match slashes and leading dots.
type findName struct {
	expression *regexp.Regexp
	wholePath  bool
}

// newFindName compiles the shell pattern of a name test.
func newFindName(glob string, ignoreCase bool, wholePath bool) (findExpression, error) {
	mode := pattern.EntireString | pattern.GlobLeadingDot
	if ignoreCase {
		mode |= pattern.NoGlobCase
	}

	source, err := pattern.Regexp(glob, mode)
	if err != nil {
		return nil, fmt.Errorf("invalid pattern '%s': %w", glob, err)
	}

	expression, err := regexp.Compile(source)
	if err != nil {
		return nil, fmt.Errorf("invalid pattern '%s': %w", glob, err)
	}

	return findName{
		expression: expression,
		wholePath:  wholePath,
	}, nil
}

// evaluate matches the base name, or the whole name for path tests.
func (e findName) evaluate(_ context.Context, _ *findRun, entry findEntry) bool {
	subject := entry.name
	if !e.wholePath {
		subject = baseName(entry.name, "")
	}

	return e.expression.MatchString(subject)
}

// findType is -type f and -type d.
type findType struct {
	directory bool
}

// evaluate reports whether the entry satisfies the expression.
func (e findType) evaluate(_ context.Context, _ *findRun, entry findEntry) bool {
	return entry.info.IsDir() == e.directory
}

// findEmpty is -empty: an empty file or a directory without entries.
type findEmpty struct{}

// evaluate reports whether the entry satisfies the expression.
func (findEmpty) evaluate(_ context.Context, run *findRun, entry findEntry) bool {
	if !entry.info.IsDir() {
		return entry.info.Size() == 0
	}

	entries, err := run.call.fs().ReadDir(run.call.resolve(entry.name))

	return err == nil && len(entries) == 0
}

// findSize is -size [+-]N[ckMG]; without a unit N counts 512-byte blocks.
type findSize struct {
	comparison byte
	size       int64
	unit       int64
}

// parseFindSize parses the value of -size.
func parseFindSize(value string) (findExpression, error) {
	expression := findSize{
		unit: 512,
	}

	if value != "" && (value[0] == '+' || value[0] == '-') {
		expression.comparison = value[0]
		value = value[1:]
	}

	units := map[byte]int64{
		'c': 1,
		'w': 2,
		'b': 512,
		'k': 1024,
		'M': 1024 * 1024,
		'G': 1024 * 1024 * 1024,
	}

	if value != "" {
		if unit, ok := units[value[len(value)-1]]; ok {
			expression.unit = unit
			value = value[:len(value)-1]
		}
	}

	size, err := strconv.ParseInt(value, 10, 64)
	if err != nil {
		return nil, fmt.Errorf("invalid argument '%s' to '-size'", value)
	}

	expression.size = size

	return expression, nil
}

// evaluate reports whether the entry satisfies the expression.
func (e findSize) evaluate(_ context.Context, _ *findRun, entry findEntry) bool {
	units := (entry.info.Size() + e.unit - 1) / e.unit

	switch e.comparison {
	case '+':
		return units > e.size
	case '-':
		return units < e.size
	default:
		return units == e.size
	}
}

// findPrint is -print and -print0.
type findPrint struct {
	terminator string
}

// evaluate reports whether the entry satisfies the expression.
func (e findPrint) evaluate(_ context.Context, run *findRun, entry findEntry) bool {
	terminator := e.terminator
	if terminator == "" {
		terminator = "\n"
	}

	run.call.writeOutput(entry.name + terminator)

	return true
}

// findDelete is -delete.
type findDelete struct{}

// evaluate reports whether the entry satisfies the expression.
func (findDelete) evaluate(_ context.Context, run *findRun, entry findEntry) bool {
	if err := run.call.fs().Remove(run.call.resolve(entry.name)); err != nil {
		run.call.report("cannot delete '%s': %s", entry.name, describeError(err))
		run.ok = false

		return false
	}

	return true
}

// findExec is "-exec command ;" and "-exec command {} +".
type findExec struct {
	command []string
	batched bool
}

// evaluate reports whether the entry satisfies the expression.
func (e *findExec) evaluate(ctx context.Context, run *findRun, entry findEntry) bool {
	if e.batched {
		run.batches[e] = append(run.batches[e], entry.name)

		return true
	}

	return e.runWith(ctx, run, []string{entry.name}) == 0
}

// runWith runs the command with "{}" replaced by the names.
func (e *findExec) runWith(ctx context.Context, run *findRun, names []string) int {
	var args []string

	for _, arg := range e.command {
		if arg == "{}" && e.batched {
			args = append(args, names...)

			continue
		}

		args = append(args, strings.ReplaceAll(arg, "{}", names[0]))
	}

	return run.call.shell.runArgs(ctx, run.call, args)
}
