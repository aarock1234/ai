package sandboxshell

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"math/rand/v2"
	"path"
	"strconv"
	"strings"
	"time"
)

// runCat carries out "cat [-n] [file...]", streaming the standard input.
func runCat(ctx context.Context, call *commandCall) int {
	parsed, ok := call.parseCommandArgs(argSpec{})
	if !ok {
		return exitFailure
	}

	numberLines := false
	for _, flag := range parsed.flags {
		switch flag.name {
		case "-n", "--number":
			numberLines = true
		case "-u":
		default:
			return call.invalidFlag(flag)
		}
	}

	operands := parsed.operands
	if len(operands) == 0 {
		operands = []string{
			"-",
		}
	}

	output := call.stdout
	if numberLines {
		output = &lineNumberWriter{
			writer: call.stdout,
			atLine: true,
		}
	}

	ok = true
	for _, operand := range operands {
		if ctx.Err() != nil {
			return exitInterrupted
		}

		reader, err := call.openOperand(operand)
		if err != nil {
			call.reportFile(operand, err)
			ok = false

			continue
		}

		if _, err := io.Copy(output, reader); err != nil {
			return exitFailure
		}
	}

	return statusOf(ok)
}

// openOperand opens an input operand for streaming.
func (c *commandCall) openOperand(operand string) (io.Reader, error) {
	if operand == "-" || c.resolve(operand) == "/dev/stdin" {
		return c.input(), nil
	}

	contents, err := c.readOperand(operand)
	if err != nil {
		return nil, err
	}

	return bytes.NewReader(contents), nil
}

// lineNumberWriter numbers the lines written through it, as cat -n does.
type lineNumberWriter struct {
	writer io.Writer
	line   int
	atLine bool
}

// Write writes data, prefixing every line with its number.
func (w *lineNumberWriter) Write(data []byte) (int, error) {
	var numbered bytes.Buffer

	for _, character := range data {
		if w.atLine {
			w.line++
			fmt.Fprintf(&numbered, "%6d\t", w.line)
			w.atLine = false
		}

		numbered.WriteByte(character)
		w.atLine = character == '\n'
	}

	if _, err := w.writer.Write(numbered.Bytes()); err != nil {
		return 0, err
	}

	return len(data), nil
}

// runTee carries out "tee [-a] [file...]".
func runTee(ctx context.Context, call *commandCall) int {
	parsed, ok := call.parseCommandArgs(argSpec{})
	if !ok {
		return exitFailure
	}

	appendToFiles := false
	for _, flag := range parsed.flags {
		switch flag.name {
		case "-a", "--append":
			appendToFiles = true
		case "-i", "-p":
		default:
			return call.invalidFlag(flag)
		}
	}

	writers := []io.Writer{
		call.stdout,
	}

	ok = true
	for _, operand := range parsed.operands {
		writer, err := call.openOutput(operand, appendToFiles)
		if err != nil {
			call.reportFile(operand, err)
			ok = false

			continue
		}

		writers = append(writers, writer)
	}

	if _, err := io.Copy(io.MultiWriter(writers...), contextReader{
		ctx:    ctx,
		reader: call.input(),
	}); err != nil {
		return exitFailure
	}

	return statusOf(ok)
}

// openOutput opens an output operand: a device or a virtual file, which is
// truncated unless appending.
func (c *commandCall) openOutput(operand string, appending bool) (io.Writer, error) {
	resolved := c.resolve(operand)

	switch resolved {
	case "/dev/null":
		return io.Discard, nil
	case "/dev/stdout":
		return c.stdout, nil
	case "/dev/stderr":
		return c.stderr, nil
	}

	flag := openWriteOnly | openCreate | openTruncate
	if appending {
		flag = openWriteOnly | openCreate | openAppend
	}

	return c.fs().OpenFile(resolved, flag, 0o644)
}

// contextReader stops reading when its context ends.
type contextReader struct {
	ctx    context.Context
	reader io.Reader
}

// Read reads unless the context ended.
func (r contextReader) Read(data []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}

	return r.reader.Read(data)
}

// runMkdir carries out "mkdir [-p] [-m mode] directory...".
func runMkdir(_ context.Context, call *commandCall) int {
	parsed, ok := call.parseCommandArgs(argSpec{
		valueFlags: []string{
			"-m",
			"--mode",
		},
	})
	if !ok {
		return exitFailure
	}

	parents := false
	verbose := false
	perm := fs.FileMode(0o755)

	for _, flag := range parsed.flags {
		switch flag.name {
		case "-p", "--parents":
			parents = true
		case "-v", "--verbose":
			verbose = true
		case "-m", "--mode":
			mode, err := applyMode(perm, true, flag.value)
			if err != nil {
				return call.fail("invalid mode '%s'", flag.value)
			}

			perm = mode
		default:
			return call.invalidFlag(flag)
		}
	}

	if len(parsed.operands) == 0 {
		return call.fail("missing operand")
	}

	ok = true
	for _, operand := range parsed.operands {
		resolved := call.resolve(operand)

		err := call.fs().Mkdir(resolved, perm)
		if parents {
			err = call.fs().MkdirAll(resolved, perm)
		}

		if err != nil {
			call.report("cannot create directory '%s': %s", operand, describeError(err))
			ok = false

			continue
		}

		if verbose {
			call.writeOutput(fmt.Sprintf("mkdir: created directory '%s'\n", operand))
		}
	}

	return statusOf(ok)
}

// runRmdir carries out "rmdir [-p] directory...".
func runRmdir(_ context.Context, call *commandCall) int {
	parsed, ok := call.parseCommandArgs(argSpec{})
	if !ok {
		return exitFailure
	}

	parents := false
	for _, flag := range parsed.flags {
		switch flag.name {
		case "-p", "--parents":
			parents = true
		case "--ignore-fail-on-non-empty", "-v", "--verbose":
		default:
			return call.invalidFlag(flag)
		}
	}

	if len(parsed.operands) == 0 {
		return call.fail("missing operand")
	}

	ok = true
	for _, operand := range parsed.operands {
		for _, directory := range removalOrder(operand, parents) {
			if err := call.removeDirectory(directory); err != nil {
				call.report("failed to remove '%s': %s", directory, describeError(err))
				ok = false

				break
			}
		}
	}

	return statusOf(ok)
}

// removalOrder returns the directory operand and, with parents, its
// ancestors named in the operand, innermost first.
func removalOrder(operand string, parents bool) []string {
	directories := []string{
		operand,
	}

	if !parents {
		return directories
	}

	for current := path.Dir(strings.TrimRight(operand, "/")); current != "." && current != "/"; current = path.Dir(current) {
		directories = append(directories, current)
	}

	return directories
}

// removeDirectory removes an empty directory.
func (c *commandCall) removeDirectory(operand string) error {
	resolved := c.resolve(operand)

	info, err := c.fs().Stat(resolved)
	if err != nil {
		return err
	}

	if !info.IsDir() {
		return errNotDirectory
	}

	return c.fs().Remove(resolved)
}

// runRemove carries out "rm [-r] [-f] [-d] file...".
func runRemove(_ context.Context, call *commandCall) int {
	parsed, ok := call.parseCommandArgs(argSpec{})
	if !ok {
		return exitFailure
	}

	var options removeOptions
	for _, flag := range parsed.flags {
		switch flag.name {
		case "-r", "-R", "--recursive":
			options.recursive = true
		case "-f", "--force":
			options.force = true
		case "-d", "--dir":
			options.directories = true
		case "-v", "--verbose":
			options.verbose = true
		case "-i", "-I":
		default:
			return call.invalidFlag(flag)
		}
	}

	if len(parsed.operands) == 0 && !options.force {
		return call.fail("missing operand")
	}

	ok = true
	for _, operand := range parsed.operands {
		if !call.removeOperand(operand, options) {
			ok = false
		}
	}

	return statusOf(ok)
}

// removeOptions are the flags of rm.
type removeOptions struct {
	recursive   bool
	force       bool
	directories bool
	verbose     bool
}

// removeOperand removes one operand of rm and reports whether it succeeded.
func (c *commandCall) removeOperand(operand string, options removeOptions) bool {
	resolved := c.resolve(operand)
	if resolved == "/" {
		c.report("it is dangerous to operate recursively on '/'")

		return false
	}

	info, err := c.fs().Stat(resolved)
	if err != nil {
		if isNotExist(err) && options.force {
			return true
		}

		c.report("cannot remove '%s': %s", operand, describeError(err))

		return false
	}

	switch {
	case info.IsDir() && options.recursive:
		err = c.fs().RemoveAll(resolved)
	case info.IsDir() && !options.directories:
		err = errIsDirectory
	default:
		err = c.fs().Remove(resolved)
	}

	if err != nil {
		c.report("cannot remove '%s': %s", operand, describeError(err))

		return false
	}

	if options.verbose {
		c.writeOutput(fmt.Sprintf("removed '%s'\n", operand))
	}

	return true
}

// runTouch carries out "touch [-c] file...".
func runTouch(_ context.Context, call *commandCall) int {
	parsed, ok := call.parseCommandArgs(argSpec{
		valueFlags: []string{
			"-d",
			"-t",
			"-r",
			"--date",
			"--reference",
		},
	})
	if !ok {
		return exitFailure
	}

	create := true
	for _, flag := range parsed.flags {
		switch flag.name {
		case "-c", "--no-create":
			create = false
		case "-a", "-m", "-d", "-t", "-r", "--date", "--reference":
		default:
			return call.invalidFlag(flag)
		}
	}

	if len(parsed.operands) == 0 {
		return call.fail("missing file operand")
	}

	ok = true
	now := time.Now()

	for _, operand := range parsed.operands {
		err := call.fs().Touch(call.resolve(operand), now, create)
		if err == nil || isNotExist(err) && !create {
			continue
		}

		call.report("cannot touch '%s': %s", operand, describeError(err))
		ok = false
	}

	return statusOf(ok)
}

// runCopy carries out "cp [-r] [-n] source... destination".
func runCopy(ctx context.Context, call *commandCall) int {
	parsed, ok := call.parseCommandArgs(argSpec{})
	if !ok {
		return exitFailure
	}

	var options transferOptions
	for _, flag := range parsed.flags {
		switch flag.name {
		case "-r", "-R", "-a", "--recursive", "--archive":
			options.recursive = true
		case "-n", "--no-clobber":
			options.noClobber = true
		case "-v", "--verbose":
			options.verbose = true
		case "-f", "--force", "-p", "-P", "-L", "-d", "--preserve":
		default:
			return call.invalidFlag(flag)
		}
	}

	return call.transfer(ctx, parsed.operands, options, call.copyOperand)
}

// runMove carries out "mv [-n] source... destination".
func runMove(ctx context.Context, call *commandCall) int {
	parsed, ok := call.parseCommandArgs(argSpec{})
	if !ok {
		return exitFailure
	}

	var options transferOptions
	for _, flag := range parsed.flags {
		switch flag.name {
		case "-n", "--no-clobber":
			options.noClobber = true
		case "-v", "--verbose":
			options.verbose = true
		case "-f", "--force", "-i":
		default:
			return call.invalidFlag(flag)
		}
	}

	options.recursive = true

	return call.transfer(ctx, parsed.operands, options, call.moveOperand)
}

// transferOptions are the flags of cp and mv.
type transferOptions struct {
	recursive bool
	noClobber bool
	verbose   bool
}

// transferFunc copies or moves one source to its final target.
type transferFunc func(source string, target string, options transferOptions) error

// transfer applies cp or mv semantics to operands: with several sources or
// an existing directory destination, sources go inside the destination.
func (c *commandCall) transfer(ctx context.Context, operands []string, options transferOptions, apply transferFunc) int {
	if len(operands) < 2 {
		return c.fail("missing destination file operand")
	}

	sources := operands[:len(operands)-1]
	destination := c.resolve(operands[len(operands)-1])

	info, err := c.fs().Stat(destination)
	intoDirectory := err == nil && info.IsDir()

	if len(sources) > 1 && !intoDirectory {
		return c.fail("target '%s' is not a directory", operands[len(operands)-1])
	}

	ok := true
	for _, source := range sources {
		if ctx.Err() != nil {
			return exitInterrupted
		}

		target := destination
		if intoDirectory {
			target = path.Join(destination, path.Base(c.resolve(source)))
		}

		if options.noClobber && c.exists(target) {
			continue
		}

		if err := apply(c.resolve(source), target, options); err != nil {
			c.report("%s", describeTransferError(source, err))
			ok = false

			continue
		}

		if options.verbose {
			c.writeOutput(fmt.Sprintf("'%s' -> '%s'\n", source, target))
		}
	}

	return statusOf(ok)
}

// exists reports whether a virtual path exists.
func (c *commandCall) exists(name string) bool {
	_, err := c.fs().Stat(name)

	return err == nil
}

// copyOperand copies a file or, recursively, a directory.
func (c *commandCall) copyOperand(source string, target string, options transferOptions) error {
	info, err := c.fs().Stat(source)
	if err != nil {
		return err
	}

	if !info.IsDir() {
		return c.copyFile(source, target, info.Mode())
	}

	if !options.recursive {
		return transferError{
			message: fmt.Sprintf("-r not specified; omitting directory '%s'", source),
		}
	}

	if target == source || isWithin(target, source) {
		return transferError{
			message: fmt.Sprintf("cannot copy a directory, '%s', into itself, '%s'", source, target),
		}
	}

	return c.copyTree(source, target, info.Mode())
}

// copyFile copies the contents and permissions of a file.
func (c *commandCall) copyFile(source string, target string, mode fs.FileMode) error {
	contents, err := c.fs().ReadFile(source)
	if err != nil {
		return err
	}

	if err := c.fs().WriteFile(target, contents, mode.Perm()); err != nil {
		return err
	}

	return c.fs().Chmod(target, mode.Perm())
}

// copyTree copies a directory and everything below it.
func (c *commandCall) copyTree(source string, target string, mode fs.FileMode) error {
	if err := c.fs().MkdirAll(target, mode.Perm()); err != nil {
		return err
	}

	entries, err := c.fs().ReadDir(source)
	if err != nil {
		return err
	}

	for _, entry := range entries {
		info, err := entry.Info()
		if err != nil {
			return err
		}

		sourceChild := path.Join(source, entry.Name())
		targetChild := path.Join(target, entry.Name())

		if entry.IsDir() {
			err = c.copyTree(sourceChild, targetChild, info.Mode())
		} else {
			err = c.copyFile(sourceChild, targetChild, info.Mode())
		}

		if err != nil {
			return err
		}
	}

	return nil
}

// moveOperand renames a file or directory.
func (c *commandCall) moveOperand(source string, target string, _ transferOptions) error {
	return c.fs().Rename(source, target)
}

// transferError is a cp or mv failure with its own message.
type transferError struct {
	message string
}

// Error returns the message.
func (e transferError) Error() string {
	return e.message
}

// describeTransferError formats a cp or mv failure.
func describeTransferError(source string, err error) string {
	if transfer, ok := errors.AsType[transferError](err); ok {
		return transfer.message
	}

	return fmt.Sprintf("cannot stat '%s': %s", source, describeError(err))
}

// runChmod carries out "chmod [-R] mode file...". Flags are parsed by hand,
// because symbolic modes such as "-w" look like flags.
func runChmod(_ context.Context, call *commandCall) int {
	recursive := false
	args := call.args

	for len(args) > 0 && isChmodFlag(args[0]) {
		if args[0] == "-R" || args[0] == "--recursive" {
			recursive = true
		}

		args = args[1:]
	}

	if len(args) > 0 && args[0] == "--" {
		args = args[1:]
	}

	if len(args) < 2 {
		return call.fail("missing operand")
	}

	if _, err := applyMode(0, false, args[0]); err != nil {
		return call.fail("invalid mode: '%s'", args[0])
	}

	ok := true
	for _, operand := range args[1:] {
		if err := call.changeMode(call.resolve(operand), args[0], recursive); err != nil {
			call.report("cannot access '%s': %s", operand, describeError(err))
			ok = false
		}
	}

	return statusOf(ok)
}

// isChmodFlag reports whether an argument is a flag of chmod.
func isChmodFlag(arg string) bool {
	switch arg {
	case "-R", "--recursive", "-v", "--verbose", "-f", "--silent":
		return true
	default:
		return false
	}
}

// changeMode applies a mode specification to a path and, recursively, to
// its contents.
func (c *commandCall) changeMode(name string, specification string, recursive bool) error {
	info, err := c.fs().Stat(name)
	if err != nil {
		return err
	}

	mode, err := applyMode(info.Mode()&permissionBits, info.IsDir(), specification)
	if err != nil {
		return err
	}

	if err := c.fs().Chmod(name, mode); err != nil {
		return err
	}

	if !recursive || !info.IsDir() {
		return nil
	}

	entries, err := c.fs().ReadDir(name)
	if err != nil {
		return err
	}

	for _, entry := range entries {
		if err := c.changeMode(path.Join(name, entry.Name()), specification, true); err != nil {
			return err
		}
	}

	return nil
}

// runMktemp carries out "mktemp [-d] [-p directory] [-t] [template]".
func runMktemp(_ context.Context, call *commandCall) int {
	parsed, ok := call.parseCommandArgs(argSpec{
		valueFlags: []string{
			"-p",
			"--tmpdir",
			"--suffix",
		},
	})
	if !ok {
		return exitFailure
	}

	request := temporaryRequest{
		template: "tmp.XXXXXXXXXX",
	}

	for _, flag := range parsed.flags {
		switch flag.name {
		case "-d", "--directory":
			request.directory = true
		case "-p", "--tmpdir":
			request.parent = flag.value
			request.inTemporary = true
		case "-t":
			request.inTemporary = true
		case "--suffix":
			request.suffix = flag.value
		case "-q", "--quiet", "-u", "--dry-run":
		default:
			return call.invalidFlag(flag)
		}
	}

	if len(parsed.operands) > 0 {
		request.template = parsed.operands[0]
	} else {
		request.inTemporary = true
	}

	created, err := call.createTemporary(request)
	if err != nil {
		return call.fail("failed to create %s: %s", request.kind(), describeError(err))
	}

	call.writeOutput(created + "\n")

	return 0
}

// temporaryRequest describes the file or directory mktemp creates.
type temporaryRequest struct {
	template    string
	suffix      string
	parent      string
	directory   bool
	inTemporary bool
}

// kind names what mktemp creates.
func (r temporaryRequest) kind() string {
	if r.directory {
		return "directory"
	}

	return "file"
}

// createTemporary creates a uniquely named file or directory from the
// template's trailing X characters.
func (c *commandCall) createTemporary(request temporaryRequest) (string, error) {
	base := request.template
	if request.inTemporary {
		parent := request.parent
		if parent == "" {
			parent = c.env.Get("TMPDIR").String()
		}

		if parent == "" {
			parent = temporaryDirectory
		}

		base = path.Join(parent, request.template)
	}

	prefix := strings.TrimRight(base, "X")
	randomLength := max(len(base)-len(prefix), 3)

	var err error
	for range 100 {
		name := c.resolve(prefix + randomName(randomLength) + request.suffix)

		if request.directory {
			err = c.fs().Mkdir(name, 0o700)
		} else {
			_, err = c.fs().OpenFile(name, openWriteOnly|openCreate|openExclusive, 0o600)
		}

		if err == nil {
			return name, nil
		}

		if !isExist(err) {
			return "", err
		}
	}

	return "", err
}

// randomName returns random letters and digits.
func randomName(length int) string {
	const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

	name := make([]byte, length)
	for index := range name {
		name[index] = alphabet[rand.IntN(len(alphabet))]
	}

	return string(name)
}

// runRealpath carries out "realpath [-e|-m] file...". There are no
// symbolic links, so resolving a path only makes it absolute and clean.
func runRealpath(_ context.Context, call *commandCall) int {
	parsed, ok := call.parseCommandArgs(argSpec{})
	if !ok {
		return exitFailure
	}

	mustExist := false
	missingAllowed := false
	quiet := false

	for _, flag := range parsed.flags {
		switch flag.name {
		case "-e", "--canonicalize-existing":
			mustExist = true
		case "-m", "--canonicalize-missing":
			missingAllowed = true
		case "-q", "--quiet":
			quiet = true
		case "-s", "-L", "-P", "-E", "--no-symlinks":
		default:
			return call.invalidFlag(flag)
		}
	}

	if len(parsed.operands) == 0 {
		return call.fail("missing operand")
	}

	ok = true
	for _, operand := range parsed.operands {
		resolved := call.resolve(operand)

		checked := resolved
		if !mustExist {
			checked = path.Dir(resolved)
		}

		if _, err := call.fs().Stat(checked); err != nil && !missingAllowed {
			if !quiet {
				call.reportFile(operand, err)
			}

			ok = false

			continue
		}

		call.writeOutput(resolved + "\n")
	}

	return statusOf(ok)
}

// runBasename carries out "basename name [suffix]" and "basename -a
// [-s suffix] name...".
func runBasename(_ context.Context, call *commandCall) int {
	parsed, ok := call.parseCommandArgs(argSpec{
		valueFlags: []string{
			"-s",
			"--suffix",
		},
	})
	if !ok {
		return exitFailure
	}

	multiple := false
	suffix := ""

	for _, flag := range parsed.flags {
		switch flag.name {
		case "-a", "--multiple":
			multiple = true
		case "-s", "--suffix":
			suffix = flag.value
			multiple = true
		default:
			return call.invalidFlag(flag)
		}
	}

	names := parsed.operands
	switch {
	case len(names) == 0:
		return call.fail("missing operand")
	case !multiple && len(names) == 2:
		suffix = names[1]
		names = names[:1]
	case !multiple && len(names) > 2:
		return call.fail("extra operand '%s'", names[2])
	}

	for _, name := range names {
		call.writeOutput(baseName(name, suffix) + "\n")
	}

	return 0
}

// baseName returns the last element of a name without a suffix, as
// basename prints it.
func baseName(name string, suffix string) string {
	trimmed := strings.TrimRight(name, "/")
	if trimmed == "" {
		if name == "" {
			return ""
		}

		return "/"
	}

	base := path.Base(trimmed)
	if suffix != "" && base != suffix {
		base = strings.TrimSuffix(base, suffix)
	}

	return base
}

// runDirname carries out "dirname name...".
func runDirname(_ context.Context, call *commandCall) int {
	if len(call.args) == 0 {
		return call.fail("missing operand")
	}

	for _, name := range call.args {
		call.writeOutput(directoryName(name) + "\n")
	}

	return 0
}

// directoryName returns a name without its last element, as dirname prints
// it.
func directoryName(name string) string {
	trimmed := strings.TrimRight(name, "/")
	if trimmed == "" {
		if strings.HasPrefix(name, "/") {
			return "/"
		}

		return "."
	}

	return path.Dir(trimmed)
}

// runStat carries out "stat [-c format] file...".
func runStat(_ context.Context, call *commandCall) int {
	parsed, ok := call.parseCommandArgs(argSpec{
		valueFlags: []string{
			"-c",
			"--format",
			"--printf",
		},
	})
	if !ok {
		return exitFailure
	}

	format := ""
	for _, flag := range parsed.flags {
		switch flag.name {
		case "-c", "--format":
			format = flag.value + "\n"
		case "--printf":
			format = unescapeBackslashes(flag.value)
		case "-L", "--dereference":
		default:
			return call.invalidFlag(flag)
		}
	}

	if len(parsed.operands) == 0 {
		return call.fail("missing operand")
	}

	ok = true
	for _, operand := range parsed.operands {
		info, err := call.fs().Stat(call.resolve(operand))
		if err != nil {
			call.report("cannot stat '%s': %s", operand, describeError(err))
			ok = false

			continue
		}

		if format == "" {
			call.writeOutput(defaultStat(operand, info))

			continue
		}

		call.writeOutput(formatStat(format, operand, info))
	}

	return statusOf(ok)
}

// defaultStat returns stat's default description of a file.
func defaultStat(operand string, info fs.FileInfo) string {
	return fmt.Sprintf(
		"  File: %s\n  Size: %d\tBlocks: %d\tIO Block: 4096\t%s\nAccess: (%04o/%s)  Uid: ( %s/ %s)   Gid: ( %s/ %s)\nModify: %s\n",
		operand,
		info.Size(),
		(info.Size()+511)/512,
		fileTypeName(info),
		info.Mode().Perm(),
		modeString(info.Mode()),
		virtualUserID,
		virtualUser,
		virtualUserID,
		virtualUser,
		info.ModTime().UTC().Format("2006-01-02 15:04:05.000000000 -0700"),
	)
}

// formatStat expands the stat format directives.
func formatStat(format string, operand string, info fs.FileInfo) string {
	var output strings.Builder

	for index := 0; index < len(format); index++ {
		if format[index] != '%' || index+1 == len(format) {
			output.WriteByte(format[index])

			continue
		}

		index++
		output.WriteString(statDirective(format[index], operand, info))
	}

	return output.String()
}

// statDirective expands one stat format directive.
func statDirective(directive byte, operand string, info fs.FileInfo) string {
	switch directive {
	case 'n':
		return operand
	case 's':
		return strconv.FormatInt(info.Size(), 10)
	case 'a':
		return strconv.FormatUint(uint64(info.Mode().Perm()), 8)
	case 'A':
		return modeString(info.Mode())
	case 'F':
		return fileTypeName(info)
	case 'U', 'G':
		return virtualUser
	case 'u', 'g':
		return virtualUserID
	case 'Y':
		return strconv.FormatInt(info.ModTime().Unix(), 10)
	case 'y':
		return info.ModTime().UTC().Format("2006-01-02 15:04:05.000000000 -0700")
	case '%':
		return "%"
	default:
		return "%" + string(directive)
	}
}

// fileTypeName names the type of a file as stat prints it.
func fileTypeName(info fs.FileInfo) string {
	switch {
	case info.IsDir():
		return "directory"
	case info.Mode()&fs.ModeCharDevice != 0:
		return "character special file"
	case info.Size() == 0:
		return "regular empty file"
	default:
		return "regular file"
	}
}
