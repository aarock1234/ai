package sandboxshell

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"io/fs"
	"path"
	"slices"
	"strconv"
	"strings"
	"time"
)

// listOptions are the flags of ls.
type listOptions struct {
	all          bool
	almostAll    bool
	long         bool
	recursive    bool
	directory    bool
	humanSizes   bool
	byTime       bool
	bySize       bool
	reverse      bool
	classify     bool
	slashDirs    bool
	unsortedList bool
}

// listEntry is one name ls prints.
type listEntry struct {
	name string
	info fs.FileInfo
}

// runList carries out "ls [-aAlRdhtSrFp1] [file...]". Output is one name
// per line, as ls prints when its output is not a terminal.
func runList(ctx context.Context, call *commandCall) int {
	parsed, ok := call.parseCommandArgs(argSpec{})
	if !ok {
		return exitUsage
	}

	options, ok := parseListOptions(call, parsed.flags)
	if !ok {
		return exitUsage
	}

	operands := parsed.operands
	if len(operands) == 0 {
		operands = []string{
			".",
		}
	}

	files, directories, ok := call.classifyListOperands(operands, options)
	sortListEntries(files, options)
	sortListEntries(directories, options)

	call.printListEntries(files, options)

	showHeaders := len(operands) > 1 || options.recursive
	for index, directory := range directories {
		if ctx.Err() != nil {
			return exitInterrupted
		}

		if index > 0 || len(files) > 0 {
			call.writeOutput("\n")
		}

		if !call.listDirectory(ctx, directory, options, showHeaders) {
			ok = false
		}
	}

	return statusOf(ok)
}

// parseListOptions reads the flags of ls.
func parseListOptions(call *commandCall, flags []parsedFlag) (listOptions, bool) {
	var options listOptions

	for _, flag := range flags {
		switch flag.name {
		case "-a", "--all":
			options.all = true
		case "-A", "--almost-all":
			options.almostAll = true
		case "-l":
			options.long = true
		case "-R", "--recursive":
			options.recursive = true
		case "-d", "--directory":
			options.directory = true
		case "-h", "--human-readable":
			options.humanSizes = true
		case "-t":
			options.byTime = true
		case "-S":
			options.bySize = true
		case "-r", "--reverse":
			options.reverse = true
		case "-F", "--classify":
			options.classify = true
		case "-p":
			options.slashDirs = true
		case "-U", "-f":
			options.unsortedList = true
		case "-n", "-o", "-g":
			options.long = true
		case "-1", "-C", "-x", "-m", "-G", "--color":
		default:
			call.invalidFlag(flag)

			return listOptions{}, false
		}
	}

	return options, true
}

// classifyListOperands splits operands into files, listed together, and
// directories, listed one by one.
func (c *commandCall) classifyListOperands(operands []string, options listOptions) ([]listEntry, []listEntry, bool) {
	var files []listEntry
	var directories []listEntry
	ok := true

	for _, operand := range operands {
		info, err := c.fs().Stat(c.resolve(operand))
		if err != nil {
			c.report("cannot access '%s': %s", operand, describeError(err))
			ok = false

			continue
		}

		entry := listEntry{
			name: operand,
			info: info,
		}

		if info.IsDir() && !options.directory {
			directories = append(directories, entry)
		} else {
			files = append(files, entry)
		}
	}

	return files, directories, ok
}

// listDirectory prints a directory's entries and, recursively, its
// subdirectories.
func (c *commandCall) listDirectory(ctx context.Context, directory listEntry, options listOptions, showHeader bool) bool {
	if showHeader {
		c.writeOutput(directory.name + ":\n")
	}

	resolved := c.resolve(directory.name)

	entries, err := c.fs().ReadDir(resolved)
	if err != nil {
		c.report("cannot open directory '%s': %s", directory.name, describeError(err))

		return false
	}

	listed := c.visibleEntries(resolved, entries, options)
	sortListEntries(listed, options)

	if options.long {
		c.writeOutput(fmt.Sprintf("total %d\n", totalBlocks(listed)))
	}

	c.printListEntries(listed, options)

	if !options.recursive {
		return true
	}

	ok := true
	for _, entry := range listed {
		if !entry.info.IsDir() || entry.name == "." || entry.name == ".." {
			continue
		}

		if ctx.Err() != nil {
			return false
		}

		c.writeOutput("\n")

		child := listEntry{
			name: path.Join(directory.name, entry.name),
			info: entry.info,
		}

		if !c.listDirectory(ctx, child, options, true) {
			ok = false
		}
	}

	return ok
}

// visibleEntries returns the entries ls shows, including "." and ".." with
// -a and hiding dot files without -a or -A.
func (c *commandCall) visibleEntries(directory string, entries []fs.DirEntry, options listOptions) []listEntry {
	var listed []listEntry

	if options.all {
		listed = append(listed, c.specialEntry(".", directory), c.specialEntry("..", path.Dir(directory)))
	}

	for _, entry := range entries {
		hidden := strings.HasPrefix(entry.Name(), ".")
		if hidden && !options.all && !options.almostAll {
			continue
		}

		info, err := entry.Info()
		if err != nil {
			continue
		}

		listed = append(listed, listEntry{
			name: entry.Name(),
			info: info,
		})
	}

	return listed
}

// specialEntry describes "." or "..".
func (c *commandCall) specialEntry(name string, directory string) listEntry {
	info, _ := c.fs().Stat(directory)

	return listEntry{
		name: name,
		info: info,
	}
}

// sortListEntries orders entries by name, time or size.
func sortListEntries(entries []listEntry, options listOptions) {
	if options.unsortedList {
		return
	}

	slices.SortStableFunc(entries, func(first, second listEntry) int {
		order := strings.Compare(first.name, second.name)

		switch {
		case options.bySize:
			order = cmp.Or(cmp.Compare(second.info.Size(), first.info.Size()), order)
		case options.byTime:
			order = cmp.Or(second.info.ModTime().Compare(first.info.ModTime()), order)
		}

		if options.reverse {
			return -order
		}

		return order
	})
}

// printListEntries prints entries in the short or long format.
func (c *commandCall) printListEntries(entries []listEntry, options listOptions) {
	for _, entry := range entries {
		name := entry.name + listSuffix(entry.info, options)

		if !options.long {
			c.writeOutput(name + "\n")

			continue
		}

		c.writeOutput(fmt.Sprintf(
			"%s 1 %s %s %s %s %s\n",
			modeString(entry.info.Mode()),
			virtualUser,
			virtualUser,
			formatListSize(entry.info, options.humanSizes),
			formatListTime(entry.info.ModTime()),
			name,
		))
	}
}

// listSuffix returns the classification suffix of -F and -p.
func listSuffix(info fs.FileInfo, options listOptions) string {
	switch {
	case info.IsDir() && (options.classify || options.slashDirs):
		return "/"
	case options.classify && isExecutable(info):
		return "*"
	default:
		return ""
	}
}

// totalBlocks returns the 1 KiB blocks of the entries, for ls -l.
func totalBlocks(entries []listEntry) int64 {
	var total int64
	for _, entry := range entries {
		total += (entry.info.Size() + 1023) / 1024
	}

	return total
}

// formatListSize formats a size column, right-aligned.
func formatListSize(info fs.FileInfo, human bool) string {
	size := info.Size()
	if info.IsDir() {
		size = 4096
	}

	if !human {
		return fmt.Sprintf("%8d", size)
	}

	return fmt.Sprintf("%5s", humanSize(size))
}

// humanSize formats a size with a binary unit suffix, as ls -h does.
func humanSize(size int64) string {
	const units = "KMGTPE"

	if size < 1024 {
		return strconv.FormatInt(size, 10)
	}

	value := float64(size)
	unit := -1

	for value >= 1024 && unit < len(units)-1 {
		value /= 1024
		unit++
	}

	if value < 10 {
		return fmt.Sprintf("%.1f%c", value, units[unit])
	}

	return fmt.Sprintf("%.0f%c", value, units[unit])
}

// formatListTime formats a modification time as ls -l does: with the time
// of day within six months, and with the year otherwise.
func formatListTime(modTime time.Time) string {
	modTime = modTime.UTC()
	if time.Since(modTime) < 180*24*time.Hour {
		return modTime.Format("Jan _2 15:04")
	}

	return modTime.Format("Jan _2  2006")
}

// modeString formats a mode as ls -l does, such as "-rw-r--r--".
func modeString(mode fs.FileMode) string {
	var text strings.Builder

	switch {
	case mode.IsDir():
		text.WriteByte('d')
	case mode&fs.ModeCharDevice != 0:
		text.WriteByte('c')
	default:
		text.WriteByte('-')
	}

	const letters = "rwxrwxrwx"
	for index := range 9 {
		if mode&(1<<(8-index)) != 0 {
			text.WriteByte(letters[index])
		} else {
			text.WriteByte('-')
		}
	}

	formatted := []byte(text.String())
	applySpecialBit(formatted, 3, mode&fs.ModeSetuid != 0, 's')
	applySpecialBit(formatted, 6, mode&fs.ModeSetgid != 0, 's')
	applySpecialBit(formatted, 9, mode&fs.ModeSticky != 0, 't')

	return string(formatted)
}

// applySpecialBit shows a setuid, setgid or sticky bit in an execute
// position: lower case when the execute bit is set, upper case otherwise.
func applySpecialBit(formatted []byte, position int, set bool, letter byte) {
	if !set {
		return
	}

	if formatted[position] == 'x' {
		formatted[position] = letter

		return
	}

	formatted[position] = letter - 'a' + 'A'
}

// errInvalidMode reports a mode chmod cannot parse.
var errInvalidMode = errors.New("invalid mode")

// applyMode applies an octal or symbolic chmod mode, such as "755",
// "u+x" or "go-w,a+r", to the permission bits of a file.
func applyMode(current fs.FileMode, isDir bool, specification string) (fs.FileMode, error) {
	if octal, err := strconv.ParseUint(specification, 8, 32); err == nil {
		return octalMode(uint32(octal)), nil
	}

	mode := current
	for clause := range strings.SplitSeq(specification, ",") {
		applied, err := applySymbolicClause(mode, isDir, clause)
		if err != nil {
			return 0, err
		}

		mode = applied
	}

	return mode, nil
}

// octalMode converts octal permission digits into a mode.
func octalMode(octal uint32) fs.FileMode {
	mode := fs.FileMode(octal & 0o777)

	if octal&0o4000 != 0 {
		mode |= fs.ModeSetuid
	}

	if octal&0o2000 != 0 {
		mode |= fs.ModeSetgid
	}

	if octal&0o1000 != 0 {
		mode |= fs.ModeSticky
	}

	return mode
}

// applySymbolicClause applies one clause such as "ug+rx" or "o=".
func applySymbolicClause(mode fs.FileMode, isDir bool, clause string) (fs.FileMode, error) {
	operations := strings.TrimLeft(clause, "ugoa")
	targets := clause[:len(clause)-len(operations)]
	if targets == "" || strings.Contains(targets, "a") {
		targets = "ugo"
	}

	if operations == "" {
		return 0, errInvalidMode
	}

	for operations != "" {
		operator := operations[0]
		if operator != '+' && operator != '-' && operator != '=' {
			return 0, errInvalidMode
		}

		remaining := strings.TrimLeft(operations[1:], "rwxXst")
		letters := operations[1 : len(operations)-len(remaining)]
		operations = remaining

		bits := symbolicBits(targets, letters, isDir || mode&0o111 != 0)

		switch operator {
		case '+':
			mode |= bits
		case '-':
			mode &^= bits
		case '=':
			mode = mode&^symbolicBits(targets, "rwxst", true) | bits
		}
	}

	return mode, nil
}

// symbolicBits returns the mode bits of permission letters for the given
// classes. X counts as x when executable is set.
func symbolicBits(targets string, letters string, executable bool) fs.FileMode {
	var bits fs.FileMode

	for _, target := range targets {
		shift := classShift(target)

		for _, letter := range letters {
			switch letter {
			case 'r':
				bits |= 0o4 << shift
			case 'w':
				bits |= 0o2 << shift
			case 'x':
				bits |= 0o1 << shift
			case 'X':
				if executable {
					bits |= 0o1 << shift
				}
			case 's':
				bits |= specialBit(target, fs.ModeSetuid, fs.ModeSetgid)
			case 't':
				if target == 'o' {
					bits |= fs.ModeSticky
				}
			}
		}
	}

	return bits
}

// classShift returns the position of a class's permission bits.
func classShift(target rune) uint {
	switch target {
	case 'u':
		return 6
	case 'g':
		return 3
	default:
		return 0
	}
}

// specialBit returns the setuid bit for the user class and the setgid bit
// for the group class.
func specialBit(target rune, user fs.FileMode, group fs.FileMode) fs.FileMode {
	switch target {
	case 'u':
		return user
	case 'g':
		return group
	default:
		return 0
	}
}
