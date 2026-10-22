package sandboxshell

import (
	"cmp"
	"context"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"unicode"
)

// sortOptions are the flags of sort.
type sortOptions struct {
	reverse    bool
	numeric    bool
	unique     bool
	ignoreCase bool
	stable     bool
	separator  string
	keys       []sortKey
}

// sortKey is a -k field range with its own ordering letters.
type sortKey struct {
	startField int
	endField   int
	numeric    bool
	reverse    bool
}

// runSort carries out "sort [-rnufs] [-t separator] [-k key] [file...]".
func runSort(_ context.Context, call *commandCall) int {
	parsed, ok := call.parseCommandArgs(argSpec{
		valueFlags: []string{
			"-t",
			"-k",
			"-o",
			"--field-separator",
			"--key",
			"--output",
		},
	})
	if !ok {
		return exitUsage
	}

	var options sortOptions
	output := ""

	for _, flag := range parsed.flags {
		switch flag.name {
		case "-r", "--reverse":
			options.reverse = true
		case "-n", "--numeric-sort", "-g", "-h":
			options.numeric = true
		case "-u", "--unique":
			options.unique = true
		case "-f", "--ignore-case":
			options.ignoreCase = true
		case "-s", "--stable":
			options.stable = true
		case "-b", "-d", "-V":
		case "-t", "--field-separator":
			options.separator = flag.value
		case "-k", "--key":
			key, err := parseSortKey(flag.value)
			if err != nil {
				return call.fail("invalid key specification '%s'", flag.value)
			}

			options.keys = append(options.keys, key)
		case "-o", "--output":
			output = flag.value
		default:
			call.invalidFlag(flag)

			return exitUsage
		}
	}

	inputs, ok := call.readInputs(parsed.operands)
	if !ok {
		return exitUsage
	}

	var lines []string
	for _, input := range inputs {
		lines = append(lines, splitLines(input.contents)...)
	}

	sorted := sortLines(lines, options)
	if output == "" {
		_ = call.writeLines(sorted)

		return 0
	}

	contents := strings.Join(sorted, "\n") + "\n"
	if err := call.fs().WriteFile(call.resolve(output), []byte(contents), 0o644); err != nil {
		call.reportFile(output, err)

		return exitUsage
	}

	return 0
}

// parseSortKey parses a key such as "2", "2,3" or "3n".
func parseSortKey(specification string) (sortKey, error) {
	start, end, hasEnd := strings.Cut(specification, ",")

	var key sortKey
	var err error

	key.startField, err = parseKeyField(start, &key)
	if err != nil {
		return sortKey{}, err
	}

	if hasEnd {
		key.endField, err = parseKeyField(end, &key)
		if err != nil {
			return sortKey{}, err
		}
	}

	return key, nil
}

// parseKeyField parses one field position of a key, such as "2n" or "1.3",
// recording its ordering letters on the key. Character offsets are ignored.
func parseKeyField(field string, key *sortKey) (int, error) {
	letters := strings.TrimLeft(field, "0123456789.")
	key.numeric = key.numeric || strings.ContainsAny(letters, "ngh")
	key.reverse = key.reverse || strings.Contains(letters, "r")

	number, _, _ := strings.Cut(field[:len(field)-len(letters)], ".")

	return strconv.Atoi(number)
}

// sortLines sorts lines with sort's options.
func sortLines(lines []string, options sortOptions) []string {
	sorted := slices.Clone(lines)

	slices.SortStableFunc(sorted, func(first, second string) int {
		return compareSortLines(first, second, options)
	})

	if !options.unique {
		return sorted
	}

	return slices.CompactFunc(sorted, func(first, second string) bool {
		return compareSortKeys(first, second, options) == 0
	})
}

// compareSortLines orders two lines by their keys and, unless the sort is
// stable, by the whole line as a last resort.
func compareSortLines(first string, second string, options sortOptions) int {
	order := compareSortKeys(first, second, options)
	if order == 0 && !options.stable && !options.unique {
		order = strings.Compare(first, second)
		if options.reverse {
			order = -order
		}
	}

	return order
}

// compareSortKeys orders two lines by their keys, or by the whole line
// without keys.
func compareSortKeys(first string, second string, options sortOptions) int {
	keys := options.keys
	if len(keys) == 0 {
		keys = []sortKey{
			{
				startField: 1,
			},
		}
	}

	for _, key := range keys {
		firstValue := sortKeyValue(first, key, options.separator, len(options.keys) == 0)
		secondValue := sortKeyValue(second, key, options.separator, len(options.keys) == 0)

		order := compareSortValues(firstValue, secondValue, options.numeric || key.numeric, options.ignoreCase)
		if options.reverse != key.reverse {
			order = -order
		}

		if order != 0 {
			return order
		}
	}

	return 0
}

// sortKeyValue extracts the text a key compares.
func sortKeyValue(line string, key sortKey, separator string, wholeLine bool) string {
	if wholeLine {
		return line
	}

	fields := splitSortFields(line, separator)
	start := key.startField - 1
	if start >= len(fields) {
		return ""
	}

	end := len(fields)
	if key.endField > 0 {
		end = min(key.endField, len(fields))
	}

	if end <= start {
		return ""
	}

	joiner := separator
	if joiner == "" {
		joiner = " "
	}

	return strings.Join(fields[start:end], joiner)
}

// splitSortFields splits a line at a separator, or at runs of blanks.
func splitSortFields(line string, separator string) []string {
	if separator == "" {
		return strings.Fields(line)
	}

	return strings.Split(line, separator)
}

// compareSortValues compares two key values numerically or as text.
func compareSortValues(first string, second string, numeric bool, ignoreCase bool) int {
	if numeric {
		return cmp.Compare(leadingNumber(first), leadingNumber(second))
	}

	if ignoreCase {
		return strings.Compare(strings.ToLower(first), strings.ToLower(second))
	}

	return strings.Compare(first, second)
}

// leadingNumber parses the number at the start of text, as sort -n does;
// text without one counts as zero.
func leadingNumber(text string) float64 {
	text = strings.TrimSpace(text)

	end := 0
	for end < len(text) && strings.ContainsRune("+-.0123456789", rune(text[end])) {
		end++
	}

	for ; end > 0; end-- {
		if number, err := strconv.ParseFloat(text[:end], 64); err == nil {
			return number
		}
	}

	return 0
}

// runTranslate carries out "tr [-d] [-s] [-c] set1 [set2]".
func runTranslate(_ context.Context, call *commandCall) int {
	parsed, ok := call.parseCommandArgs(argSpec{})
	if !ok {
		return exitFailure
	}

	var options translateOptions
	for _, flag := range parsed.flags {
		switch flag.name {
		case "-d", "--delete":
			options.deleteSet = true
		case "-s", "--squeeze-repeats":
			options.squeeze = true
		case "-c", "-C", "--complement":
			options.complement = true
		default:
			return call.invalidFlag(flag)
		}
	}

	operands := parsed.operands
	needsSecondSet := !options.deleteSet && !options.squeeze

	switch {
	case len(operands) == 0:
		return call.fail("missing operand")
	case needsSecondSet && len(operands) < 2:
		return call.fail("missing operand after '%s'", operands[0])
	}

	options.from = expandCharacterSet(operands[0])
	if len(operands) > 1 {
		options.to = expandCharacterSet(operands[1])
	}

	inputs, _ := call.readInputs(nil)
	call.writeOutput(translate(string(inputs[0].contents), options))

	return 0
}

// translateOptions are the flags and sets of tr.
type translateOptions struct {
	from       []rune
	to         []rune
	deleteSet  bool
	squeeze    bool
	complement bool
}

// translate applies tr to text. Squeezing applies to the second set when
// there is one, and to the first set otherwise.
func translate(text string, options translateOptions) string {
	var output strings.Builder
	var previous rune
	havePrevious := false

	for _, character := range text {
		inFrom := slices.Contains(options.from, character) != options.complement

		if options.deleteSet && inFrom {
			continue
		}

		if !options.deleteSet && inFrom && len(options.to) > 0 {
			character = translateRune(character, options)
		}

		inSqueezeSet := inFrom
		if len(options.to) > 0 {
			inSqueezeSet = slices.Contains(options.to, character)
		}

		if options.squeeze && havePrevious && character == previous && inSqueezeSet {
			continue
		}

		output.WriteRune(character)
		previous = character
		havePrevious = true
	}

	return output.String()
}

// translateRune maps a character of the first set to the second set; the
// last character of the second set repeats when it is shorter.
func translateRune(character rune, options translateOptions) rune {
	if options.complement {
		return options.to[len(options.to)-1]
	}

	index := slices.Index(options.from, character)

	return options.to[min(index, len(options.to)-1)]
}

// expandCharacterSet expands a tr set with escapes, ranges and character
// classes such as [:upper:].
func expandCharacterSet(set string) []rune {
	var expanded []rune
	characters := []rune(unescapeBackslashes(set))

	for index := 0; index < len(characters); index++ {
		if class, length, ok := characterClass(characters[index:]); ok {
			expanded = append(expanded, class...)
			index += length - 1

			continue
		}

		isRange := index+2 < len(characters) && characters[index+1] == '-' && characters[index] <= characters[index+2]
		if !isRange {
			expanded = append(expanded, characters[index])

			continue
		}

		for character := characters[index]; character <= characters[index+2]; character++ {
			expanded = append(expanded, character)
		}

		index += 2
	}

	return expanded
}

// characterClasses are the POSIX classes tr understands.
var characterClasses = map[string]func(rune) bool{
	"alnum": func(character rune) bool {
		return unicode.IsLetter(character) || unicode.IsDigit(character)
	},
	"alpha": unicode.IsLetter,
	"blank": func(character rune) bool {
		return character == ' ' || character == '\t'
	},
	"digit": unicode.IsDigit,
	"lower": unicode.IsLower,
	"punct": unicode.IsPunct,
	"space": unicode.IsSpace,
	"upper": unicode.IsUpper,
	"xdigit": func(character rune) bool {
		return strings.ContainsRune("0123456789abcdefABCDEF", character)
	},
}

// characterClass expands a "[:name:]" class at the start of characters,
// limited to ASCII, and returns its length in the set.
func characterClass(characters []rune) ([]rune, int, bool) {
	text := string(characters)
	if !strings.HasPrefix(text, "[:") {
		return nil, 0, false
	}

	name, _, found := strings.Cut(text[2:], ":]")
	matches, known := characterClasses[name]
	if !found || !known {
		return nil, 0, false
	}

	var class []rune
	for character := rune(0); character < 128; character++ {
		if matches(character) {
			class = append(class, character)
		}
	}

	return class, len([]rune("[:" + name + ":]")), true
}

// unescapeBackslashes expands the backslash escapes of tr and printf
// formats: \n, \t, \r, \\, \a, \b, \f, \v and octal \NNN.
func unescapeBackslashes(text string) string {
	var output strings.Builder

	for index := 0; index < len(text); index++ {
		if text[index] != '\\' || index+1 == len(text) {
			output.WriteByte(text[index])

			continue
		}

		index++
		if value, length, ok := octalEscape(text[index:]); ok {
			output.WriteByte(value)
			index += length - 1

			continue
		}

		output.WriteString(simpleEscape(text[index]))
	}

	return output.String()
}

// octalEscape parses up to three octal digits.
func octalEscape(text string) (byte, int, bool) {
	length := 0
	for length < 3 && length < len(text) && text[length] >= '0' && text[length] <= '7' {
		length++
	}

	if length == 0 {
		return 0, 0, false
	}

	value, err := strconv.ParseUint(text[:length], 8, 8)
	if err != nil {
		return 0, 0, false
	}

	return byte(value), length, true
}

// simpleEscape returns the character of a one-letter escape.
func simpleEscape(letter byte) string {
	switch letter {
	case 'n':
		return "\n"
	case 't':
		return "\t"
	case 'r':
		return "\r"
	case 'a':
		return "\a"
	case 'b':
		return "\b"
	case 'f':
		return "\f"
	case 'v':
		return "\v"
	case '\\':
		return "\\"
	default:
		return "\\" + string(letter)
	}
}

// runCut carries out "cut -f list [-d delimiter] [-s]" and "cut -c list".
func runCut(_ context.Context, call *commandCall) int {
	parsed, ok := call.parseCommandArgs(argSpec{
		valueFlags: []string{
			"-f",
			"-c",
			"-b",
			"-d",
			"--fields",
			"--characters",
			"--bytes",
			"--delimiter",
			"--output-delimiter",
		},
	})
	if !ok {
		return exitFailure
	}

	options := cutOptions{
		delimiter: "\t",
	}

	list := ""
	for _, flag := range parsed.flags {
		switch flag.name {
		case "-f", "--fields":
			list = flag.value
		case "-c", "--characters", "-b", "--bytes":
			list = flag.value
			options.characters = true
		case "-d", "--delimiter":
			options.delimiter = flag.value
		case "--output-delimiter":
			options.outputDelimiter = &flag.value
		case "-s", "--only-delimited":
			options.onlyDelimited = true
		default:
			return call.invalidFlag(flag)
		}
	}

	ranges, err := parseCutList(list)
	if err != nil {
		return call.fail("you must specify a list of bytes, characters, or fields")
	}

	options.ranges = ranges

	inputs, ok := call.readInputs(parsed.operands)
	for _, input := range inputs {
		for _, line := range splitLines(input.contents) {
			if selected, keep := cutLine(line, options); keep {
				call.writeOutput(selected + "\n")
			}
		}
	}

	return statusOf(ok)
}

// cutOptions are the flags of cut.
type cutOptions struct {
	ranges          []cutRange
	characters      bool
	delimiter       string
	outputDelimiter *string
	onlyDelimited   bool
}

// cutRange is an inclusive 1-based range; zero end means to the end.
type cutRange struct {
	start int
	end   int
}

// contains reports whether a position lies in the range.
func (r cutRange) contains(position int) bool {
	return position >= r.start && (r.end == 0 || position <= r.end)
}

// parseCutList parses a list such as "1,3-5,7-".
func parseCutList(list string) ([]cutRange, error) {
	if list == "" {
		return nil, fmt.Errorf("empty list")
	}

	var ranges []cutRange
	for item := range strings.SplitSeq(list, ",") {
		startText, endText, isRange := strings.Cut(item, "-")

		start, end := 1, 0
		var err error

		if startText != "" {
			start, err = strconv.Atoi(startText)
		}

		switch {
		case err != nil:
			return nil, err
		case !isRange:
			end = start
		case endText != "":
			end, err = strconv.Atoi(endText)
		}

		if err != nil || start < 1 {
			return nil, fmt.Errorf("invalid range %q", item)
		}

		ranges = append(ranges, cutRange{
			start: start,
			end:   end,
		})
	}

	return ranges, nil
}

// cutLine selects the characters or fields of one line, and reports
// whether the line is printed.
func cutLine(line string, options cutOptions) (string, bool) {
	if options.characters {
		var selected strings.Builder
		for position, character := range []rune(line) {
			if inCutRanges(options.ranges, position+1) {
				selected.WriteRune(character)
			}
		}

		return selected.String(), true
	}

	if !strings.Contains(line, options.delimiter) {
		return line, !options.onlyDelimited
	}

	var selected []string
	for position, field := range strings.Split(line, options.delimiter) {
		if inCutRanges(options.ranges, position+1) {
			selected = append(selected, field)
		}
	}

	joiner := options.delimiter
	if options.outputDelimiter != nil {
		joiner = *options.outputDelimiter
	}

	return strings.Join(selected, joiner), true
}

// inCutRanges reports whether a position lies in any range.
func inCutRanges(ranges []cutRange, position int) bool {
	return slices.ContainsFunc(ranges, func(item cutRange) bool {
		return item.contains(position)
	})
}
