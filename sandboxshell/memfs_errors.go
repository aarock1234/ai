package sandboxshell

import (
	"errors"
	"io"
	"io/fs"
	"os"
)

// Flags of OpenFile, as the interpreter passes them.
const (
	openWriteOnly = os.O_WRONLY
	openReadWrite = os.O_RDWR
	openCreate    = os.O_CREATE
	openExclusive = os.O_EXCL
	openTruncate  = os.O_TRUNC
	openAppend    = os.O_APPEND
)

// permissionBits are the mode bits a node keeps besides its type.
const permissionBits = fs.ModePerm | fs.ModeSticky | fs.ModeSetuid | fs.ModeSetgid

// Filesystem errors. Their messages follow the C library's descriptions,
// which shell commands print.
var (
	errNoSuchFile = fsError{
		message: "no such file or directory",
		target:  fs.ErrNotExist,
	}
	errFileExists = fsError{
		message: "file exists",
		target:  fs.ErrExist,
	}
	errIsDirectory = fsError{
		message: "is a directory",
	}
	errNotDirectory = fsError{
		message: "not a directory",
	}
	errDirectoryNotEmpty = fsError{
		message: "directory not empty",
	}
	errInvalidArgument = fsError{
		message: "invalid argument",
		target:  fs.ErrInvalid,
	}
	errBadDescriptor = fsError{
		message: "bad file descriptor",
	}
	errEndOfFile = io.EOF
)

// fsError is a filesystem error that matches an io/fs sentinel with
// errors.Is.
type fsError struct {
	message string
	target  error
}

// Error returns the description.
func (e fsError) Error() string {
	return e.message
}

// Is reports whether target is the io/fs sentinel the error stands for.
func (e fsError) Is(target error) bool {
	return e.target != nil && target == e.target
}

// pathError wraps err with an operation and a path. The interpreter only
// reports *fs.PathError values from its handlers as ordinary failures.
func pathError(operation string, name string, err error) error {
	return &fs.PathError{
		Op:   operation,
		Path: name,
		Err:  err,
	}
}

// isNotExist reports whether err means that a path does not exist.
func isNotExist(err error) bool {
	return errors.Is(err, fs.ErrNotExist)
}

// isExist reports whether err means that a path already exists.
func isExist(err error) bool {
	return errors.Is(err, fs.ErrExist)
}

// isWritable reports whether open flags allow writing.
func isWritable(flag int) bool {
	return flag&(openWriteOnly|openReadWrite) != 0
}

// isWriteOnly reports whether open flags forbid reading.
func isWriteOnly(flag int) bool {
	return flag&openWriteOnly != 0
}
