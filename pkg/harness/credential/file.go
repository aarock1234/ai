package credential

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// refreshLocks holds one mutex per stored credential, guarded by
// refreshLocksMu.
var (
	refreshLocksMu sync.Mutex
	refreshLocks   = map[string]*sync.Mutex{}
)

// LockRefresh serializes the reads, refreshes and writes of one stored
// credential within the process, so that concurrent sessions do not spend
// the same refresh token twice or overwrite each other's rotated tokens.
// Key identifies the credential, such as the absolute path of its file. It
// returns the function that releases the lock.
func LockRefresh(key string) func() {
	lock := refreshLock(key)
	lock.Lock()

	return lock.Unlock
}

// refreshLock returns the mutex of a credential, creating it on first use.
func refreshLock(key string) *sync.Mutex {
	refreshLocksMu.Lock()
	defer refreshLocksMu.Unlock()

	lock, ok := refreshLocks[key]
	if !ok {
		lock = &sync.Mutex{}
		refreshLocks[key] = lock
	}

	return lock
}

// WriteFileAtomically replaces the file at path with content, readable and
// writable only by its owner. It writes a temporary file in the same
// directory, flushes it and renames it over path, so that readers see the
// old or the new content but never a partial file. Missing directories are
// created, accessible only by the owner.
func WriteFileAtomically(path string, content []byte) error {
	directory := filepath.Dir(path)

	if err := os.MkdirAll(directory, 0o700); err != nil {
		return fmt.Errorf("create directory of %s: %w", path, err)
	}

	// CreateTemp creates the file with mode 0600.
	temporary, err := os.CreateTemp(directory, "."+filepath.Base(path)+".*.tmp")
	if err != nil {
		return fmt.Errorf("create temporary file for %s: %w", path, err)
	}

	temporaryPath := temporary.Name()

	defer func() {
		_ = os.Remove(temporaryPath)
	}()

	if err := writeAndClose(temporary, content); err != nil {
		return fmt.Errorf("write %s: %w", path, err)
	}

	if err := os.Rename(temporaryPath, path); err != nil {
		return fmt.Errorf("replace %s: %w", path, err)
	}

	return nil
}

// writeAndClose writes content to file, flushes it to disk and closes it.
func writeAndClose(file *os.File, content []byte) error {
	_, writeErr := file.Write(content)

	var syncErr error
	if writeErr == nil {
		syncErr = file.Sync()
	}

	closeErr := file.Close()

	return errors.Join(writeErr, syncErr, closeErr)
}
