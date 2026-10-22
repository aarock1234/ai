package sandboxshell

import (
	"io/fs"
	"path"
	"slices"
	"strings"
	"sync"
	"time"
)

// memFS is the in-memory filesystem of one session. It holds regular files
// and directories only; there are no symbolic or hard links. Paths are
// absolute, slash-separated and clean: callers resolve them with
// resolvePath first. It is safe for concurrent use.
//
// File contents are never modified in place. A write either appends past
// the end of the current contents or replaces them with a new slice, so a
// slice returned by ReadFile stays valid while writers continue.
type memFS struct {
	mu   sync.RWMutex
	root *memNode
}

// memNode is a file or a directory.
type memNode struct {
	mode     fs.FileMode
	modTime  time.Time
	data     []byte
	children map[string]*memNode
}

// newMemFS returns a filesystem with an empty root directory.
func newMemFS() *memFS {
	return &memFS{
		root: newDirectoryNode(0o755),
	}
}

// newDirectoryNode returns an empty directory.
func newDirectoryNode(perm fs.FileMode) *memNode {
	return &memNode{
		mode:     fs.ModeDir | perm,
		modTime:  time.Now(),
		children: make(map[string]*memNode),
	}
}

// newFileNode returns an empty regular file.
func newFileNode(perm fs.FileMode) *memNode {
	return &memNode{
		mode:    perm,
		modTime: time.Now(),
	}
}

// isDir reports whether the node is a directory.
func (n *memNode) isDir() bool {
	return n.mode.IsDir()
}

// Stat describes a file or directory.
func (f *memFS) Stat(name string) (fs.FileInfo, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()

	node, err := f.lookup(name)
	if err != nil {
		return nil, pathError("stat", name, err)
	}

	return newFileInfo(path.Base(name), node), nil
}

// ReadDir lists a directory sorted by name.
func (f *memFS) ReadDir(name string) ([]fs.DirEntry, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()

	node, err := f.lookup(name)
	if err != nil {
		return nil, pathError("readdir", name, err)
	}

	if !node.isDir() {
		return nil, pathError("readdir", name, errNotDirectory)
	}

	entries := make([]fs.DirEntry, 0, len(node.children))
	for childName, child := range node.children {
		entries = append(entries, fs.FileInfoToDirEntry(newFileInfo(childName, child)))
	}

	slices.SortFunc(entries, func(first, second fs.DirEntry) int {
		return strings.Compare(first.Name(), second.Name())
	})

	return entries, nil
}

// ReadFile returns the contents of a regular file. The caller must not
// modify the returned slice.
func (f *memFS) ReadFile(name string) ([]byte, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()

	node, err := f.lookup(name)
	if err != nil {
		return nil, pathError("open", name, err)
	}

	if node.isDir() {
		return nil, pathError("read", name, errIsDirectory)
	}

	return slices.Clip(node.data), nil
}

// WriteFile replaces the contents of a file, creating it with perm when it
// does not exist. The parent directory must exist.
func (f *memFS) WriteFile(name string, data []byte, perm fs.FileMode) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	node, err := f.createFile(name, perm)
	if err != nil {
		return pathError("open", name, err)
	}

	node.data = slices.Clone(data)
	node.modTime = time.Now()

	return nil
}

// AppendFile appends to a file, creating it with perm when it does not
// exist. The parent directory must exist.
func (f *memFS) AppendFile(name string, data []byte, perm fs.FileMode) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	node, err := f.createFile(name, perm)
	if err != nil {
		return pathError("open", name, err)
	}

	node.data = append(node.data, data...)
	node.modTime = time.Now()

	return nil
}

// Mkdir creates one directory. The parent must exist.
func (f *memFS) Mkdir(name string, perm fs.FileMode) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	parent, base, err := f.lookupParent(name)
	if err != nil {
		return pathError("mkdir", name, err)
	}

	if _, exists := parent.children[base]; exists {
		return pathError("mkdir", name, errFileExists)
	}

	parent.children[base] = newDirectoryNode(perm)
	parent.modTime = time.Now()

	return nil
}

// MkdirAll creates a directory and its missing parents. An existing
// directory is not an error.
func (f *memFS) MkdirAll(name string, perm fs.FileMode) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	node := f.root
	for _, part := range splitPath(name) {
		child, exists := node.children[part]
		if !exists {
			child = newDirectoryNode(perm)
			node.children[part] = child
			node.modTime = time.Now()
		}

		if !child.isDir() {
			return pathError("mkdir", name, errNotDirectory)
		}

		node = child
	}

	return nil
}

// Remove removes a file or an empty directory.
func (f *memFS) Remove(name string) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	parent, base, err := f.lookupParent(name)
	if err != nil {
		return pathError("remove", name, err)
	}

	node, exists := parent.children[base]
	if !exists {
		return pathError("remove", name, errNoSuchFile)
	}

	if node.isDir() && len(node.children) > 0 {
		return pathError("remove", name, errDirectoryNotEmpty)
	}

	delete(parent.children, base)
	parent.modTime = time.Now()

	return nil
}

// RemoveAll removes a file or a directory tree. A missing path is not an
// error.
func (f *memFS) RemoveAll(name string) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	parent, base, err := f.lookupParent(name)
	if isNotExist(err) {
		return nil
	}

	if err != nil {
		return pathError("remove", name, err)
	}

	delete(parent.children, base)
	parent.modTime = time.Now()

	return nil
}

// Rename moves a file or directory. An existing file at the destination is
// replaced; an existing directory is replaced only by a directory and only
// when it is empty.
func (f *memFS) Rename(from string, to string) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	if from == to {
		return nil
	}

	if isWithin(to, from) {
		return pathError("rename", to, errInvalidArgument)
	}

	sourceParent, sourceBase, err := f.lookupParent(from)
	if err != nil {
		return pathError("rename", from, err)
	}

	source, exists := sourceParent.children[sourceBase]
	if !exists {
		return pathError("rename", from, errNoSuchFile)
	}

	targetParent, targetBase, err := f.lookupParent(to)
	if err != nil {
		return pathError("rename", to, err)
	}

	if err := checkReplaceable(source, targetParent.children[targetBase]); err != nil {
		return pathError("rename", to, err)
	}

	delete(sourceParent.children, sourceBase)
	targetParent.children[targetBase] = source
	sourceParent.modTime = time.Now()
	targetParent.modTime = time.Now()

	return nil
}

// Chmod changes the permission bits of a file or directory.
func (f *memFS) Chmod(name string, mode fs.FileMode) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	node, err := f.lookup(name)
	if err != nil {
		return pathError("chmod", name, err)
	}

	node.mode = node.mode.Type() | mode&permissionBits

	return nil
}

// Touch sets the modification time of a file or directory, creating an
// empty file when it does not exist and create is set.
func (f *memFS) Touch(name string, modTime time.Time, create bool) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	node, err := f.lookup(name)
	if isNotExist(err) && create {
		node, err = f.createFile(name, 0o644)
	}

	if err != nil {
		return pathError("touch", name, err)
	}

	node.modTime = modTime

	return nil
}

// CreateDevice creates a character device node, such as /dev/null, so that
// the shell's device files can be listed and tested. Opening devices is
// handled by the shell, not by the filesystem.
func (f *memFS) CreateDevice(name string) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	node, err := f.createFile(name, 0o666)
	if err != nil {
		return pathError("mknod", name, err)
	}

	node.mode = fs.ModeDevice | fs.ModeCharDevice | 0o666

	return nil
}

// Clear removes everything, releasing the memory of the files.
func (f *memFS) Clear() {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.root = newDirectoryNode(0o755)
}

// OpenFile opens a file with os.OpenFile flags for the interpreter's
// redirections.
func (f *memFS) OpenFile(name string, flag int, perm fs.FileMode) (*memFile, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	node, err := f.openNode(name, flag, perm)
	if err != nil {
		return nil, pathError("open", name, err)
	}

	return &memFile{
		fs:   f,
		node: node,
		flag: flag,
	}, nil
}

// lookup returns the node at name. The caller holds mu.
func (f *memFS) lookup(name string) (*memNode, error) {
	node := f.root
	for _, part := range splitPath(name) {
		if !node.isDir() {
			return nil, errNotDirectory
		}

		child, exists := node.children[part]
		if !exists {
			return nil, errNoSuchFile
		}

		node = child
	}

	return node, nil
}

// lookupParent returns the directory containing name and the base name of
// name. The caller holds mu.
func (f *memFS) lookupParent(name string) (*memNode, string, error) {
	if name == "/" {
		return nil, "", errInvalidArgument
	}

	parent, err := f.lookup(path.Dir(name))
	if err != nil {
		return nil, "", err
	}

	if !parent.isDir() {
		return nil, "", errNotDirectory
	}

	return parent, path.Base(name), nil
}

// createFile returns the regular file at name, creating it when it does not
// exist. The caller holds mu.
func (f *memFS) createFile(name string, perm fs.FileMode) (*memNode, error) {
	parent, base, err := f.lookupParent(name)
	if err != nil {
		return nil, err
	}

	node, exists := parent.children[base]
	if exists && node.isDir() {
		return nil, errIsDirectory
	}

	if exists {
		return node, nil
	}

	node = newFileNode(perm & permissionBits)
	parent.children[base] = node
	parent.modTime = time.Now()

	return node, nil
}

// openNode applies the creation and truncation flags of an open call. The
// caller holds mu.
func (f *memFS) openNode(name string, flag int, perm fs.FileMode) (*memNode, error) {
	node, err := f.lookup(name)

	switch {
	case isNotExist(err) && flag&openCreate != 0:
		return f.createFile(name, perm)
	case err != nil:
		return nil, err
	case flag&openCreate != 0 && flag&openExclusive != 0:
		return nil, errFileExists
	case node.isDir() && isWritable(flag):
		return nil, errIsDirectory
	}

	if flag&openTruncate != 0 && isWritable(flag) {
		node.data = nil
		node.modTime = time.Now()
	}

	return node, nil
}

// checkReplaceable reports whether source may replace target in a rename.
// A nil target is always replaceable.
func checkReplaceable(source *memNode, target *memNode) error {
	switch {
	case target == nil:
		return nil
	case target.isDir() && !source.isDir():
		return errIsDirectory
	case !target.isDir() && source.isDir():
		return errNotDirectory
	case target.isDir() && len(target.children) > 0:
		return errDirectoryNotEmpty
	default:
		return nil
	}
}

// splitPath returns the components of an absolute path.
func splitPath(name string) []string {
	trimmed := strings.Trim(name, "/")
	if trimmed == "" {
		return nil
	}

	return strings.Split(trimmed, "/")
}

// isWithin reports whether name lies strictly inside directory.
func isWithin(name string, directory string) bool {
	if directory == "/" {
		return name != "/"
	}

	return strings.HasPrefix(name, directory+"/")
}

// memFile is an open file of the interpreter's redirections.
type memFile struct {
	fs     *memFS
	node   *memNode
	flag   int
	offset int
}

// Read reads from the current offset.
func (m *memFile) Read(data []byte) (int, error) {
	if isWriteOnly(m.flag) {
		return 0, errBadDescriptor
	}

	m.fs.mu.Lock()
	defer m.fs.mu.Unlock()

	if m.node.isDir() {
		return 0, errIsDirectory
	}

	if m.offset >= len(m.node.data) {
		return 0, errEndOfFile
	}

	read := copy(data, m.node.data[m.offset:])
	m.offset += read

	return read, nil
}

// Write writes at the current offset, or at the end of the file when the
// file was opened for appending.
func (m *memFile) Write(data []byte) (int, error) {
	if !isWritable(m.flag) {
		return 0, errBadDescriptor
	}

	m.fs.mu.Lock()
	defer m.fs.mu.Unlock()

	if m.flag&openAppend != 0 {
		m.offset = len(m.node.data)
	}

	m.node.data = writeAt(m.node.data, m.offset, data)
	m.node.modTime = time.Now()
	m.offset += len(data)

	return len(data), nil
}

// Close closes the file. It holds no resources.
func (m *memFile) Close() error {
	return nil
}

// writeAt returns contents with data written at offset. Appending at the
// end extends the existing slice; any other write copies it, so that slices
// handed out earlier never change.
func writeAt(contents []byte, offset int, data []byte) []byte {
	if offset == len(contents) {
		return append(contents, data...)
	}

	written := make([]byte, max(len(contents), offset+len(data)))
	copy(written, contents)
	copy(written[offset:], data)

	return written
}

// memFileInfo describes a node.
type memFileInfo struct {
	name    string
	size    int64
	mode    fs.FileMode
	modTime time.Time
}

// newFileInfo describes node under name. The caller holds the lock of the
// node's filesystem.
func newFileInfo(name string, node *memNode) memFileInfo {
	return memFileInfo{
		name:    name,
		size:    int64(len(node.data)),
		mode:    node.mode,
		modTime: node.modTime,
	}
}

// Name returns the base name.
func (i memFileInfo) Name() string {
	return i.name
}

// Size returns the length of a file's contents.
func (i memFileInfo) Size() int64 {
	return i.size
}

// Mode returns the type and permission bits.
func (i memFileInfo) Mode() fs.FileMode {
	return i.mode
}

// ModTime returns the modification time.
func (i memFileInfo) ModTime() time.Time {
	return i.modTime
}

// IsDir reports whether the node is a directory.
func (i memFileInfo) IsDir() bool {
	return i.mode.IsDir()
}

// Sys returns nil: virtual files have no system-specific data.
func (i memFileInfo) Sys() any {
	return nil
}
