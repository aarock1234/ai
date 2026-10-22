// Package sandboxshell is an isolated, in-memory sandbox session: the Go
// port of the Vercel AI SDK @ai-sdk/sandbox-just-bash package. It is a
// separate module, so that the core module does not depend on a shell
// interpreter.
//
// New creates a fresh sandbox with its own virtual filesystem and applies
// an optional harness template, as upstream's
// createJustBashNetworkSandboxSession does. Commands are bash scripts
// interpreted in-process by mvdan.cc/sh/v3/interp. The filesystem is shared
// by every command and file operation of the session, while shell state
// (variables, functions, working directory) starts fresh for every Run and
// Spawn, as in just-bash:
//
//	session, err := sandboxshell.New(ctx, sandboxshell.Options{
//		WorkingDirectory: "/work",
//	})
//	if err != nil {
//		return err
//	}
//	defer session.Destroy(ctx)
//
//	tools := session.Restricted()
//	err = tools.WriteTextFile(ctx, sandbox.WriteTextFileOptions{
//		Path:    "/work/hello.txt",
//		Content: "hi",
//	})
//
//	result, err := tools.Run(ctx, sandbox.ProcessOptions{
//		Command: "cat /work/hello.txt",
//	})
//
// The sandbox starts with just-bash's layout: /home/user (the default
// working directory and HOME), /bin with a stub for every command, /usr/bin,
// /usr/local/bin, /tmp and the device files /dev/null, /dev/stdin,
// /dev/stdout and /dev/stderr. The commands are Go implementations of
// just-bash's command set: base64, basename, bash, cat, chmod, clear, cp,
// cut, date, dirname, egrep, env, fgrep, find, grep, head, hostname, ls,
// md5sum, mkdir, mktemp, mv, nl, printenv, realpath, rev, rm, rmdir, sed,
// seq, sh, sha1sum, sha256sum, sleep, sort, stat, tac, tail, tee, timeout,
// touch, tr, uniq, wc, which, whoami, xargs and yes, plus the interpreter's
// builtins such as echo, printf, test, cd, read and export. Executable
// files of the virtual filesystem run as shell scripts. Any other name
// fails with "command not found" and exit code 127.
//
// # Isolation
//
// A sandbox never touches the host filesystem, host programs or the
// network:
//
//   - Every file operation goes to the session's in-memory filesystem: the
//     session's file methods, redirections, here strings, globbing, cd,
//     test operators, source and every command. Host files are never
//     opened, created, listed or inspected.
//   - Commands are Go functions; host programs are never executed. The
//     interpreter's lookups that would search the host PATH (type,
//     command -v, source) are answered from the virtual filesystem, pwd -P
//     does not resolve host links, ~name never reads the host user
//     database, and the interpreter's temporary directory is set to a path
//     the host rejects.
//   - Process substitution and the -O and -G file tests, which would create
//     host FIFOs or read the host user database, are rejected with an error
//     before they run, including in eval, trap and sourced files.
//   - Commands see only the default variables, Options.Env and the
//     command's Env; the host environment is not inherited. date reports
//     UTC, because loading another time zone would read the host's zone
//     database.
//   - There is no network: no command opens connections, Ports is empty,
//     PortEndpoint fails with harness.CapabilityUnsupportedError and network
//     policies are not supported. Bridge-backed harness adapters cannot use
//     this sandbox, as with upstream.
//
// The sandbox runs inside the host process; it is not a security boundary
// against resource exhaustion. CPU time, memory and output are bounded only
// by the caller's context and Destroy. The interpreter uses anonymous OS
// pipes for pipelines and here-documents, $$ and $PPID report the host
// process IDs, and date reads the host clock.
//
// # Go adaptations
//
// Cancellation comes from the context; Kill makes Wait report exit code
// 137. There is no symbolic or hard link support, so ln is not provided and
// realpath only cleans paths. Shell functions are not reported by type and
// command -v, because the interpreter does not expose them to its handlers.
// Upstream's adapters for existing just-bash instances, its deprecated
// provider, its overlay and read-write host filesystems, custom commands,
// execution limits and the optional curl, Python, JavaScript, SQLite and jq
// commands are not ported. The interpreter applies the host's path rules,
// so on Windows hosts cd and glob patterns with absolute paths are
// unreliable; use relative paths there.
package sandboxshell
