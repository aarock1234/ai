package sandboxshell

import (
	"maps"
	"strings"

	"mvdan.cc/sh/v3/expand"
)

// The virtual identity and layout of a sandbox, following just-bash's
// default environment.
const (
	defaultWorkingDirectory = "/home/user"
	virtualUser             = "user"
	virtualUserID           = "1000"
	virtualHostname         = "localhost"
	virtualSearchPath       = "/usr/local/bin:/usr/bin:/bin"
	virtualShell            = "/bin/bash"
	temporaryDirectory      = "/tmp"
)

// homeLookupPrefix prefixes the variables the interpreter consults before
// asking the host's user database for the home directory of "~name".
const homeLookupPrefix = "HOME "

// defaultVariables returns the variables every command starts with, before
// the session's and the command's variables.
func defaultVariables(workingDirectory string) map[string]string {
	return map[string]string{
		"HOME":     defaultWorkingDirectory,
		"HOSTNAME": virtualHostname,
		"LOGNAME":  virtualUser,
		"PATH":     virtualSearchPath,
		"PWD":      workingDirectory,
		"SHELL":    virtualShell,
		"TMPDIR":   temporaryDirectory,
		"USER":     virtualUser,
	}
}

// mergeVariables merges variable maps; later maps take precedence.
func mergeVariables(layers ...map[string]string) map[string]string {
	merged := make(map[string]string)
	for _, layer := range layers {
		maps.Copy(merged, layer)
	}

	return merged
}

// shellEnviron is the environment a script starts with. It answers the
// interpreter's "HOME name" lookups itself, so that "~name" never reaches
// the host's user database.
type shellEnviron map[string]expand.Variable

// newShellEnviron returns an environment of exported variables plus the
// virtual user and group IDs, which the interpreter would otherwise read
// from the host process.
func newShellEnviron(exported map[string]string) shellEnviron {
	environ := make(shellEnviron, len(exported)+3)

	for name, value := range exported {
		environ[name] = expand.Variable{
			Set:      true,
			Exported: true,
			Kind:     expand.String,
			Str:      value,
		}
	}

	for _, name := range []string{"UID", "EUID", "GID"} {
		environ[name] = expand.Variable{
			Set:      true,
			ReadOnly: true,
			Kind:     expand.String,
			Str:      virtualUserID,
		}
	}

	return environ
}

// Get returns a variable. "HOME name" resolves the home directory of a
// user name for tilde expansion.
func (e shellEnviron) Get(name string) expand.Variable {
	if user, ok := strings.CutPrefix(name, homeLookupPrefix); ok {
		return expand.Variable{
			Set:  true,
			Kind: expand.String,
			Str:  virtualHomeDirectory(user),
		}
	}

	return e[name]
}

// Each visits every variable.
func (e shellEnviron) Each(visit func(name string, variable expand.Variable) bool) {
	for name, variable := range e {
		if !visit(name, variable) {
			return
		}
	}
}

// virtualHomeDirectory returns the home directory of a user name: the
// virtual user's home, /root for root, and "~name" unchanged otherwise, as
// bash leaves the tilde prefix of an unknown user.
func virtualHomeDirectory(user string) string {
	switch user {
	case virtualUser:
		return defaultWorkingDirectory
	case "root":
		return "/root"
	default:
		return "~" + user
	}
}

// exportedVariables returns the exported string variables of env, which a
// child process inherits.
func exportedVariables(env expand.Environ) map[string]string {
	exported := make(map[string]string)

	env.Each(func(name string, variable expand.Variable) bool {
		if variable.IsSet() && variable.Exported && variable.Kind == expand.String {
			exported[name] = variable.Str
		} else {
			delete(exported, name)
		}

		return true
	})

	return exported
}

// childEnviron returns the environment of a child shell started from env.
func childEnviron(env expand.Environ) expand.Environ {
	return newShellEnviron(exportedVariables(env))
}
