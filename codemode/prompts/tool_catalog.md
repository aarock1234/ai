{{if .Declarations}}```ts
declare const tools: {
{{.Declarations}}
};
```

Tool call examples:
```ts
{{.Examples}}
```{{else}}No host tools. Do not call `tools.*`.{{end}}
