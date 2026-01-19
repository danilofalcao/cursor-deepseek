package main

import (
	"log"
	"os"
	"os/exec"
)

func main() {
	// TODO: implement the entrypoint

	proxyCmd := exec.Command("go", "run")
	proxyCmd.Stdin = os.Stdin
	proxyCmd.Stdout = os.Stdout
	proxyCmd.Stderr = os.Stderr
	proxyCmd.Env = os.Environ()

	switch {
	case os.Getenv("DEEPSEEK_API_KEY") != "":
		proxyCmd.Args = append(proxyCmd.Args, "proxy.go")
	case os.Getenv("OPENROUTER_API_KEY") != "":
		proxyCmd.Args = append(proxyCmd.Args, "proxy-openrouter.go")
	case os.Getenv("OLLAMA_API_ENDPOINT") != "":
		proxyCmd.Args = append(proxyCmd.Args, "proxy-ollama.go")
	}
	if err := proxyCmd.Run(); err != nil {
		log.Fatalf("Failed to run proxy: %v", err)
	}
}
