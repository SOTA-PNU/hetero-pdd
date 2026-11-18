// Package main registers experimental plugins before entering the official EPP lifecycle.
package main

import (
	"fmt"
	"os"

	"github.com/SOTA-PNU/hetero-pdd/llmd/epp/plugins"
	"github.com/llm-d/llm-d-router/cmd/epp/runner"
	ctrl "sigs.k8s.io/controller-runtime"
)

func main() {
	plugins.Register()
	if err := runner.NewRunner().Run(ctrl.SetupSignalHandler()); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
