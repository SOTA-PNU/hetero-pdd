// Command serve runs the llm-d scheduling graph and its loopback HTTP ingress.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"log/slog"
	"net"
	"net/http"
	"os"
	"runtime/debug"
	"time"

	"github.com/SOTA-PNU/hetero-pdd/llmd/epp/scheduling"
	"github.com/SOTA-PNU/hetero-pdd/llmd/epp/transport"
	"github.com/go-logr/logr"
	ctrllog "sigs.k8s.io/controller-runtime/pkg/log"
)

type config struct {
	Listen string `json:"listen"`
	Model  string `json:"model"`
	scheduling.Config
	Producers map[string]transport.Producer `json:"producers"`
	Endpoints []scheduling.Endpoint         `json:"endpoints"`
}

func run(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	var cfg config
	d := json.NewDecoder(f)
	d.DisallowUnknownFields()
	if err := d.Decode(&cfg); err != nil {
		return err
	}
	host, _, err := net.SplitHostPort(cfg.Listen)
	if err != nil || !net.ParseIP(host).IsLoopback() || cfg.Model == "" {
		return fmt.Errorf("model and loopback listen address with port required")
	}
	for _, ep := range cfg.Endpoints {
		if ep.Role == "prefill" {
			producer := cfg.Producers[ep.ID]
			if producer.EngineID == "" || producer.Host != "127.0.0.1" || producer.Port < 1 || producer.Port > 65535 {
				return fmt.Errorf("missing or invalid NIXL identity for prefill endpoint %q", ep.ID)
			}
		}
	}
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	ctrllog.SetLogger(logr.FromSlogHandler(logger.Handler()))
	cfg.Config.Model = cfg.Model
	scheduler, err := scheduling.New(cfg.Config, cfg.Endpoints)
	if err != nil {
		return err
	}
	server := &http.Server{
		Addr: cfg.Listen, ReadHeaderTimeout: 10 * time.Second,
		Handler: transport.New(cfg.Model, cfg.Policy, scheduler, cfg.Producers, logger),
	}
	return server.ListenAndServe()
}

func version() error {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return fmt.Errorf("no Go build metadata")
	}
	versions := map[string]string{"sidecar_protocol": "NIXLv2-v0.5.1"}
	for _, dep := range info.Deps {
		if dep.Replace != nil {
			continue
		}
		switch dep.Path {
		case "github.com/llm-d/llm-d-inference-scheduler":
			versions["epp"] = dep.Version
		case "sigs.k8s.io/gateway-api-inference-extension":
			versions["gateway_inference_extension"] = dep.Version
		}
	}
	if versions["epp"] == "" || versions["gateway_inference_extension"] == "" {
		return fmt.Errorf("llm-d and GIE release build metadata required")
	}
	for _, setting := range info.Settings {
		if setting.Key == "vcs.revision" {
			versions["vcs_revision"] = setting.Value
		}
		if setting.Key == "vcs.modified" {
			versions["vcs_modified"] = setting.Value
		}
	}
	return json.NewEncoder(os.Stdout).Encode(versions)
}

func main() {
	path := flag.String("config", "", "generated router JSON")
	showVersion := flag.Bool("version", false, "print Go build dependency versions")
	flag.Parse()
	var err error
	if *showVersion {
		err = version()
	} else {
		err = run(*path)
	}
	if err != nil {
		log.Fatal(err)
	}
}
