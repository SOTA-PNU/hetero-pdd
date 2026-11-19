// Package plugins provides out-of-tree plugins for the llm-d v1 EPP runner.
package plugins

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"sync"

	api "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	sched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const labelPrefix = "pdd.sota.dev/"

type named struct{ name api.TypedName }

func (p named) TypedName() api.TypedName { return p.name }

// Register adds factories to the public registry before the runner loads YAML.
func Register() {
	register("route-contract", newContract)
	register("reserved-field-guard", func(n named, _ *json.Decoder) (api.Plugin, error) { return &requestGuard{n}, nil })
	register("round-robin-picker", func(n named, _ *json.Decoder) (api.Plugin, error) { return &roundRobin{named: n}, nil })
	register("first-available-handler", newGreedy)
	register("trace-allocation-handler", newTraceAllocation)
	register("pair-handler", func(n named, d *json.Decoder) (api.Plugin, error) {
		var c pairConfig
		if err := decode(d, &c); err != nil {
			return nil, err
		}
		return &fixedPair{named: n, pairConfig: c}, nil
	})
	register("cost-route-handler", newCostPolicy)
	register("live-state-extractor", newExtractor)
	register("fresh-queue-filter", newFreshQueue)
	register("endpoint-label-scorer", newLabelScorer)
}

type factory func(named, *json.Decoder) (api.Plugin, error)

func register(kind string, f factory) {
	api.Register(kind, api.StabilityAlpha, func(name string, d *json.Decoder, _ api.Handle) (api.Plugin, error) {
		return f(named{api.TypedName{Name: name, Type: kind}}, d)
	})
}
func decode(d *json.Decoder, v any) error {
	if d != nil {
		return d.Decode(v)
	}
	return nil
}

func label(e sched.Endpoint, key string) string { return e.GetMetadata().Labels[labelPrefix+key] }
func endpointID(e sched.Endpoint) string        { return label(e, "id") }
func order(a, b sched.Endpoint) int {
	i, _ := strconv.Atoi(label(a, "order"))
	j, _ := strconv.Atoi(label(b, "order"))
	if i < j {
		return -1
	}
	if i > j {
		return 1
	}
	return strings.Compare(endpointID(a), endpointID(b))
}

type roundRobin struct {
	named
	mu     sync.Mutex
	cursor uint64
}

func (p *roundRobin) Pick(_ context.Context, c []*sched.ScoredEndpoint) *sched.ProfileRunResult {
	if len(c) == 0 {
		return nil
	}
	slices.SortFunc(c, func(a, b *sched.ScoredEndpoint) int { return order(a.Endpoint, b.Endpoint) })
	p.mu.Lock()
	e := c[p.cursor%uint64(len(c))].Endpoint
	p.cursor++
	p.mu.Unlock()
	return target(e)
}
func target(e sched.Endpoint) *sched.ProfileRunResult {
	return &sched.ProfileRunResult{TargetEndpoints: []sched.Endpoint{e}}
}

type runAll struct{}

func (runAll) Pick(_ context.Context, _ *sched.InferenceRequest, profiles map[string]sched.SchedulerProfile, done map[string]*sched.ProfileRunResult) map[string]sched.SchedulerProfile {
	if len(done) > 0 {
		return nil
	}
	return profiles
}
func candidates(r map[string]*sched.ProfileRunResult, profile string) []sched.ScoredEndpoint {
	p := r[profile]
	if p == nil {
		return nil
	}
	c := slices.Clone(p.ScoredCandidates)
	slices.SortFunc(c, func(a, b sched.ScoredEndpoint) int { return order(a.Endpoint, b.Endpoint) })
	return c
}
func fullResult(e sched.Endpoint) *sched.SchedulingResult {
	return &sched.SchedulingResult{PrimaryProfileName: "full", ProfileResults: map[string]*sched.ProfileRunResult{"full": target(e)}}
}
func pairResult(p, d sched.Endpoint) *sched.SchedulingResult {
	return &sched.SchedulingResult{PrimaryProfileName: "decode", ProfileResults: map[string]*sched.ProfileRunResult{"prefill": target(p), "decode": target(d)}}
}

type pairConfig struct {
	Producer string `json:"producer"`
	Consumer string `json:"consumer"`
}

func (c pairConfig) compatible(p, d sched.Endpoint) bool {
	pid, did := endpointID(p), endpointID(d)
	if pid == "" || did == "" || pid == did || label(p, "producer") != "true" || label(d, "consumer") != "true" || label(d, "peer-"+pid) != "true" {
		return false
	}
	return label(p, "kv-group") != "" && label(p, "kv-group") == label(d, "kv-group") &&
		label(p, "model") != "" && label(p, "model") == label(d, "model") &&
		(c.Producer == "" || c.Producer == pid) && (c.Consumer == "" || c.Consumer == did)
}

type fixedPair struct {
	named
	runAll
	pairConfig
}

func (h *fixedPair) ProcessResults(_ context.Context, _ *sched.InferenceRequest, r map[string]*sched.ProfileRunResult) (*sched.SchedulingResult, error) {
	var best *sched.SchedulingResult
	var bestScore float64
	for _, p := range candidates(r, "prefill") {
		for _, d := range candidates(r, "decode") {
			if h.compatible(p.Endpoint, d.Endpoint) && (best == nil || p.Score+d.Score > bestScore) {
				best = pairResult(p.Endpoint, d.Endpoint)
				bestScore = p.Score + d.Score
			}
		}
	}
	if best == nil {
		return nil, fmt.Errorf("no compatible discovered producer/consumer pair (pinned producer %q consumer %q)", h.Producer, h.Consumer)
	}
	return best, nil
}
