// The historical trace driver stores an allocation by immutable request index.
// This plugin replays the same targets and eligible PDD indices in a native EPP
// profile, with endpoint discovery and slot ownership handled at request time.
package plugins

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"

	api "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	s "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

type traceConfig struct {
	Allocation        map[int]string `json:"allocation"`
	PDDTargets        []string       `json:"pddTargets"`
	PDDRequestIndices []int          `json:"pddRequestIndices"`
}
type traceAllocation struct {
	*greedy
	allocation map[int]string
	pddTargets map[string]bool
	indices    map[int]bool
}

func newTraceAllocation(n named, d *json.Decoder) (api.Plugin, error) {
	var c traceConfig
	if err := decode(d, &c); err != nil {
		return nil, err
	}
	if len(c.Allocation) == 0 {
		return nil, fmt.Errorf("trace allocation is empty")
	}
	for index, id := range c.Allocation {
		if index < 0 || id == "" {
			return nil, fmt.Errorf("invalid trace allocation index or target")
		}
	}
	g, _ := newGreedy(n, nil)
	p := &traceAllocation{greedy: g.(*greedy), allocation: c.Allocation, pddTargets: map[string]bool{}}
	for _, id := range c.PDDTargets {
		p.pddTargets[id] = true
	}
	if c.PDDRequestIndices != nil {
		p.indices = map[int]bool{}
		for _, index := range c.PDDRequestIndices {
			if _, ok := c.Allocation[index]; !ok {
				return nil, fmt.Errorf("PDD index %d lacks allocation", index)
			}
			p.indices[index] = true
		}
	}
	return p, nil
}
func (h *traceAllocation) ProcessResults(ctx context.Context, r *s.InferenceRequest, profiles map[string]*s.ProfileRunResult) (*s.SchedulingResult, error) {
	index, err := strconv.Atoi(r.Headers["x-pdd-request-index"])
	targetID, assigned := h.allocation[index]
	if err != nil || !assigned {
		return nil, fmt.Errorf("trace index %q is absent from allocation", r.Headers["x-pdd-request-index"])
	}
	split := h.pddTargets[targetID] && (h.indices == nil || h.indices[index])
	var route *s.SchedulingResult
	if split {
		route, err = (&fixedPair{pairConfig: pairConfig{Consumer: targetID}}).ProcessResults(ctx, r, profiles)
		if err != nil {
			return nil, err
		}
	} else {
		for _, e := range candidates(profiles, "full") {
			if endpointID(e.Endpoint) == targetID {
				route = fullResult(e.Endpoint)
				break
			}
		}
		if route == nil {
			return nil, fmt.Errorf("assigned endpoint %q lacks full capability or availability", targetID)
		}
	}
	capacity := &s.ProfileRunResult{ScoredCandidates: []s.ScoredEndpoint{{Endpoint: route.PrimaryEndpoint()}}}
	if _, err := h.greedy.ProcessResults(ctx, r, map[string]*s.ProfileRunResult{"full": capacity}); err != nil {
		return nil, err
	}
	return route, nil
}
