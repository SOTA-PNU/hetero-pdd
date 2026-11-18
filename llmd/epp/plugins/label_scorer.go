// A standalone example of extending a native profile with a weighted scorer.
package plugins

import (
	"context"
	"encoding/json"
	"fmt"
	api "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	s "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

type labelScorer struct {
	named
	Key   string `json:"key"`
	Value string `json:"value"`
}

func newLabelScorer(n named, d *json.Decoder) (api.Plugin, error) {
	p := &labelScorer{named: n}
	if err := decode(d, p); err != nil {
		return nil, err
	}
	if p.Key == "" || p.Value == "" {
		return nil, fmt.Errorf("label scorer requires key and value")
	}
	return p, nil
}
func (*labelScorer) Category() s.ScorerCategory { return s.Affinity }
func (p *labelScorer) Score(_ context.Context, _ *s.InferenceRequest, es []s.Endpoint) map[s.Endpoint]float64 {
	scores := make(map[s.Endpoint]float64, len(es))
	for _, e := range es {
		if e.GetMetadata().Labels[p.Key] == p.Value {
			scores[e] = 1
		} else {
			scores[e] = 0
		}
	}
	return scores
}
