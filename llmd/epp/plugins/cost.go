package plugins

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"strconv"
	"sync"
	"time"

	dl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	api "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	rc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	s "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

var estimateKey = api.NewDataKey("hetero.route-estimate", "cost-route-handler")

type estimate struct {
	Policy                                                               string
	Input, Output                                                        int
	InputSource                                                          string
	Priority                                                             int
	TTFTMs, FinishMs, DecodeWaitMs, TransferMs, TTFTGoalMs, PreloadBytes float64
	Preload                                                              bool
}
type costs struct {
	Prefill   float64 `json:"prefillMsPerToken"`
	Decode    float64 `json:"decodeMsPerToken"`
	Transfer  float64 `json:"transferMsPerToken"`
	Setup     float64 `json:"transferFixedMs"`
	Remaining int     `json:"remainingDecodeTokens"`
	Bytes     float64 `json:"kvBytesPerToken"`
}

// Transfer costs can depend on transport and NUMA placement for a specific edge.
type linkCost struct {
	Fixed    float64 `json:"transferFixedMs"`
	PerToken float64 `json:"transferMsPerToken"`
}
type costConfig struct {
	Links map[string]map[string]linkCost `json:"links"`

	Costs      map[string]costs `json:"costs"`
	TTL        int              `json:"ttlMillis"`
	Prior      int              `json:"outputPriorTokens"`
	TextBytes  float64          `json:"textBytesPerToken"`
	MaxKV      float64          `json:"maxKvUtilization"`
	Preload    bool             `json:"preload"`
	ValidUntil *time.Time       `json:"validUntil"`
}
type reservation struct {
	decoder string
	bytes   float64
}
type costPolicy struct {
	named
	runAll
	costConfig
	mu           sync.Mutex
	reservations map[string]reservation
}

func newCostPolicy(n named, d *json.Decoder) (api.Plugin, error) {
	c := costConfig{TTL: 5000, Prior: 32, TextBytes: 4, MaxKV: 1}
	if err := decode(d, &c); err != nil {
		return nil, err
	}
	if c.TTL <= 0 || c.Prior < 1 || c.TextBytes <= 0 || c.MaxKV <= 0 || c.MaxKV > 1 || len(c.Costs) == 0 {
		return nil, fmt.Errorf("invalid TTL, output prior, text ratio, KV ceiling, or empty costs")
	}
	for id, row := range c.Costs {
		if id == "" || row.Prefill <= 0 || row.Decode <= 0 || row.Transfer < 0 || row.Setup < 0 || row.Remaining < 1 || row.Bytes < 0 {
			return nil, fmt.Errorf("invalid cost row %q", id)
		}
	}
	for producer, links := range c.Links {
		for consumer, link := range links {
			if producer == "" || consumer == "" || link.Fixed < 0 || link.PerToken < 0 {
				return nil, fmt.Errorf("invalid link cost %q to %q", producer, consumer)
			}
		}
	}
	return &costPolicy{named: n, costConfig: c, reservations: map[string]reservation{}}, nil
}
func (*costPolicy) Consumes() api.DataDependencies {
	deps := liveDependency()
	deps.Required[estimateKey] = estimate{}
	return deps
}
func (*costPolicy) Produces() map[api.DataKey]any {
	return map[api.DataKey]any{estimateKey: estimate{}}
}
func (h *costPolicy) features(r *s.InferenceRequest) estimate {
	e := estimate{Policy: h.name.Name, Output: h.Prior, Priority: r.Objectives.Priority}
	b := r.Body
	if b != nil {
		if b.MaxOutputTokens != nil && *b.MaxOutputTokens < int64(e.Output) {
			e.Output = int(*b.MaxOutputTokens)
		}
		if b.TokenizedRequest != nil {
			for _, p := range b.TokenizedRequest.Prompts {
				e.Input += len(p.TokenIDs)
			}
		}
		if e.Input == 0 && b.Completions != nil {
			for _, ids := range b.Completions.Prompt.TokenIDs {
				e.Input += len(ids)
			}
		}
		if e.Input > 0 {
			e.InputSource = "parser-token-ids"
		} else {
			n := 0
			if b.Completions != nil {
				n = len(b.Completions.Prompt.Raw)
				for _, str := range b.Completions.Prompt.Strings {
					n += len(str)
				}
			}
			if b.ChatCompletions != nil {
				for _, m := range b.ChatCompletions.Messages {
					n += len(m.Content.PlainText())
				}
			}
			e.InputSource = "text-bytes"
			if n == 0 {
				n = r.RequestSizeBytes
				e.InputSource = "request-bytes"
			}
			e.Input = max(1, int(math.Ceil(float64(n)/h.TextBytes)))
		}
	}
	e.Input = max(1, e.Input)
	// Optional per-request TTFT target, in milliseconds. Among equally feasible
	// requests prefer minimum predicted completion; when infeasible minimize miss.
	if goal, err := strconv.ParseFloat(r.Headers["x-pdd-ttft-ms"], 64); err == nil && goal > 0 && !math.IsInf(goal, 0) {
		e.TTFTGoalMs = goal
	}
	return e
}

type pricedEndpoint struct {
	e     s.Endpoint
	cost  costs
	state liveState
	wait  float64
	score float64
}

func (h *costPolicy) priced(es []s.ScoredEndpoint, now time.Time) []pricedEndpoint {
	var out []pricedEndpoint
	for _, cand := range es {
		e := cand.Endpoint
		c, ok := h.Costs[endpointID(e)]
		if !ok {
			c, ok = h.Costs[label(e, "backend")]
		}
		if !ok {
			continue
		}
		l, ok := readLive(e, now, time.Duration(h.TTL)*time.Millisecond)
		if !ok || l.Queue < 0 || l.Running < 0 || l.KV < 0 || l.KV >= h.MaxKV {
			continue
		}
		slots, err := strconv.Atoi(label(e, "slots"))
		if err != nil || slots < 1 {
			continue
		}
		// Running work below the batch limit need not finish before admission.
		batches := math.Floor((l.Queue + l.Running) / float64(slots))
		out = append(out, pricedEndpoint{e: e, cost: c, state: l, wait: batches * float64(c.Remaining) * c.Decode, score: cand.Score})
	}
	return out
}
func (h *costPolicy) credit(e pricedEndpoint, bytes float64) bool {
	slots, free := e.state.HostSlots, e.state.HostBytes
	// Conservatively hold EPP grants until EOS. The connector independently
	// reserves actual resources atomically, including across EPP replicas.
	for _, r := range h.reservations {
		if r.decoder == endpointID(e.e) {
			slots--
			free -= r.bytes
		}
	}
	return bytes > 0 && slots >= 1 && free >= bytes
}
func (h *costPolicy) ProcessResults(_ context.Context, r *s.InferenceRequest, results map[string]*s.ProfileRunResult) (*s.SchedulingResult, error) {
	now := time.Now()
	if h.ValidUntil != nil && !now.Before(*h.ValidUntil) {
		return nil, fmt.Errorf("routing cost table expired at %s", h.ValidUntil)
	}
	base := h.features(r)
	full := h.priced(candidates(results, "full"), now)
	ps := h.priced(candidates(results, "prefill"), now)
	ds := h.priced(candidates(results, "decode"), now)
	var selection *s.SchedulingResult
	var chosen estimate
	var winningScore, miss float64
	consider := func(res *s.SchedulingResult, e estimate, score float64) {
		over := 0.
		if e.TTFTGoalMs > 0 {
			over = math.Max(0, e.TTFTMs-e.TTFTGoalMs)
		}
		if selection == nil || over < miss || (over == miss && (e.FinishMs < chosen.FinishMs || (e.FinishMs == chosen.FinishMs && score > winningScore))) {
			selection = res
			chosen = e
			winningScore = score
			miss = over
		}
	}
	for _, e := range full {
		if label(e.e, "full") != "true" {
			continue
		}
		pred := base
		pred.DecodeWaitMs = e.wait
		pred.TTFTMs = e.wait + float64(base.Input)*e.cost.Prefill + e.cost.Decode
		pred.FinishMs = pred.TTFTMs + float64(max(0, base.Output-1))*e.cost.Decode
		consider(fullResult(e.e), pred, e.score)
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, p := range ps {
		for _, d := range ds {
			if !(pairConfig{}).compatible(p.e, d.e) {
				continue
			}
			pred := base
			link, ok := h.Links[endpointID(p.e)][endpointID(d.e)]
			if !ok {
				link = linkCost{Fixed: p.cost.Setup, PerToken: p.cost.Transfer}
			}
			pred.TransferMs = link.Fixed + float64(base.Input)*link.PerToken
			pred.DecodeWaitMs = d.wait
			ready := p.wait + float64(base.Input)*p.cost.Prefill
			block, err := strconv.Atoi(label(d.e, "block-size"))
			if err == nil && block > 0 {
				pred.PreloadBytes = math.Ceil(float64(base.Input)/float64(block)) * float64(block) * d.cost.Bytes
			}
			// Ordinary PD overlaps producer computation with draining the decoder.
			// Only the authorized lookahead route can also hide the host transfer.
			tt := math.Max(ready, d.wait) + pred.TransferMs
			if h.Preload && d.state.Running > 0 && d.wait > ready && h.credit(d, pred.PreloadBytes) {
				pred.Preload = true
				tt = math.Max(ready+pred.TransferMs, d.wait)
			}
			pred.TTFTMs = tt + d.cost.Decode
			pred.FinishMs = tt + float64(max(1, base.Output))*d.cost.Decode
			consider(pairResult(p.e, d.e), pred, p.score+d.score)
		}
	}
	if selection == nil {
		return nil, fmt.Errorf("no eligible full endpoint or compatible pair with fresh queue, running, KV and matching cost rows")
	}
	if chosen.Preload {
		h.reservations[r.RequestID] = reservation{endpointID(selection.PrimaryEndpoint()), chosen.PreloadBytes}
	}
	r.PutAttribute(estimateKey, chosen)
	return selection, nil
}
func (h *costPolicy) ResponseBody(_ context.Context, r *s.InferenceRequest, resp *rc.Response, _ *dl.EndpointMetadata) {
	if resp.EndOfStream {
		h.mu.Lock()
		delete(h.reservations, r.RequestID)
		h.mu.Unlock()
	}
}

func (h *costPolicy) PreRequest(ctx context.Context, r *s.InferenceRequest, res *s.SchedulingResult) error {
	e, ok := s.ReadRequestAttribute[estimate](r, estimateKey)
	if !ok {
		return fmt.Errorf("cost handler produced no estimate")
	}
	reason := "minimum_estimated_completion"
	if e.TTFTGoalMs > 0 {
		reason = "ttft_goal_then_completion"
	}
	if e.Preload {
		reason += "_with_host_lookahead"
	}
	r.Headers["x-hetero-reason"] = reason
	if e.Preload && res.ProfileResults["prefill"].FirstEndpoint() != nil {
		r.Headers["x-hetero-preload"] = "1"
	}
	log.FromContext(ctx).Info("hetero_route_prediction", "request_id", r.Headers["x-request-id"], "epp_request_id", r.RequestID, "estimate", e)
	return nil
}
