package plugins

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"sync"
	"time"

	dl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	api "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	rc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	s "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

var estimateKey = api.NewDataKey("hetero.route-estimate", "cost-route-handler")

type estimate struct {
	Policy                                                                         string
	Input, Output                                                                  int
	InputSource, OutputSource                                                      string
	PrefillMs, PrefillWaitMs, ImportMs, DispatchMs, HiddenReceiveMs, PreloadGateMs float64
	Priority                                                                       int
	TTFTMs, FinishMs, DecodeWaitMs, TransferMs, TTFTGoalMs, PreloadBytes           float64
	Preload                                                                        bool
}
type costs struct {
	PrefillFixed  float64  `json:"prefillFixedMs"`
	FullFixed     *float64 `json:"fullFixedMs"`
	Prefill       float64  `json:"prefillMsPerToken"`
	Decode        float64  `json:"decodeMsPerToken"`
	DecodeContext float64  `json:"decodeMsPerContextToken"`
	Transfer      float64  `json:"transferMsPerToken"`
	Setup         float64  `json:"transferFixedMs"`
	Remaining     int      `json:"remainingDecodeTokens"`
	Bytes         float64  `json:"kvBytesPerToken"`
}

// Mean decode step for the request's context range.
func (c costs) decodeTime(input, output int) float64 {
	return c.Decode + c.DecodeContext*(float64(input)+float64(max(0, output-1))/2)
}

// Transfer costs can depend on transport and NUMA placement for a specific edge.
type linkCost struct {
	PerBlock       float64 `json:"transferMsPerBlock"`
	ImportFixed    float64 `json:"importFixedMs"`
	ImportPerBlock float64 `json:"importMsPerBlock"`
	Dispatch       float64 `json:"dispatchFixedMs"`
	Fixed          float64 `json:"transferFixedMs"`
	PerToken       float64 `json:"transferMsPerToken"`
}
type costConfig struct {
	Links map[string]map[string]linkCost `json:"links"`

	Objective        string           `json:"objective"`
	Costs            map[string]costs `json:"costs"`
	TTL              int              `json:"ttlMillis"`
	Prior            int              `json:"outputPriorTokens"`
	TextBytes        float64          `json:"textBytesPerToken"`
	MaxKV            float64          `json:"maxKvUtilization"`
	MaxPreloadWaitMs float64          `json:"maxPreloadWaitMs"`
	Preload          bool             `json:"preload"`
	ValidUntil       *time.Time       `json:"validUntil"`
}
type costPolicy struct {
	named
	runAll
	costConfig
	mu           sync.Mutex
	reservations map[string]reservation
	retired      []stage
}

func newCostPolicy(n named, d *json.Decoder) (api.Plugin, error) {
	c := costConfig{TTL: 5000, Prior: 32, Objective: "minimumCompletion", TextBytes: 4, MaxKV: 1}
	if err := decode(d, &c); err != nil {
		return nil, err
	}
	if c.MaxPreloadWaitMs < 0 || c.TTL <= 0 || c.Prior < 1 || c.TextBytes <= 0 || c.MaxKV <= 0 || c.MaxKV > 1 || len(c.Costs) == 0 {
		return nil, fmt.Errorf("invalid TTL, output prior, text ratio, KV ceiling, or empty costs")
	}
	if c.Objective != "minimumCompletion" && c.Objective != "workConserving" {
		return nil, fmt.Errorf("invalid objective %q", c.Objective)
	}
	for id, row := range c.Costs {
		if id == "" || (row.Prefill == 0 && row.PrefillFixed == 0) || row.Prefill < 0 || row.PrefillFixed < 0 || (row.FullFixed != nil && *row.FullFixed < 0) || row.Decode <= 0 || row.DecodeContext < 0 || row.Transfer < 0 || row.Setup < 0 || row.Remaining < 0 || row.Bytes < 0 {
			return nil, fmt.Errorf("invalid cost row %q", id)
		}
	}
	for producer, links := range c.Links {
		for consumer, link := range links {
			if producer == "" || consumer == "" || link.Fixed < 0 || link.PerToken < 0 || link.PerBlock < 0 || link.ImportFixed < 0 || link.ImportPerBlock < 0 || link.Dispatch < 0 {
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
	e := estimate{Policy: h.name.Name, Output: h.Prior, OutputSource: "prior", Priority: r.Objectives.Priority}
	b := r.Body
	if b != nil {
		if cap := b.MaxOutputTokens; cap != nil {
			// The common load generator disables EOS and supplies no stop sequences.
			// An ordinary client's upper bound alone is not an expected length.
			fixed := false
			if b.Payload != nil {
				if payload, ok := b.Payload.AsMap(); ok {
					ignoresEOS, _ := payload["ignore_eos"].(bool)
					fixed = ignoresEOS && payload["stop"] == nil && payload["stop_token_ids"] == nil
				}
			}
			if fixed || *cap < int64(e.Output) {
				e.Output = int(*cap)
				e.OutputSource = "request-cap"
				if fixed {
					e.OutputSource = "fixed-generation"
				}
			}
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
	e        s.Endpoint
	cost     costs
	state    liveState
	wait     float64
	decoding []stage
	score    float64
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
		if err != nil || slots < 1 || slots > 65536 {
			continue
		}
		ep := pricedEndpoint{e: e, cost: c, state: l, score: cand.Score}
		ep.wait, ep.decoding = h.backlog(endpointID(e), slots, c, l, now)
		out = append(out, ep)
	}
	return out
}

// The work-conserving objective can fill an idle second device even when a
// queued request on the faster device would finish earlier. A single request
// on idle engines is still priced by its full measured completion time.
func (h *costPolicy) prefer(a, b estimate, scoreA, scoreB float64) bool {
	if a.TTFTGoalMs > 0 {
		missA := math.Max(0, a.TTFTMs-a.TTFTGoalMs)
		missB := math.Max(0, b.TTFTMs-b.TTFTGoalMs)
		if missA != missB {
			return missA < missB
		}
	}
	if h.Objective == "workConserving" && len(h.reservations) > 0 && (a.DecodeWaitMs == 0) != (b.DecodeWaitMs == 0) {
		return a.DecodeWaitMs == 0
	}
	if a.FinishMs != b.FinishMs {
		return a.FinishMs < b.FinishMs
	}
	return scoreA > scoreB
}

func (h *costPolicy) ProcessResults(ctx context.Context, r *s.InferenceRequest, results map[string]*s.ProfileRunResult) (*s.SchedulingResult, error) {
	base := h.features(r)
	h.mu.Lock()
	defer h.mu.Unlock()
	var gated time.Duration
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		now := time.Now()
		if h.ValidUntil != nil && !now.Before(*h.ValidUntil) {
			return nil, fmt.Errorf("routing cost table expired at %s", h.ValidUntil)
		}
		if _, exists := h.reservations[r.RequestID]; exists {
			return nil, fmt.Errorf("request %q already scheduled", r.RequestID)
		}
		h.pruneRetired(now)
		selection, chosen, lease := h.choose(ctx, r.Headers["x-request-id"], base, results, now, millis(gated))
		if selection == nil {
			return nil, fmt.Errorf("no eligible full endpoint or compatible pair with fresh queue, running, KV and matching cost rows")
		}
		if gated == 0 {
			if signal, limit := h.preloadGate(selection, chosen, now); signal != nil && limit > 0 {
				timer := time.NewTimer(limit)
				h.mu.Unlock()
				select {
				case <-signal:
				case <-timer.C:
				case <-ctx.Done():
				}
				timer.Stop()
				h.mu.Lock()
				gated = time.Since(now)
				continue
			}
		}
		lease.assigned = now
		if r.Body != nil && r.Body.Stream {
			lease.awaitingFirst = make(chan struct{})
		}
		h.reservations[r.RequestID] = lease
		r.PutAttribute(estimateKey, chosen)
		return selection, nil
	}
}

func (h *costPolicy) choose(ctx context.Context, requestID string, base estimate, results map[string]*s.ProfileRunResult, now time.Time, gateMs float64) (*s.SchedulingResult, estimate, reservation) {
	var selection *s.SchedulingResult
	var chosen estimate
	var lease reservation
	var winningScore float64
	consider := func(res *s.SchedulingResult, e estimate, work reservation, score float64, target pricedEndpoint) {
		e.PreloadGateMs = gateMs
		e.TTFTMs += gateMs
		e.FinishMs += gateMs
		log.FromContext(ctx).Info("hetero_route_candidate", "request_id", requestID, "target", work.primary.id, "producer", work.producer.id, "state", target.state, "estimate", e)
		if selection == nil || h.prefer(e, chosen, score, winningScore) {
			selection, chosen, lease, winningScore = res, e, work, score
		}
	}
	for _, d := range h.priced(candidates(results, "full"), now) {
		if label(d.e, "full") != "true" {
			continue
		}
		e := base
		e.DecodeWaitMs = d.wait
		decode := d.cost.decodeTime(e.Input, e.Output)
		e.PrefillMs = d.cost.PrefillFixed + float64(e.Input)*d.cost.Prefill
		e.DispatchMs = decode // legacy tables priced the first token separately
		if d.cost.FullFixed != nil {
			e.DispatchMs = *d.cost.FullFixed
		}
		e.TTFTMs = d.wait + e.PrefillMs + e.DispatchMs
		e.FinishMs = e.TTFTMs + float64(max(0, e.Output-1))*decode
		work := reservation{primary: newStage(endpointID(d.e), now, 0, d.wait, e.FinishMs), first: at(now, e.TTFTMs), decodeMs: decode, output: e.Output}
		consider(fullResult(d.e), e, work, d.score, d)
	}
	producers := h.priced(candidates(results, "prefill"), now)
	decoders := h.priced(candidates(results, "decode"), now)
	for _, p := range producers {
		for _, d := range decoders {
			if !(pairConfig{}).compatible(p.e, d.e) {
				continue
			}
			link, ok := h.Links[endpointID(p.e)][endpointID(d.e)]
			if !ok {
				link = linkCost{Fixed: p.cost.Setup, PerToken: p.cost.Transfer}
			}
			block, err := strconv.Atoi(label(d.e, "block-size"))
			if err != nil || block <= 0 {
				continue
			}
			blocks := math.Ceil(float64(base.Input) / float64(block))
			e := base
			e.PrefillWaitMs, e.DecodeWaitMs = p.wait, d.wait
			decode := d.cost.decodeTime(e.Input, e.Output)
			e.PrefillMs = p.cost.PrefillFixed + float64(e.Input)*p.cost.Prefill
			e.DispatchMs = link.Dispatch
			e.TransferMs = link.Fixed + float64(e.Input)*link.PerToken + blocks*link.PerBlock
			e.ImportMs = link.ImportFixed + blocks*link.ImportPerBlock
			e.PreloadBytes = blocks * float64(block) * d.cost.Bytes
			prefillEnd := p.wait + e.PrefillMs
			ready := prefillEnd + e.DispatchMs
			start := math.Max(ready, d.wait)
			receiveEnd := start + e.TransferMs
			if h.Preload && d.wait > ready && e.TransferMs > 0 && h.credit(d, e.PreloadBytes) {
				// Queue depth alone does not prove a decode window. Use an
				// earlier predicted/observed decode interval covering arrival.
				for _, previous := range d.decoding {
					arrival := at(now, ready)
					if !previous.start.After(arrival) && previous.end.After(arrival) {
						e.Preload = true
						receiveEnd = math.Max(ready+e.TransferMs, d.wait)
						e.HiddenReceiveMs = start + e.TransferMs - receiveEnd
						break
					}
				}
			}
			e.TTFTMs = receiveEnd + e.ImportMs + decode
			e.FinishMs = e.TTFTMs + float64(max(0, e.Output-1))*decode
			if e.Preload {
				start = receiveEnd
			}
			work := reservation{
				producer: newStage(endpointID(p.e), now, 0, p.wait, prefillEnd),
				primary:  newStage(endpointID(d.e), now, ready, start, e.FinishMs),
				first:    at(now, e.TTFTMs), decodeMs: decode, output: e.Output,
			}
			if e.Preload {
				work.bytes = e.PreloadBytes
			}
			consider(pairResult(p.e, d.e), e, work, p.score+d.score, d)
		}
	}
	return selection, chosen, lease
}

func (h *costPolicy) ResponseHeader(_ context.Context, r *s.InferenceRequest, _ *rc.Response, _ *dl.EndpointMetadata) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if work, ok := h.reservations[r.RequestID]; ok && work.awaitingFirst != nil {
		work.headers = time.Now()
		h.reservations[r.RequestID] = work
	}
}

func (h *costPolicy) ResponseBody(ctx context.Context, r *s.InferenceRequest, resp *rc.Response, _ *dl.EndpointMetadata) {
	h.mu.Lock()
	defer h.mu.Unlock()
	work, ok := h.reservations[r.RequestID]
	if !ok {
		return
	} // includes repeated EOS
	now := time.Now()
	if work.awaitingFirst != nil && (resp.StreamedEvents > 0 || resp.EndOfStream) {
		close(work.awaitingFirst)
		work.awaitingFirst = nil
		if resp.StreamedEvents > 0 && !resp.EndOfStream {
			work.first = now
			log.FromContext(ctx).Info("hetero_first_stream_event", "request_id", r.Headers["x-request-id"], "first_data_ms", millis(now.Sub(work.assigned)), "received_headers", !work.headers.IsZero())
		}
	}
	if resp.EndOfStream {
		// llm-d sends this hook on upstream error, cancellation and eviction.
		delete(h.reservations, r.RequestID)
		h.retire(work, now)
	} else if resp.StreamedEvents > 0 {
		// A data event proves that the producer, host receive and import
		// have finished. StreamedEvents is an estimate for multi-token chunks.
		work.bytes = 0
		if work.producer.end.After(now) {
			work.producer.end = now
		}
		if work.first.After(now) {
			work.first = now
		}
		work.primary.start = now
		work.primary.end = at(now, float64(max(1, work.output-resp.StreamedEvents))*work.decodeMs)
		h.reservations[r.RequestID] = work
	}
}

func (h *costPolicy) PreRequest(ctx context.Context, r *s.InferenceRequest, res *s.SchedulingResult) error {
	e, ok := s.ReadRequestAttribute[estimate](r, estimateKey)
	if !ok {
		return fmt.Errorf("cost handler produced no estimate")
	}
	reason := "minimum_estimated_completion"
	if h.Objective == "workConserving" {
		reason = "work_conserving_then_completion"
	}
	if e.TTFTGoalMs > 0 {
		reason = "ttft_goal_then_" + reason
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
