package plugins

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"time"

	dl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	api "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	s "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	src "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/metrics"
	dto "github.com/prometheus/client_model/go"
)

var liveKey = api.NewDataKey("hetero.live", "live-state-extractor")

type liveState struct {
	At                                       time.Time
	Queue, Running, HostSlots, HostBytes, KV float64 // -1 marks a missing or invalid family
}

func (l liveState) Clone() dl.Cloneable { return l }

type extractor struct{ named }

func newExtractor(n named, _ *json.Decoder) (api.Plugin, error) { return &extractor{n}, nil }
func (*extractor) Produces() map[api.DataKey]any                { return map[api.DataKey]any{liveKey: liveState{}} }
func liveDependency() api.DataDependencies {
	return api.DataDependencies{Required: map[api.DataKey]any{liveKey: liveState{}}}
}

const (
	rblnQueue     = "hetero_pdd_waiting_requests"
	rblnRunning   = "hetero_pdd_running_requests"
	rblnKV        = "hetero_pdd_kv_usage"
	queueMetric   = "vllm:num_requests_waiting"
	runningMetric = "vllm:num_requests_running"
	kvMetric      = "vllm:kv_cache_usage_perc"
	slotsMetric   = "hetero_pdd_prepare_slots_free"
	bytesMetric   = "hetero_pdd_prepare_bytes_free"
)

func (*extractor) MetricFamilies() []string {
	return []string{queueMetric, runningMetric, kvMetric, slotsMetric, bytesMetric, rblnQueue, rblnRunning, rblnKV}
}

// Counts sum over worker series. A shared host budget is bounded by the
// smallest reported rank and KV occupancy by the busiest rank.
func gauge(f *dto.MetricFamily, reduce func(float64, float64) float64) float64 {
	if f == nil || len(f.Metric) == 0 {
		return -1
	}
	var result float64
	for i, m := range f.Metric {
		var v float64
		switch {
		case m.Gauge != nil:
			v = m.Gauge.GetValue()
		case m.Untyped != nil:
			v = m.Untyped.GetValue()
		default:
			return -1
		}
		if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 {
			return -1
		}
		if i == 0 {
			result = v
		} else {
			result = reduce(result, v)
		}
	}
	return result
}
func sum(a, b float64) float64 { return a + b }
func (*extractor) Extract(_ context.Context, in dl.PollInput[src.PrometheusMetricMap]) error {
	f := in.Payload
	state := liveState{At: time.Now(), Queue: gauge(f[queueMetric], sum), Running: gauge(f[runningMetric], sum), KV: gauge(f[kvMetric], math.Max), HostSlots: gauge(f[slotsMetric], math.Min), HostBytes: gauge(f[bytesMetric], math.Min)}
	// RBLN's scheduler publishes these gauges on each state transition.
	// Native multiprocess vLLM statistics may retain an older interval.
	if queue := gauge(f[rblnQueue], sum); queue >= 0 {
		state.Queue = queue
		m := in.Endpoint.GetMetrics().Clone()
		if m == nil {
			m = dl.NewMetrics()
		}
		m.WaitingQueueSize = int(queue)
		m.UpdateTime = state.At
		in.Endpoint.UpdateMetrics(m)
	}
	if running := gauge(f[rblnRunning], sum); running >= 0 {
		state.Running = running
	}
	if kv := gauge(f[rblnKV], math.Max); kv >= 0 {
		state.KV = kv
	}
	in.Endpoint.GetAttributes().Put(liveKey, state)
	return nil
}
func readLive(e s.Endpoint, now time.Time, ttl time.Duration) (liveState, bool) {
	v, ok := e.Get(liveKey)
	if !ok {
		return liveState{}, false
	}
	l, ok := v.(liveState)
	return l, ok && !l.At.IsZero() && !now.Before(l.At) && now.Sub(l.At) <= ttl
}

type freshQueue struct {
	named
	ttl time.Duration
}

func newFreshQueue(n named, d *json.Decoder) (api.Plugin, error) {
	c := struct {
		TTLMillis int `json:"ttlMillis"`
	}{5000}
	if err := decode(d, &c); err != nil {
		return nil, err
	}
	if c.TTLMillis <= 0 {
		return nil, fmt.Errorf("ttlMillis must be positive")
	}
	return &freshQueue{n, time.Duration(c.TTLMillis) * time.Millisecond}, nil
}
func (*freshQueue) Consumes() api.DataDependencies { return liveDependency() }
func (p *freshQueue) Filter(_ context.Context, _ *s.InferenceRequest, eps []s.Endpoint) []s.Endpoint {
	now := time.Now()
	out := make([]s.Endpoint, 0, len(eps))
	for _, e := range eps {
		if l, ok := readLive(e, now, p.ttl); ok && l.Queue >= 0 {
			out = append(out, e)
		}
	}
	return out
}
