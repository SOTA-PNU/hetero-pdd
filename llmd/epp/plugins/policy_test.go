package plugins

import (
	"context"
	"math"
	"testing"
	"time"

	dl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	api "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	rc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	rh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	s "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	src "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/metrics"
	dto "github.com/prometheus/client_model/go"
)

func ep(id string, labels map[string]string, live liveState) s.Endpoint {
	l := map[string]string{labelPrefix + "id": id, labelPrefix + "kv-group": "cohort", labelPrefix + "model": "revision", labelPrefix + "slots": "1", labelPrefix + "block-size": "16"}
	for k, v := range labels {
		l[labelPrefix+k] = v
	}
	e := s.NewEndpoint(&dl.EndpointMetadata{Name: id, Address: "127.0.0.1", Port: "8000", Labels: l}, dl.NewMetrics(), nil)
	e.Put(liveKey, live)
	return e
}
func profiles(full, ps, ds []s.Endpoint) map[string]*s.ProfileRunResult {
	r := map[string]*s.ProfileRunResult{}
	for n, es := range map[string][]s.Endpoint{"full": full, "prefill": ps, "decode": ds} {
		var cs []s.ScoredEndpoint
		for _, e := range es {
			cs = append(cs, s.ScoredEndpoint{Endpoint: e})
		}
		r[n] = &s.ProfileRunResult{ScoredCandidates: cs}
	}
	return r
}
func TestPairUsesConsumerAllowlistAndCoherentGroup(t *testing.T) {
	p := ep("p", map[string]string{"producer": "true"}, liveState{})
	for _, tc := range []struct {
		label string
		value string
	}{{"peer-p", "false"}, {"kv-group", "other"}, {"model", "wrong"}, {"consumer", "false"}, {"id", "p"}} {
		l := map[string]string{"consumer": "true", "peer-p": "true"}
		l[tc.label] = tc.value
		d := ep("d", l, liveState{})
		if (pairConfig{}).compatible(p, d) {
			t.Fatal(tc)
		}
	}
	d := ep("d", map[string]string{"consumer": "true", "peer-p": "true"}, liveState{})
	if !(pairConfig{Producer: "p", Consumer: "d"}).compatible(p, d) || (pairConfig{Producer: "other"}).compatible(p, d) {
		t.Fatal("pinned pair")
	}
}
func TestSnapshotRejectsStaleMissingAndNaN(t *testing.T) {
	now := time.Now()
	filter := freshQueue{ttl: time.Second}
	live := liveState{At: now, Queue: 0}
	ok := ep("ok", nil, live)
	live.Queue = -1
	missing := ep("missing", nil, live)
	live.Queue = 0
	live.At = now.Add(-2 * time.Second)
	stale := ep("stale", nil, live)
	candidates := filter.Filter(context.Background(), nil, []s.Endpoint{missing, stale, ok})
	if len(candidates) != 1 || candidates[0] != ok {
		t.Fatal(candidates)
	}
	nan := math.NaN()
	if gauge(&dto.MetricFamily{Metric: []*dto.Metric{{Gauge: &dto.Gauge{Value: &nan}}}}, sum) != -1 {
		t.Fatal("nonfinite gauge accepted")
	}
}
func TestFeaturesCapTextAndExpiredTable(t *testing.T) {
	maxOut := int64(5)
	h := &costPolicy{costConfig: costConfig{TextBytes: 4, Prior: 32}}
	r := &s.InferenceRequest{Body: &rh.InferenceRequestBody{Completions: &rh.CompletionsRequest{Prompt: rh.Prompt{Strings: []string{"hello", "abcdefgh"}}}, MaxOutputTokens: &maxOut}, Headers: map[string]string{"x-pdd-ttft-ms": "12"}}
	got := h.features(r)
	if got.Input != 4 || got.Output != 5 || got.InputSource != "text-bytes" || got.TTFTGoalMs != 12 {
		t.Fatal(got)
	}
	r.Body.Completions.Prompt = rh.Prompt{TokenIDs: [][]uint32{{1, 2}, {3}}}
	if h.features(r).Input != 3 {
		t.Fatal("token IDs ignored")
	}
	past := time.Now().Add(-time.Second)
	h.ValidUntil = &past
	if _, err := h.ProcessResults(context.Background(), r, nil); err == nil {
		t.Fatal("expired table used")
	}
}
func TestTikTokReservesOnceReleasesOnEOS(t *testing.T) {
	l := liveState{At: time.Now(), KV: .3}
	p := ep("p", map[string]string{"producer": "true", "backend": "gpu"}, l)
	l.Running = 1
	l.HostSlots = 1
	l.HostBytes = 512
	d := ep("d", map[string]string{"consumer": "true", "backend": "npu", "peer-p": "true"}, l)
	h := &costPolicy{costConfig: costConfig{TTL: 5000, MaxKV: 1, TextBytes: 4, Prior: 10, Preload: true, Costs: map[string]costs{
		"gpu": {Prefill: 1, Decode: 1, Remaining: 100, Setup: 4}, "npu": {Prefill: 1, Decode: 1, Remaining: 100, Bytes: 32}}},
		reservations: map[string]reservation{}}
	route := func(id string) bool {
		req := &s.InferenceRequest{RequestID: id, Headers: map[string]string{}, Body: &rh.InferenceRequestBody{Completions: &rh.CompletionsRequest{Prompt: rh.Prompt{TokenIDs: [][]uint32{{1, 2, 3}}}}}}
		rr, err := h.ProcessResults(context.Background(), req, profiles(nil, []s.Endpoint{p}, []s.Endpoint{d}))
		if err != nil {
			t.Fatal(err)
		}
		if err := h.PreRequest(context.Background(), req, rr); err != nil {
			t.Fatal(err)
		}
		return req.Headers["x-hetero-preload"] == "1"
	}
	if !route("a") || route("b") {
		t.Fatal("overbooked or missed credit")
	}
	h.ResponseBody(context.Background(), &s.InferenceRequest{RequestID: "a"}, &rc.Response{}, nil)
	if route("c") {
		t.Fatal("stream chunks released credit early")
	}
	h.ResponseBody(context.Background(), &s.InferenceRequest{RequestID: "a"}, &rc.Response{EndOfStream: true}, nil)
	if !route("d") {
		t.Fatal("EOS did not release credit")
	}
}
func TestGreedyInterleavingCancellationAndResize(t *testing.T) {
	plugin, _ := newGreedy(named{api.TypedName{Name: "greedy"}}, nil)
	g := plugin.(*greedy)
	build := func(slots string) map[string]*s.ProfileRunResult {
		return profiles([]s.Endpoint{ep("a", map[string]string{"slots": slots}, liveState{}), ep("b", nil, liveState{})}, nil, nil)
	}
	acquire := func(id, slots string) string {
		ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
		defer cancel()
		res, err := g.ProcessResults(ctx, &s.InferenceRequest{RequestID: id}, build(slots))
		if err != nil {
			t.Fatal(err)
		}
		return endpointID(res.PrimaryEndpoint())
	}
	for i, tc := range []struct{ id, target string }{{"first", "a"}, {"second", "b"}, {"third", "a"}} {
		if got := acquire(tc.id, "2"); got != tc.target {
			t.Fatalf("%d: %s", i, got)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := g.ProcessResults(ctx, &s.InferenceRequest{RequestID: "cancelled"}, build("1")); err == nil {
		t.Fatal("cancelled waiter got a token")
	}
	if _, err := g.ProcessResults(ctx, &s.InferenceRequest{RequestID: "again"}, build("2")); err == nil {
		t.Fatal("resizing duplicated an outstanding token")
	}
	g.release("second")
	if acquire("next", "2") != "b" {
		t.Fatal("completion order lost")
	}
	g.release("third")
	g.release("third")
	if acquire("last", "2") != "a" {
		t.Fatal("double release")
	}
}

func TestTraceAllocationReplaysHistoricalIndices(t *testing.T) {
	p := ep("gpu0", map[string]string{"producer": "true", "full": "true"}, liveState{})
	d := ep("npu0", map[string]string{"consumer": "true", "peer-gpu0": "true", "full": "true"}, liveState{})
	results := profiles([]s.Endpoint{p, d}, []s.Endpoint{p}, []s.Endpoint{d})
	for _, tc := range []struct {
		indices  string
		expected []string
	}{
		{"", []string{"full", "decode", "decode"}},
		{",\"pddRequestIndices\":[1]", []string{"full", "decode", "full"}},
		{",\"pddRequestIndices\":[]", []string{"full", "full", "full"}},
	} {
		j := `{"allocation":{"0":"gpu0","1":"npu0","2":"npu0"},"pddTargets":["npu0"]` + tc.indices + "}"
		raw := api.StrictDecoder([]byte(j))
		obj, err := newTraceAllocation(named{}, raw)
		if err != nil {
			t.Fatal(err)
		}
		h := obj.(*traceAllocation)
		for i, id := range []string{"0", "1", "2"} {
			r := &s.InferenceRequest{RequestID: id, Headers: map[string]string{"x-pdd-request-index": id}}
			route, err := h.ProcessResults(context.Background(), r, results)
			if err != nil || route.PrimaryProfileName != tc.expected[i] {
				t.Fatal(i, route, err)
			}
			h.ResponseBody(context.Background(), r, &rc.Response{EndOfStream: true}, nil)
		}
		if _, err := h.ProcessResults(context.Background(), &s.InferenceRequest{}, results); err == nil {
			t.Fatal("missing trace index accepted")
		}
	}
}

func TestColdWorkerUsesMeasuredRBLNGauges(t *testing.T) {
	e := dl.NewEndpoint(nil, nil)
	family := func(v float64) *dto.MetricFamily {
		return &dto.MetricFamily{Metric: []*dto.Metric{{Gauge: &dto.Gauge{Value: &v}}}}
	}
	f := src.PrometheusMetricMap{rblnQueue: family(2), rblnRunning: family(1), rblnKV: family(.125), queueMetric: &dto.MetricFamily{}}
	ext := extractor{}
	in := dl.PollInput[src.PrometheusMetricMap]{Payload: f, Endpoint: e}
	read := func() liveState {
		t.Helper()
		if err := ext.Extract(context.Background(), in); err != nil {
			t.Fatal(err)
		}
		v, ok := e.GetAttributes().Get(liveKey)
		if !ok {
			t.Fatal("missing observation")
		}
		return v.(liveState)
	}
	l := read()
	if l.Queue != 2 || l.Running != 1 || l.KV != .125 || e.GetMetrics().WaitingQueueSize != 2 || !e.GetMetrics().Updated() {
		t.Fatal(l, e.GetMetrics())
	}
	f[queueMetric] = family(0)
	if l := read(); l.Queue != 2 {
		t.Fatalf("scheduler transition must take precedence over lagging stats: %v", l)
	}
	delete(f, rblnQueue)
	if l := read(); l.Queue != 0 {
		t.Fatal("missing native fallback", l)
	}
	delete(f, queueMetric)
	if l := read(); l.Queue != -1 {
		t.Fatal("missing queue retained value from earlier scrape")
	}
}

func TestProducerIdentitySource(t *testing.T) {
	p := ep("pool", map[string]string{"producer": "true", "engine-id": "static-engine", "kv-port": "5557", "tp": "1", "num-blocks": "256"}, liveState{})
	d := ep("d", map[string]string{"consumer": "true", "peer-pool": "true"}, liveState{})
	meta := p.GetMetadata()
	meta.Name = "deployed-engine-64bbcc99-d9fz7"
	meta.ID.Name = meta.Name + "-rank-0"
	res := pairResult(p, d)
	c := &contract{}
	for _, tc := range []struct{ source, pod, want string }{
		{"", meta.Name, "static-engine"},
		{"pod-name", meta.Name, meta.Name},
		{"pod-name", "replacement-pod", "replacement-pod"},
		{"pod-name", "", ""},
		{"unrecognized", meta.Name, ""},
	} {
		meta.Labels[labelPrefix+"engine-id-source"] = tc.source
		meta.Name = tc.pod
		r := &s.InferenceRequest{RequestID: "test", Headers: map[string]string{}}
		err := c.PreRequest(context.Background(), r, res)
		if tc.want == "" {
			if err == nil {
				t.Fatal("silently accepted unavailable/unsupported identity", tc)
			}
			continue
		}
		if err != nil || r.Headers["x-hetero-expected-engine"] != tc.want {
			t.Fatal(tc, r.Headers, err)
		}
	}
}
