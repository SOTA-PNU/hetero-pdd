package scheduling

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

var engines = []Endpoint{
	{ID: "gpu", URL: "http://127.0.0.1:8100", Role: "full", Slots: 2},
	{ID: "npu", URL: "http://127.0.0.1:8200", Role: "full", Slots: 1},
}

func scheduler(t *testing.T, policy string, engines []Endpoint) *Scheduler {
	t.Helper()
	s, err := New(Config{Policy: policy}, engines)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func selectEngine(t *testing.T, s *Scheduler) Selection {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	sel, err := s.Schedule(ctx, t.Name())
	if err != nil {
		t.Fatal(err)
	}
	return sel
}

func TestProfiles(t *testing.T) {
	baseline := scheduler(t, "baseline", engines[:1])
	sel := selectEngine(t, baseline)
	if sel.Prefill != nil || sel.Target.ID != "gpu" {
		t.Fatal(sel)
	}
	rr := scheduler(t, "round-robin", engines)
	for _, want := range []string{"gpu", "npu", "gpu", "npu"} {
		if got := selectEngine(t, rr).Target.ID; got != want {
			t.Fatalf("RR selected %s, want %s", got, want)
		}
	}
	fixed := scheduler(t, "fixed-pdd", []Endpoint{
		{"gpu", "http://127.0.0.1:9101", "prefill", 1},
		{"r1", "http://127.0.0.1:9201", "decode", 1},
		{"r0", "http://127.0.0.1:9200", "decode", 1},
	})
	for _, want := range []string{"r1", "r0", "r1"} {
		sel := selectEngine(t, fixed)
		if sel.Prefill == nil || sel.Prefill.ID != "gpu" || sel.Target.ID != want {
			t.Fatal(sel)
		}
	}
	fixed.pods = fixed.pods[1:] // prefill profile now has no candidates
	if _, err := fixed.Schedule(t.Context(), "no-prefill"); err == nil {
		t.Fatal("accepted decode-only fallback")
	}
}

func TestLeastQueueRequiresCurrentMetrics(t *testing.T) {
	makeMetrics := func(queue *atomic.Int64) *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/metrics" {
				http.NotFound(w, r)
				return
			}
			n := queue.Load()
			if n >= 0 {
				fmt.Fprintf(w, "# TYPE vllm:num_requests_waiting gauge\nvllm:num_requests_waiting{model_name=\"test\"} %d\n", n)
			}
		}))
	}
	var a, b atomic.Int64
	first := makeMetrics(&a)
	defer first.Close()
	second := makeMetrics(&b)
	defer second.Close()
	s := scheduler(t, "least-queue", []Endpoint{{"first", first.URL, "full", 1}, {"second", second.URL, "full", 1}})
	a.Store(2)
	if got := selectEngine(t, s).Target.ID; got != "second" {
		t.Fatal(got)
	}
	b.Store(5)
	if got := selectEngine(t, s).Target.ID; got != "first" {
		t.Fatal(got)
	}
	a.Store(-1)
	if _, err := s.Schedule(t.Context(), "missing"); err == nil {
		t.Fatal("missing metric treated as zero")
	}
	a.Store(1)
	second.Close()
	if _, err := s.Schedule(t.Context(), "offline"); err == nil {
		t.Fatal("offline candidate treated as an empty queue")
	}
}

func TestGreedyFIFOAndCancellation(t *testing.T) {
	s := scheduler(t, "greedy", engines)
	first := selectEngine(t, s)
	second := selectEngine(t, s)
	third := selectEngine(t, s)
	if first.Target.ID != "gpu" || second.Target.ID != "npu" || third.Target.ID != "gpu" {
		t.Fatal(first.Target, second.Target, third.Target)
	}
	type outcome struct {
		sel Selection
		err error
	}
	enqueued := func(ctx context.Context, n int) <-chan outcome {
		result := make(chan outcome, 1)
		go func() {
			sel, err := s.Schedule(ctx, "queued")
			result <- outcome{sel, err}
		}()
		until := time.Now().Add(3 * time.Second)
		for {
			s.slots.mu.Lock()
			waiting := s.slots.waiters.Len()
			s.slots.mu.Unlock()
			if waiting == n {
				return result
			}
			if time.Now().After(until) {
				t.Fatalf("queue length = %d, want %d", waiting, n)
			}
			time.Sleep(time.Millisecond)
		}
	}
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	next := enqueued(ctx, 1)
	midctx, midcancel := context.WithCancel(ctx)
	middle := enqueued(midctx, 2)
	last := enqueued(ctx, 3)
	midcancel()
	if err := (<-middle).err; !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	second.Release()
	second.Release() // a repeated completion must not produce a second slot
	selected := <-next
	if selected.err != nil || selected.sel.Target.ID != "npu" {
		t.Fatal(selected)
	}
	first.Release()
	selectedLast := <-last
	if selectedLast.err != nil || selectedLast.sel.Target.ID != "gpu" {
		t.Fatal(selectedLast)
	}
	selected.sel.Release()
	selectedLast.sel.Release()
	third.Release()
	s.slots.mu.Lock()
	defer s.slots.mu.Unlock()
	if len(s.slots.free) != 3 || s.slots.waiters.Len() != 0 {
		t.Fatalf("free=%v, pending=%d", s.slots.free, s.slots.waiters.Len())
	}
}

func TestInvalidConfiguration(t *testing.T) {
	for _, tc := range []struct {
		policy  string
		engines []Endpoint
	}{
		{"baseline", nil}, {"lookahead", engines}, {"fixed-pdd", engines},
		{"greedy", []Endpoint{{"bad", "http://127.0.0.1:80", "full", 0}}},
		{"baseline", []Endpoint{engines[0], engines[0]}},
		{"fixed-pdd", []Endpoint{{"g", "http://127.0.0.1:80", "prefill", 1}}},
	} {
		if _, err := New(Config{Policy: tc.policy}, tc.engines); err == nil {
			t.Fatal(tc)
		}
	}
}
