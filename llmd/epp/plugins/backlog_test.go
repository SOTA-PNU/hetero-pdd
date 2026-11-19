package plugins

import (
	"context"
	"fmt"
	api "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	rc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	rh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	s "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"math"
	"sync"
	"testing"
	"time"
)

func TestFixedGenerationUsesRequestLength(t *testing.T) {
	h := costPolicy{costConfig: costConfig{Prior: 32, TextBytes: 4}}
	maxOutput := int64(128)
	req := &s.InferenceRequest{Body: &rh.InferenceRequestBody{MaxOutputTokens: &maxOutput, Payload: rh.PayloadMap{"ignore_eos": true}}}
	e := h.features(req)
	if e.Output != 128 || e.OutputSource != "fixed-generation" {
		t.Fatal(e)
	}
	req.Body.Payload = rh.PayloadMap{"ignore_eos": true, "stop": "end"}
	if h.features(req).Output != 32 {
		t.Fatal("stop condition ignored")
	}
	req.Body.Payload = rh.PayloadMap{"ignore_eos": false}
	if h.features(req).Output != 32 {
		t.Fatal("ordinary cap used as actual length")
	}
	c := costs{Decode: 1.5, DecodeContext: .01}
	if math.Abs(c.decodeTime(64, 129)-2.78) > 1e-12 {
		t.Fatal(c.decodeTime(64, 129))
	}
}

func TestWorkAndScrapeReconciliation(t *testing.T) {
	now := time.Now()
	h := costPolicy{costConfig: costConfig{TTL: 5000, Prior: 32}, reservations: map[string]reservation{}}
	row := costs{Decode: 1, Remaining: 32}
	old := now.Add(-50 * time.Millisecond)
	h.reservations["pdd"] = reservation{assigned: old, producer: newStage("gpu", old, 0, 0, 12), primary: newStage("npu", old, 20, 25, 120), decodeMs: 1, output: 100}
	state := liveState{At: at(old, 30), Running: 1}
	wait, _ := h.backlog("npu", 1, row, state, now)
	if wait != 70 {
		t.Fatalf("one running NPU request contributed %.3f ms; want 70", wait)
	}
	state.Queue = 1 // second, externally scheduled request
	wait, _ = h.backlog("npu", 1, row, state, now)
	if wait != 102 {
		t.Fatal(wait)
	}
	wait, _ = h.backlog("gpu", 1, row, liveState{At: at(old, 5), Running: 1}, now)
	if wait != 0 {
		t.Fatal("old GPU prefill counted during NPU decode", wait)
	}
	req := &s.InferenceRequest{RequestID: "pdd"}
	h.ResponseBody(context.Background(), req, &rc.Response{StreamedEvents: 92}, nil)
	work := h.reservations["pdd"]
	if work.primary.end.Sub(work.primary.start) != 8*time.Millisecond {
		t.Fatal(work)
	}
	h.ResponseBody(context.Background(), req, &rc.Response{EndOfStream: true, TerminationCause: rc.TerminationCauseClientDisconnect}, nil)
	h.ResponseBody(context.Background(), req, &rc.Response{EndOfStream: true}, nil)
	wait, _ = h.backlog("npu", 1, row, liveState{At: at(old, 30), Running: 1}, time.Now())
	if wait != 0 || len(h.reservations) != 0 {
		t.Fatal("old running observation survived terminal hook", wait)
	}
}

func TestBlockTransferAndSafeImport(t *testing.T) {
	h := costPolicy{costConfig: costConfig{TTL: 5000, Prior: 10, MaxKV: 1, TextBytes: 4, Costs: map[string]costs{
		"p": {PrefillFixed: 2, Prefill: 1, Decode: 1}, "d": {Decode: 2, Remaining: 100, Bytes: 1},
	}, Links: map[string]map[string]linkCost{"p": {"d": {Fixed: 3, PerBlock: 8, ImportFixed: 7, ImportPerBlock: 12, Dispatch: 1}}}}, reservations: map[string]reservation{}}
	rows := profiles(nil, []s.Endpoint{ep("p", map[string]string{"producer": "true"}, liveState{At: time.Now()})},
		[]s.Endpoint{ep("d", map[string]string{"consumer": "true", "peer-p": "true"}, liveState{At: time.Now(), Running: 1, HostSlots: 1, HostBytes: 32})})
	// Input=17, 2 blocks, 19 ms producer, 19 ms receive, 31 ms import.
	req := &s.InferenceRequest{RequestID: "first", Body: &rh.InferenceRequestBody{Completions: &rh.CompletionsRequest{Prompt: rh.Prompt{TokenIDs: [][]uint32{make([]uint32, 17)}}}}}
	for _, on := range []bool{false, true} {
		h.Preload = on
		if _, err := h.ProcessResults(context.Background(), req, rows); err != nil {
			t.Fatal(err)
		}
		e, _ := s.ReadRequestAttribute[estimate](req, estimateKey)
		ttft := 252.
		if on {
			ttft = 233
		}
		if e.Preload != on || e.PrefillMs != 19 || e.TransferMs != 19 || e.ImportMs != 31 || e.PreloadBytes != 32 || e.TTFTMs != ttft {
			t.Fatal(e)
		}
		h.ResponseBody(context.Background(), req, &rc.Response{EndOfStream: true}, nil)
	}
}

func TestConcurrentRoutesReservePrimaryBeforeNextScrape(t *testing.T) {
	cores := []s.Endpoint{
		ep("fast", map[string]string{"full": "true"}, liveState{At: time.Now()}),
		ep("slow", map[string]string{"full": "true"}, liveState{At: time.Now()}),
	}
	rows := profiles(cores, nil, nil)
	h := costPolicy{costConfig: costConfig{TTL: 5000, Prior: 32, MaxKV: 1, TextBytes: 4, Objective: "workConserving", Costs: map[string]costs{"fast": {Prefill: 1, Decode: 1}, "slow": {Prefill: 2, Decode: 10}}}, reservations: map[string]reservation{}}
	run := func(id string) string {
		res, err := h.ProcessResults(context.Background(), &s.InferenceRequest{RequestID: id}, rows)
		if err != nil {
			t.Error(err)
			return ""
		}
		return endpointID(res.PrimaryEndpoint())
	}
	if run("initial") != "fast" {
		t.Fatal("isolated request picked the slow device")
	}
	held := make(chan string, 4)
	var group sync.WaitGroup
	for i := range 4 {
		group.Add(1)
		go func() { defer group.Done(); held <- run(fmt.Sprint(i)) }()
	}
	group.Wait()
	close(held)
	slow := 0
	for id := range held {
		if id == "slow" {
			slow++
		}
	}
	if slow != 1 {
		t.Fatalf("empty slow slot received %d requests", slow)
	}
	for id := range h.reservations {
		h.ResponseBody(context.Background(), &s.InferenceRequest{RequestID: id}, &rc.Response{EndOfStream: true, TerminationCause: rc.TerminationCauseError}, nil)
	}
	if len(h.reservations) != 0 {
		t.Fatal(h.reservations)
	}
}
func TestStrictCostConfig(t *testing.T) {
	for _, raw := range []string{
		`{"objective":"forcePD","costs":{"x":{"prefillFixedMs":1,"decodeMsPerToken":2}}}`,
		`{"costs":{"x":{"fullFixedMs":-1,"prefillFixedMs":1,"decodeMsPerToken":2}}}`,
		`{"costs":{"x":{"prefillFixedMs":1,"decodeMsPerToken":2}},"links":{"p":{"d":{"importMsPerBlock":-1}}}}`,
		`{"costs":{"x":{"prefillFixedMs":1,"decodeMsPerToken":2,"improtMs":1}}}`,
	} {
		if _, err := newCostPolicy(named{}, api.StrictDecoder([]byte(raw))); err == nil {
			t.Fatal("accepted", raw)
		}
	}
}
