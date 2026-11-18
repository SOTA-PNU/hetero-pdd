package transport

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/SOTA-PNU/hetero-pdd/llmd/epp/scheduling"
)

func engine(t *testing.T, role string, handle http.HandlerFunc) scheduling.Endpoint {
	t.Helper()
	s := httptest.NewServer(handle)
	t.Cleanup(s.Close)
	return scheduling.Endpoint{ID: role + "-1", Role: role, URL: s.URL, Slots: 1}
}

func ingress(t *testing.T, policy string, endpoints ...scheduling.Endpoint) *httptest.Server {
	t.Helper()
	scheduler, err := scheduling.New(scheduling.Config{Policy: policy}, endpoints)
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(New("model", policy, scheduler, map[string]Producer{"prefill-1": {EngineID: "producer", Host: "127.0.0.1", Port: 5555}}, slog.New(slog.NewTextHandler(io.Discard, nil))))
	t.Cleanup(server.Close)
	return server
}

func request(ctx context.Context, url, id, payload string) (*http.Response, error) {
	r, err := http.NewRequestWithContext(ctx, "POST", url+"/v1/completions", strings.NewReader(payload))
	if err != nil {
		return nil, err
	}
	r.Header.Set("X-Request-Id", id)
	return http.DefaultClient.Do(r)
}

const body = `{"model":"model","prompt":[22,23],"max_tokens":8,"min_tokens":6,"stream":true,"stream_options":{"include_usage":true}}`
const validKV = `{"do_remote_prefill":true,"do_remote_decode":false,"remote_engine_id":"producer","remote_request_id":"cmpl-job-0","remote_host":"127.0.0.1","remote_port":5555,"tp_size":1,"remote_block_ids":[[1,2]],"future_field":"preserved"}`

func TestNIXLWireAndStream(t *testing.T) {
	parse := func(r *http.Request) map[string]any {
		var payload map[string]any
		if r.Header.Get("X-Request-Id") != "job" {
			t.Errorf("request id %s", r.Header.Get("X-Request-Id"))
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Error(err)
		}
		return payload
	}
	p := engine(t, "prefill", func(w http.ResponseWriter, r *http.Request) {
		payload := parse(r)
		if payload["stream"] != false || payload["max_tokens"] != float64(1) || payload["stream_options"] != nil || payload["min_tokens"] != float64(1) || payload["kv_transfer_params"].(map[string]any)["do_remote_decode"] != true {
			t.Errorf("prefill payload %v", payload)
		}
		fmt.Fprint(w, `{"kv_transfer_params":`+validKV+`}`)
	})
	const frame = "data: {\"choices\":[{\"text\":\"first\"}]}\n\n"
	unblock := make(chan struct{})
	d := engine(t, "decode", func(w http.ResponseWriter, r *http.Request) {
		payload := parse(r)
		if payload["stream"] != true || payload["max_tokens"] != float64(8) || payload["min_tokens"] != float64(6) || payload["stream_options"] == nil || payload["kv_transfer_params"].(map[string]any)["future_field"] != "preserved" {
			t.Errorf("decode payload %v", payload)
		}
		w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
		fmt.Fprint(w, frame)
		w.(http.Flusher).Flush()
		select {
		case <-unblock:
			fmt.Fprint(w, "data: [DONE]\n\n")
		case <-r.Context().Done():
		}
	})
	s := ingress(t, "fixed-pdd", p, d)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	resp, err := request(ctx, s.URL, "job", body)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 || resp.Header.Get("X-EPP-Target") != d.ID || resp.Header.Get("X-EPP-Prefill") != p.ID || resp.Header.Get("X-KV-Remote-Request-Id") != "cmpl-job-0" || resp.Header.Get("X-Request-Id") != "job" {
		t.Fatalf("headers %d %v", resp.StatusCode, resp.Header)
	}
	first := make([]byte, len(frame))
	if _, err := io.ReadFull(resp.Body, first); err != nil || string(first) != frame {
		t.Fatalf("first frame %q: %v", first, err)
	}
	close(unblock)
	last, err := io.ReadAll(resp.Body)
	if err != nil || string(last) != "data: [DONE]\n\n" {
		t.Fatalf("final frame %q: %v", last, err)
	}
}

func TestNoDecodeOnPrefillFailure(t *testing.T) {
	var count atomic.Int32
	d := engine(t, "decode", func(http.ResponseWriter, *http.Request) { count.Add(1) })
	for _, c := range []struct {
		name   string
		code   int
		result string
	}{
		{"engine failed", 500, "unavailable"},
		{"unexpected producer", 200, strings.Replace(`{"kv_transfer_params":`+validKV+`}`, "producer", "another-engine", 1)},
		{"missing transfer", 200, `{"choices":[{"text":"decoder-only"}]}`},
		{"missing blocks", 200, strings.Replace(`{"kv_transfer_params":`+validKV+`}`, "[[1,2]]", "[[]]", 1)},
		{"missing remote request id", 200, strings.Replace(`{"kv_transfer_params":`+validKV+`}`, "cmpl-job-0", "", 1)},
	} {
		t.Run(c.name, func(t *testing.T) {
			p := engine(t, "prefill", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(c.code); fmt.Fprint(w, c.result) })
			s := ingress(t, "fixed-pdd", p, d)
			resp, err := request(context.Background(), s.URL, "x", body)
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			if resp.StatusCode != 502 || count.Load() != 0 {
				t.Fatalf("HTTP %d, %d decode calls", resp.StatusCode, count.Load())
			}
		})
	}
}

func TestGreedySlotReleasedOnCanceledStreamAndErrors(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	streamCtx, abort := context.WithCancel(ctx)
	defer abort()
	canceled := make(chan struct{})
	ep := engine(t, "full", func(w http.ResponseWriter, r *http.Request) {
		switch r.Header.Get("X-Request-Id") {
		case "stream":
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "data: first\n\n")
			w.(http.Flusher).Flush()
			<-r.Context().Done()
			close(canceled)
		case "error":
			http.Error(w, "engine error", 429)
		default:
			fmt.Fprint(w, `{"choices":[{"text":"next"}]}`)
		}
	})
	s := ingress(t, "greedy", ep)
	resp, err := request(streamCtx, s.URL, "stream", body)
	if err != nil {
		t.Fatal(err)
	}
	abort()
	resp.Body.Close()
	select {
	case <-canceled:
	case <-ctx.Done():
		t.Fatal("backend still running after cancel")
	}
	for _, tc := range []struct {
		id   string
		code int
	}{{"error", 429}, {"next", 200}} {
		next, err := request(ctx, s.URL, tc.id, body)
		if err != nil {
			t.Fatal(err)
		}
		next.Body.Close()
		if next.StatusCode != tc.code {
			t.Fatalf("%s status %d", tc.id, next.StatusCode)
		}
	}
}

func TestClientCannotSupplyTransferParamsOrWrongModel(t *testing.T) {
	ep := engine(t, "full", func(http.ResponseWriter, *http.Request) { t.Error("request reached backend") })
	s := ingress(t, "baseline", ep)
	for _, v := range []string{`{"model":"other"}`, `{"model":"model","kv_transfer_params":null}`} {
		resp, err := request(context.Background(), s.URL, "test", v)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusBadRequest {
			t.Fatalf("HTTP %d for %s", resp.StatusCode, v)
		}
	}
}

func TestFixedPDDRejectsBatchedPrompts(t *testing.T) {
	reject := func(http.ResponseWriter, *http.Request) { t.Error("unexpected backend invocation") }
	s := ingress(t, "fixed-pdd", engine(t, "prefill", reject), engine(t, "decode", reject))
	for _, payload := range []string{`{"model":"model","prompt":["a","b"]}`, `{"model":"model","prompt":[[11],[12]]}`, `{"model":"model","prompt":"p","n":2}`} {
		resp, err := request(context.Background(), s.URL, "job", payload)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusBadRequest {
			t.Fatalf("HTTP %d for %s", resp.StatusCode, payload)
		}
	}
}
