// Package transport forwards OpenAI completions according to a single EPP decision.
package transport

import (
	"bytes"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"strings"
	"sync"

	"github.com/SOTA-PNU/hetero-pdd/llmd/epp/scheduling"
)

type router struct {
	model     string
	policy    string
	scheduler *scheduling.Scheduler
	producers map[string]Producer
	logger    *slog.Logger
	client    *http.Client
	active    sync.Map
}

func New(model, policy string, scheduler *scheduling.Scheduler, producers map[string]Producer, logger *slog.Logger) http.Handler {
	rt := &router{model: model, policy: policy, scheduler: scheduler, producers: producers, logger: logger,
		client: &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }},
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	mux.HandleFunc("POST /v1/completions", rt.complete)
	return mux
}

func (rt *router) complete(w http.ResponseWriter, r *http.Request) {
	id := r.Header.Get("X-Request-Id")
	if id == "" {
		id = rand.Text()
	}
	w.Header().Set("X-Request-Id", id)
	if _, exists := rt.active.LoadOrStore(id, true); exists {
		http.Error(w, "X-Request-Id is already in flight", http.StatusConflict)
		return
	}
	defer rt.active.Delete(id)

	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 16<<20))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	var payload map[string]json.RawMessage
	var model string
	if json.Unmarshal(body, &payload) != nil || json.Unmarshal(payload["model"], &model) != nil || model != rt.model {
		http.Error(w, "request must be a JSON object with the configured model", http.StatusBadRequest)
		return
	}
	if _, ok := payload["kv_transfer_params"]; ok {
		http.Error(w, "kv_transfer_params is assigned by the router", http.StatusBadRequest)
		return
	}

	if rt.policy == "fixed-pdd" && !singleCompletion(payload) {
		http.Error(w, "fixed-pdd requires one text prompt or token-ID sequence, n=1, and no beam search", http.StatusBadRequest)
		return
	}

	logger := rt.logger.With("request_id", id, "policy", rt.policy)
	selection, err := rt.scheduler.Schedule(r.Context(), id)
	if err != nil {
		logger.Error("schedule_failed", "error", err)
		http.Error(w, err.Error(), http.StatusServiceUnavailable)
		return
	}
	defer selection.Release()
	logger = logger.With("target", selection.Target.ID, "target_url", selection.Target.URL)
	if pre := selection.Prefill; pre != nil {
		logger = logger.With("prefill", pre.ID, "prefill_url", pre.URL)
	}
	if selection.WaitingRequests != nil {
		logger.Info("selected", "waiting_requests", selection.WaitingRequests)
	} else {
		logger.Info("selected")
	}

	if pre := selection.Prefill; pre != nil {
		transfer, meta, err := rt.prefill(r, *pre, id, payload)
		if err != nil {
			logger.Error("prefill_failed", "error", err)
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		logger.Info("prefill_metadata", "remote_request_id", meta.RequestID, "remote_engine_id", meta.EngineID,
			"remote_host", meta.Host, "remote_port", meta.Port, "remote_block_count", meta.blockCount(), "tp_size", meta.TP)
		payload["kv_transfer_params"] = transfer
		body, _ = json.Marshal(payload) // all values came from parsed JSON
		w.Header().Set("X-EPP-Prefill", pre.ID)
		w.Header().Set("X-KV-Remote-Request-Id", meta.RequestID)
	}
	resp, err := rt.post(r, selection.Target.URL, id, body)
	if err != nil {
		logger.Error("target_failed", "error", err)
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()
	u, _ := url.Parse(selection.Target.URL) // validated at startup
	w.Header().Set("X-EPP-Target", selection.Target.ID)
	w.Header().Set("X-Decode-Target", u.Host)
	w.Header().Set("Content-Type", resp.Header.Get("Content-Type"))
	w.WriteHeader(resp.StatusCode)
	// Flush SSE frames as they arrive. Cancellation of the original request
	// also cancels the request to the engine (including a blocked body read).
	var dest io.Writer = w
	if strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream") {
		dest = &flushWriter{w}
	}
	n, copyErr := io.Copy(dest, resp.Body)
	if copyErr != nil {
		logger.Error("response_interrupted", "status", resp.StatusCode, "bytes", n, "error", copyErr)
		// Prevent a premature EOF from becoming a valid, shorter 200 response.
		panic(http.ErrAbortHandler)
	}
	logger.Info("response_finished", "status", resp.StatusCode, "bytes", n)
}

// vLLM CompletionResponse carries the first result's KV handle even when
// the request contains several prompts. Each transfer requires its own handle.
func singleCompletion(payload map[string]json.RawMessage) bool {
	for _, key := range []string{"n", "best_of"} {
		if value, ok := payload[key]; ok && string(value) != "1" {
			return false
		}
	}
	if string(payload["use_beam_search"]) == "true" || payload["prompt_embeds"] != nil {
		return false
	}
	prompt := payload["prompt"]
	if len(prompt) > 0 && prompt[0] == '"' {
		return true
	}
	var tokens []int
	return json.Unmarshal(prompt, &tokens) == nil && len(tokens) > 0
}

type flushWriter struct{ w http.ResponseWriter }

func (f *flushWriter) Write(p []byte) (int, error) {
	n, err := f.w.Write(p)
	if err == nil {
		err = http.NewResponseController(f.w).Flush()
	}
	return n, err
}

func (rt *router) post(r *http.Request, origin, id string, body []byte) (*http.Response, error) {
	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, origin+"/v1/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Request-Id", id)
	for _, key := range []string{"Authorization", "traceparent", "tracestate"} {
		if value := r.Header.Get(key); value != "" {
			req.Header.Set(key, value)
		}
	}
	return rt.client.Do(req)
}

// Strict NIXLv2 wire adapter. llm-d v0.5.1 pkg/sidecar/proxy/connector_nixlv2.go
// exposes its protocol runner only internally and forwards missing transfer
// metadata. The RBLN path requires complete producer metadata before decode.
type Producer struct {
	EngineID string `json:"engine_id"`
	Host     string `json:"host"`
	Port     int    `json:"port"`
}

type kvMetadata struct {
	RemotePrefill bool    `json:"do_remote_prefill"`
	RemoteDecode  bool    `json:"do_remote_decode"`
	EngineID      string  `json:"remote_engine_id"`
	RequestID     string  `json:"remote_request_id"`
	Host          string  `json:"remote_host"`
	Port          int     `json:"remote_port"`
	TP            int     `json:"tp_size"`
	BlockIDs      [][]int `json:"remote_block_ids"`
}

func (m kvMetadata) blockCount() int {
	count := 0
	for _, group := range m.BlockIDs {
		for _, id := range group {
			if id < 0 {
				return 0
			}
		}
		count += len(group)
	}
	return count
}

func (rt *router) prefill(r *http.Request, endpoint scheduling.Endpoint, id string, original map[string]json.RawMessage) (json.RawMessage, kvMetadata, error) {
	var meta kvMetadata
	payload := make(map[string]json.RawMessage, len(original)+1)
	for k, v := range original {
		payload[k] = v
	}
	payload["kv_transfer_params"] = json.RawMessage(`{"do_remote_decode":true,"do_remote_prefill":false,"remote_engine_id":null,"remote_block_ids":null,"remote_host":null,"remote_port":null}`)
	payload["stream"] = json.RawMessage("false")
	delete(payload, "stream_options")
	payload["max_tokens"] = json.RawMessage("1")
	if _, ok := payload["max_completion_tokens"]; ok {
		payload["max_completion_tokens"] = json.RawMessage("1")
	}
	// A length-capped prefill is required for NIXL to retain blocks even for
	// prompts whose first generated token would already satisfy a stop condition.
	payload["min_tokens"] = json.RawMessage("1")
	payload["ignore_eos"] = json.RawMessage("true")
	delete(payload, "stop")
	delete(payload, "stop_token_ids")
	body, _ := json.Marshal(payload)
	resp, err := rt.post(r, endpoint.URL, id, body)
	if err != nil {
		return nil, meta, fmt.Errorf("prefill: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return nil, meta, fmt.Errorf("prefill returned HTTP %d: %s", resp.StatusCode, msg)
	}
	var out struct {
		KV json.RawMessage `json:"kv_transfer_params"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, meta, fmt.Errorf("prefill response: %w", err)
	}
	if err := json.Unmarshal(out.KV, &meta); err != nil || !meta.RemotePrefill || meta.RemoteDecode || meta.EngineID == "" || meta.RequestID == "" || meta.Host == "" || meta.Port <= 0 || meta.Port > 65535 || meta.TP != 1 || meta.blockCount() == 0 {
		return nil, meta, errors.New("prefill returned invalid NIXL transfer metadata or no remote blocks")
	}
	producer := rt.producers[endpoint.ID]
	if meta.EngineID != producer.EngineID || meta.Host != producer.Host || meta.Port != producer.Port {
		return nil, meta, fmt.Errorf("prefill %s returned NIXL identity %s %s:%d, expected %s %s:%d", endpoint.ID, meta.EngineID, meta.Host, meta.Port, producer.EngineID, producer.Host, producer.Port)
	}
	return out.KV, meta, nil
}
