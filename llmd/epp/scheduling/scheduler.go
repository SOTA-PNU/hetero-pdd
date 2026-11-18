// Package scheduling binds the llm-d EPP scheduling graph to the local engine
// endpoints supplied by the experiment launcher.
package scheduling

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"sync"
	"time"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/plugins/filter"
	pd "github.com/llm-d/llm-d-inference-scheduler/pkg/plugins/profile"
	ktypes "k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	metrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	epp "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/picker"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/profile"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/scorer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

type Config struct {
	Model  string `json:"-"`
	Policy string `json:"policy"`
}

type Endpoint struct {
	ID    string `json:"id"`
	URL   string `json:"url"`
	Role  string `json:"role"`
	Slots int    `json:"slots"`
}

// Release ends the reservation after the final response byte, backend error,
// or cancellation. It is safe to call repeatedly.
type Selection struct {
	WaitingRequests map[string]int // per-endpoint queue lengths used by the scorer
	Target          Endpoint
	Prefill         *Endpoint
	Release         func()
}

type Scheduler struct {
	model     string
	policy    string
	endpoints map[string]Endpoint
	pods      []types.Pod
	epp       *epp.Scheduler
	slots     *slotPicker
	queues    *metrics.PodMetricsClientImpl
}

func New(cfg Config, endpoints []Endpoint) (*Scheduler, error) {
	if len(endpoints) == 0 {
		return nil, errors.New("no engine endpoints")
	}
	s := &Scheduler{model: cfg.Model, policy: cfg.Policy, endpoints: make(map[string]Endpoint)}
	roleCount := make(map[string]int)
	urls := make(map[string]bool)
	for _, engine := range endpoints {
		u, err := url.Parse(engine.URL)
		if err != nil || u.Scheme != "http" || u.Hostname() == "" || u.Port() == "" || u.User != nil || u.Path != "" || u.RawQuery != "" || u.Fragment != "" {
			return nil, fmt.Errorf("engine %q requires an HTTP origin with an explicit port", engine.ID)
		}
		if engine.ID == "" || engine.Slots < 1 || urls[engine.URL] {
			return nil, fmt.Errorf("invalid engine ID, slots or duplicate origin for %q", engine.ID)
		}
		if _, exists := s.endpoints[engine.ID]; exists {
			return nil, fmt.Errorf("duplicate engine ID %q", engine.ID)
		}
		if cfg.Policy == "fixed-pdd" {
			if engine.Role != "decode" && engine.Role != "prefill" {
				return nil, fmt.Errorf("fixed-pdd engine %s must be prefill or decode", engine.ID)
			}
		} else if engine.Role != "full" {
			return nil, fmt.Errorf("%s requires whole-request engines, got %s", cfg.Policy, engine.Role)
		}
		roleCount[engine.Role]++
		urls[engine.URL] = true
		s.endpoints[engine.ID] = engine
		s.pods = append(s.pods, &types.PodMetrics{
			Pod: &backend.Pod{NamespacedName: ktypes.NamespacedName{Name: engine.ID}, PodName: engine.ID,
				Address: u.Hostname(), Port: u.Port(), MetricsHost: u.Host,
				Labels: map[string]string{filter.RoleLabel: engine.Role}},
			MetricsState: &metrics.MetricsState{},
		})
	}
	full := framework.NewSchedulerProfile()
	profiles := map[string]*framework.SchedulerProfile{"full": full}
	var handler framework.ProfileHandler = profile.NewSingleProfileHandler()
	switch cfg.Policy {
	case "baseline":
		full.WithPicker(picker.NewRandomPicker(1))
	case "round-robin":
		full.WithPicker(newRoundRobin(endpoints))
	case "greedy":
		s.slots = newSlotPicker(endpoints)
		full.WithPicker(s.slots)
	case "least-queue":
		s.queues = &metrics.PodMetricsClientImpl{
			Client:                   &http.Client{Timeout: 2 * time.Second},
			ModelServerMetricsScheme: "http", ModelServerMetricsPath: "/metrics",
			MetricMapping: &metrics.MetricMapping{TotalQueuedRequests: &metrics.MetricSpec{MetricName: "vllm:num_requests_waiting"}},
		}
		full.WithScorers(framework.NewWeightedScorer(scorer.NewQueueScorer(), 1)).WithPicker(picker.NewMaxScorePicker(1))
	case "fixed-pdd":
		if roleCount["prefill"] == 0 || roleCount["decode"] == 0 {
			return nil, errors.New("fixed-pdd requires both prefill and decode engines")
		}
		// In llm-d v0.5.1 a zero threshold runs the prefill profile after
		// every successful decode selection. No cache-affinity plugin is used.
		handler = pd.NewPdProfileHandler("prefill", "decode", "", "", 0, 1, 0)
		profiles = map[string]*framework.SchedulerProfile{
			"prefill": framework.NewSchedulerProfile().WithFilters(filter.NewPrefillRole()).WithPicker(newRoundRobin(endpoints)),
			"decode":  framework.NewSchedulerProfile().WithFilters(filter.NewDecodeRole()).WithPicker(newRoundRobin(endpoints)),
		}
	default:
		return nil, fmt.Errorf("unknown EPP policy %q", cfg.Policy)
	}
	s.epp = epp.NewSchedulerWithConfig(epp.NewSchedulerConfig(handler, profiles))
	return s, nil
}

func (s *Scheduler) Schedule(ctx context.Context, id string) (Selection, error) {
	if err := ctx.Err(); err != nil {
		return Selection{}, err
	}
	pods := s.pods
	var waitingRequests map[string]int
	if s.queues != nil {
		// Each decision uses a fresh snapshot. Unknown queue lengths must not
		// become zero-length queues and bias the comparison.
		pods = make([]types.Pod, 0, len(s.pods))
		waitingRequests = make(map[string]int, len(s.pods))
		for _, pod := range s.pods {
			state, err := s.queues.FetchMetrics(ctx, pod.GetPod(), pod.GetMetrics())
			if err != nil {
				return Selection{}, fmt.Errorf("read queue of %s: %w", pod.GetPod().PodName, err)
			}
			waitingRequests[pod.GetPod().PodName] = state.WaitingQueueSize
			pods = append(pods, &types.PodMetrics{Pod: pod.GetPod(), MetricsState: state})
		}
	}
	result, err := s.epp.Schedule(ctx, &types.LLMRequest{RequestId: id, TargetModel: s.model}, pods)
	if err != nil {
		if ctx.Err() != nil {
			return Selection{}, ctx.Err()
		}
		return Selection{}, err
	}
	primary := result.PrimaryProfileName
	target := s.endpoints[result.ProfileResults[primary].TargetPods[0].GetPod().PodName]
	selection := Selection{Target: target, WaitingRequests: waitingRequests, Release: func() {}}
	if s.slots != nil {
		selection.Release = sync.OnceFunc(func() { s.slots.release(target.ID) })
	}
	if s.policy == "fixed-pdd" {
		// Upstream permits decode-only fallback when the prefill profile fails.
		// This experiment requires proof of both selections.
		prefill := result.ProfileResults["prefill"]
		if prefill == nil || len(prefill.TargetPods) != 1 {
			return Selection{}, errors.New("fixed-pdd prefill selection failed")
		}
		p := s.endpoints[prefill.TargetPods[0].GetPod().PodName]
		selection.Prefill = &p
	}
	if err := ctx.Err(); err != nil {
		selection.Release()
		return Selection{}, err
	}
	return selection, nil
}
