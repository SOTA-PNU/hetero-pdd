package scheduling

import (
	"container/list"
	"context"
	"slices"
	"sync"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

var _ framework.Picker = (*roundRobin)(nil)
var _ framework.Picker = (*slotPicker)(nil)

// Each EPP profile holds its own cursor. The scored slice comes from an
// upstream map, so recover YAML order before advancing it.
type roundRobin struct {
	order map[string]int
	mu    sync.Mutex
	next  int
}

func newRoundRobin(endpoints []Endpoint) *roundRobin {
	p := &roundRobin{order: make(map[string]int)}
	for i, endpoint := range endpoints {
		p.order[endpoint.ID] = i
	}
	return p
}

func (*roundRobin) TypedName() plugins.TypedName {
	return plugins.TypedName{Type: "round-robin-picker", Name: "round-robin-picker"}
}

func (p *roundRobin) Pick(_ context.Context, _ *types.CycleState, pods []*types.ScoredPod) *types.ProfileRunResult {
	slices.SortFunc(pods, func(a, b *types.ScoredPod) int {
		return p.order[a.GetPod().PodName] - p.order[b.GetPod().PodName]
	})
	p.mu.Lock()
	defer p.mu.Unlock()
	selected := pods[p.next%len(pods)]
	p.next = (p.next + 1) % len(pods)
	return &types.ProfileRunResult{TargetPods: []types.Pod{selected}}
}

// slotPicker mirrors tools/hetero/loadgen.py: the initial worker list
// follows tools/hetero/profiler/run.py: one slot per endpoint per round. Completions hand each freed slot to the oldest
// waiting request. Mutex acquisition defines arrival/completion order for
// simultaneous calls. All eligible engines are static for this profile.
type slotPicker struct {
	mu      sync.Mutex
	free    []string
	waiters list.List // FIFO of buffered chan string
}

func newSlotPicker(endpoints []Endpoint) *slotPicker {
	p := &slotPicker{}
	maxSlots := 0
	for _, endpoint := range endpoints {
		maxSlots = max(maxSlots, endpoint.Slots)
	}
	for slot := 0; slot < maxSlots; slot++ {
		for _, endpoint := range endpoints {
			if slot < endpoint.Slots {
				p.free = append(p.free, endpoint.ID)
			}
		}
	}
	return p
}

func (*slotPicker) TypedName() plugins.TypedName {
	return plugins.TypedName{Type: "first-available-picker", Name: "first-available-picker"}
}

func (p *slotPicker) Pick(ctx context.Context, _ *types.CycleState, pods []*types.ScoredPod) *types.ProfileRunResult {
	id := p.acquire(ctx)
	if id == "" {
		return nil
	}
	for _, pod := range pods {
		if pod.GetPod().PodName == id {
			return &types.ProfileRunResult{TargetPods: []types.Pod{pod}}
		}
	}
	p.release(id)
	return nil
}

func (p *slotPicker) acquire(ctx context.Context) string {
	p.mu.Lock()
	if len(p.free) > 0 {
		id := p.free[0]
		p.free = p.free[1:]
		p.mu.Unlock()
		return id
	}
	ch := make(chan string, 1)
	waiter := p.waiters.PushBack(ch)
	p.mu.Unlock()
	select {
	case id := <-ch:
		return id
	case <-ctx.Done():
		p.mu.Lock()
		defer p.mu.Unlock()
		select {
		case id := <-ch:
			// Handoff raced with cancellation. Return the slot immediately.
			p.releaseLocked(id)
		default:
			p.waiters.Remove(waiter)
		}
		return ""
	}
}

func (p *slotPicker) release(id string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.releaseLocked(id)
}

func (p *slotPicker) releaseLocked(id string) {
	if next := p.waiters.Front(); next != nil {
		p.waiters.Remove(next)
		next.Value.(chan string) <- id
		return
	}
	p.free = append(p.free, id)
}
