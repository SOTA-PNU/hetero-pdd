package plugins

import (
	"container/list"
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"sync"

	dl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	api "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	rc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	s "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

type slot struct {
	id   string
	rank int
}
type waiter struct {
	candidates map[string]s.Endpoint
	ready      chan slot
}
type greedy struct {
	named
	runAll
	mu       sync.Mutex
	free     []slot
	known    map[slot]bool
	capacity map[string]int
	owners   map[string]slot
	waiters  list.List
}

func newGreedy(n named, _ *json.Decoder) (api.Plugin, error) {
	return &greedy{named: n, known: map[slot]bool{}, capacity: map[string]int{}, owners: map[string]slot{}}, nil
}

// Publish completed tokens to the oldest compatible waiter. Requests whose
// candidate set has been filtered differently cannot consume each other's slots.
func (g *greedy) dispatch() {
	for w := g.waiters.Front(); w != nil; {
		next := w.Next()
		waiting := w.Value.(*waiter)
		for i, sl := range g.free {
			if waiting.candidates[sl.id] != nil {
				g.free = append(g.free[:i], g.free[i+1:]...)
				g.waiters.Remove(w)
				waiting.ready <- sl
				break
			}
		}
		w = next
	}
}
func (g *greedy) ProcessResults(ctx context.Context, r *s.InferenceRequest, results map[string]*s.ProfileRunResult) (*s.SchedulingResult, error) {
	es := candidates(results, "full")
	if len(es) == 0 {
		return nil, fmt.Errorf("no full endpoint for greedy scheduling")
	}
	w := &waiter{candidates: map[string]s.Endpoint{}, ready: make(chan slot, 1)}
	counts := map[string]int{}
	maxSlots := 0
	for _, e := range es {
		id := endpointID(e.Endpoint)
		n, err := strconv.Atoi(label(e.Endpoint, "slots"))
		if err != nil || n < 1 || n > 65536 || id == "" {
			return nil, fmt.Errorf("invalid slot capacity or ID for %q", id)
		}
		if _, dup := counts[id]; dup {
			return nil, fmt.Errorf("duplicate endpoint ID %q", id)
		}
		counts[id] = n
		maxSlots = max(maxSlots, n)
		w.candidates[id] = e.Endpoint
	}
	g.mu.Lock()
	for id, n := range counts {
		g.capacity[id] = n
	}
	for i := 0; i < maxSlots; i++ {
		for _, e := range es {
			id := endpointID(e.Endpoint)
			sl := slot{id, i}
			if i < counts[id] && !g.known[sl] {
				g.known[sl] = true
				g.free = append(g.free, sl)
			}
		}
	}
	// Remove superseded capacity for members in this candidate snapshot. An
	// outstanding token above the new capacity is retired when released.
	filtered := g.free[:0]
	for _, sl := range g.free {
		if sl.rank < g.capacity[sl.id] {
			filtered = append(filtered, sl)
		} else {
			delete(g.known, sl)
		}
	}
	g.free = filtered
	elem := g.waiters.PushBack(w)
	g.dispatch()
	g.mu.Unlock()
	select {
	case sl := <-w.ready:
		g.mu.Lock()
		g.owners[r.RequestID] = sl
		g.mu.Unlock()
		if err := ctx.Err(); err != nil {
			g.release(r.RequestID)
			return nil, err
		}
		return fullResult(w.candidates[sl.id]), nil
	case <-ctx.Done():
		g.mu.Lock()
		g.waiters.Remove(elem)
		select {
		case sl := <-w.ready:
			g.returnLocked(sl)
			g.dispatch()
		default:
		}
		g.mu.Unlock()
		return nil, ctx.Err()
	}
}
func (g *greedy) release(requestID string) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if sl, ok := g.owners[requestID]; ok {
		delete(g.owners, requestID)
		g.returnLocked(sl)
		g.dispatch()
	}
}
func (g *greedy) ResponseBody(_ context.Context, r *s.InferenceRequest, resp *rc.Response, _ *dl.EndpointMetadata) {
	if resp.EndOfStream {
		g.release(r.RequestID)
	}
}

func (g *greedy) returnLocked(sl slot) {
	if sl.rank < g.capacity[sl.id] {
		g.free = append(g.free, sl)
	} else {
		delete(g.known, sl)
	}
}
