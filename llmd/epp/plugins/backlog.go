package plugins

import (
	"math"
	"slices"
	"strconv"
	"time"
)

// All stages and ledger transitions are protected by costPolicy.mu. The primary
// endpoint remains committed until a terminal hook. A producer only contributes
// its short prefill stage, even if the consumer runs for several more seconds.
type stage struct {
	id                  string
	visible, start, end time.Time
}
type reservation struct {
	producer, primary stage
	assigned, first   time.Time
	bytes, decodeMs   float64
	output            int
}

func at(now time.Time, ms float64) time.Time {
	return now.Add(time.Duration(ms * float64(time.Millisecond)))
}
func millis(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }
func newStage(id string, now time.Time, visible, start, finish float64) stage {
	return stage{id, at(now, visible), at(now, start), at(now, finish)}
}

// Only keep timestamps for reconciling a still-valid scrape that predates EOS.
// These entries own no engine or host resources.
func (h *costPolicy) pruneRetired(now time.Time) {
	cutoff := now.Add(-time.Duration(h.TTL) * time.Millisecond)
	h.retired = slices.DeleteFunc(h.retired, func(s stage) bool { return !s.end.After(cutoff) })
}
func (h *costPolicy) retire(work reservation, now time.Time) {
	h.pruneRetired(now)
	if work.producer.id != "" {
		p := work.producer
		if p.end.After(now) {
			p.end = now
		}
		h.retired = append(h.retired, p)
	}
	d := work.primary
	d.end = now
	h.retired = append(h.retired, d)
}

func (h *costPolicy) backlog(id string, slots int, cost costs, live liveState, now time.Time) (float64, []stage) {
	// Aggregate gauges do not carry request IDs. Treat work known to this EPP
	// at the sample time as part of that count, and charge only the excess as
	// external jobs. Include terminal timestamps to avoid resurrecting an EOS
	// completed between the last metrics poll and this scheduling decision.
	var localAtScrape int
	var pending, decoding []stage
	for _, past := range h.retired {
		if past.id == id && !past.visible.After(live.At) && past.end.After(live.At) {
			localAtScrape++
		}
	}
	for _, r := range h.reservations {
		if p := r.producer; p.id == id {
			if !p.visible.After(live.At) && p.end.After(live.At) {
				localAtScrape++
			}
			if p.end.After(now) {
				pending = append(pending, p)
			}
		}
		if d := r.primary; d.id == id {
			if !d.visible.After(live.At) {
				localAtScrape++
			}
			if !d.end.After(now) {
				d.end = at(now, cost.Decode)
				d.start = now
			}
			pending = append(pending, d)
			decoding = append(decoding, stage{start: r.first, end: d.end})
		}
	}
	unknown := max(0, int(live.Queue+live.Running)-localAtScrape)
	externalTokens := cost.Remaining
	if externalTokens == 0 {
		externalTokens = h.Prior
	}
	quantum := float64(externalTokens) * cost.Decode
	// Each slot has its own forecast. A free engine batch slot can accept
	// work immediately, while a full slot drains its committed stages in order.
	availability := make([]float64, slots)
	for slot := range availability {
		waves := (unknown + slots - 1 - slot) / slots
		availability[slot] = float64(waves) * quantum
	}
	if live.Running > 0 && unknown > 0 {
		decoding = append(decoding, stage{start: now, end: at(now, quantum)})
	}
	slices.SortFunc(pending, func(a, b stage) int { return a.start.Compare(b.start) })
	for _, w := range pending {
		earliest := 0
		for i := 1; i < slots; i++ {
			if availability[i] < availability[earliest] {
				earliest = i
			}
		}
		remaining := millis(w.end.Sub(w.start))
		if w.start.Before(now) {
			remaining = millis(w.end.Sub(now))
		}
		ready := millis(w.visible.Sub(now))
		availability[earliest] = math.Max(availability[earliest], ready) + math.Max(0, remaining)
	}
	return slices.Min(availability), decoding
}

func (h *costPolicy) credit(e pricedEndpoint, bytes float64) bool {
	slots, free := e.state.HostSlots, e.state.HostBytes
	var count, held, newCount, newBytes float64
	for _, r := range h.reservations {
		if r.primary.id != endpointID(e.e) || r.bytes == 0 {
			continue
		}
		count++
		held += r.bytes
		if r.assigned.After(e.state.At) {
			newCount++
			newBytes += r.bytes
		}
	}
	depth, depthErr := strconv.Atoi(label(e.e, "lookahead-depth"))
	budget, budgetErr := strconv.ParseFloat(label(e.e, "lookahead-bytes"), 64)
	if depthErr == nil && budgetErr == nil {
		slots = math.Min(slots-newCount, float64(depth)-count)
		free = math.Min(free-newBytes, budget-held)
	} else {
		// Older hand-written discovery files have no capacity labels.
		slots -= count
		free -= held
	}
	return bytes > 0 && slots >= 1 && free >= bytes
}
