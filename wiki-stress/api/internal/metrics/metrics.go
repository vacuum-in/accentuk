// Package metrics implements bounded-cardinality request and pool metrics.
// Label values are always drawn from a fixed, small set (route pattern,
// HTTP status, lookup status) — never from raw request content — so
// cardinality cannot grow with traffic.
package metrics

import (
	"fmt"
	"io"
	"sort"
	"sync"
	"sync/atomic"
)

type requestKey struct {
	Endpoint string
	Status   int
}

type Metrics struct {
	mu            sync.Mutex
	requestsTotal map[requestKey]*atomic.Int64

	lookupFound     atomic.Int64
	lookupAmbiguous atomic.Int64
	lookupNotFound  atomic.Int64
	lookupTruncated atomic.Int64

	poolAcquired atomic.Int64
	poolIdle     atomic.Int64
	poolMax      atomic.Int64

	activeDataset atomic.Int64
}

func New() *Metrics {
	return &Metrics{requestsTotal: make(map[requestKey]*atomic.Int64)}
}

// RecordRequest counts one completed request. endpoint must be a route
// pattern (e.g. "GET /v1/lookup"), never a raw path containing user input.
func (m *Metrics) RecordRequest(endpoint string, status int) {
	key := requestKey{Endpoint: endpoint, Status: status}
	m.mu.Lock()
	counter, ok := m.requestsTotal[key]
	if !ok {
		counter = &atomic.Int64{}
		m.requestsTotal[key] = counter
	}
	m.mu.Unlock()
	counter.Add(1)
}

// RecordLookup counts one lookup outcome. status must be one of
// found/ambiguous/not_found; raw words are never recorded.
func (m *Metrics) RecordLookup(status string, truncated bool) {
	switch status {
	case "found":
		m.lookupFound.Add(1)
	case "ambiguous":
		m.lookupAmbiguous.Add(1)
	case "not_found":
		m.lookupNotFound.Add(1)
	}
	if truncated {
		m.lookupTruncated.Add(1)
	}
}

func (m *Metrics) SetPoolStats(acquired, idle, max int32) {
	m.poolAcquired.Store(int64(acquired))
	m.poolIdle.Store(int64(idle))
	m.poolMax.Store(int64(max))
}

func (m *Metrics) SetActiveDataset(id int64) {
	m.activeDataset.Store(id)
}

// Render writes metrics in Prometheus text exposition format.
func (m *Metrics) Render(w io.Writer) {
	fmt.Fprintln(w, "# TYPE ukstress_lookup_total counter")
	fmt.Fprintf(w, "ukstress_lookup_total{status=\"found\"} %d\n", m.lookupFound.Load())
	fmt.Fprintf(w, "ukstress_lookup_total{status=\"ambiguous\"} %d\n", m.lookupAmbiguous.Load())
	fmt.Fprintf(w, "ukstress_lookup_total{status=\"not_found\"} %d\n", m.lookupNotFound.Load())

	fmt.Fprintln(w, "# TYPE ukstress_lookup_truncated_total counter")
	fmt.Fprintf(w, "ukstress_lookup_truncated_total %d\n", m.lookupTruncated.Load())

	fmt.Fprintln(w, "# TYPE ukstress_requests_total counter")
	m.mu.Lock()
	keys := make([]requestKey, 0, len(m.requestsTotal))
	for key := range m.requestsTotal {
		keys = append(keys, key)
	}
	m.mu.Unlock()
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].Endpoint != keys[j].Endpoint {
			return keys[i].Endpoint < keys[j].Endpoint
		}
		return keys[i].Status < keys[j].Status
	})
	for _, key := range keys {
		m.mu.Lock()
		counter := m.requestsTotal[key]
		m.mu.Unlock()
		fmt.Fprintf(w, "ukstress_requests_total{endpoint=%q,status=\"%d\"} %d\n",
			key.Endpoint, key.Status, counter.Load())
	}

	fmt.Fprintln(w, "# TYPE ukstress_pool_connections gauge")
	fmt.Fprintf(w, "ukstress_pool_connections{state=\"acquired\"} %d\n", m.poolAcquired.Load())
	fmt.Fprintf(w, "ukstress_pool_connections{state=\"idle\"} %d\n", m.poolIdle.Load())
	fmt.Fprintf(w, "ukstress_pool_connections{state=\"max\"} %d\n", m.poolMax.Load())

	fmt.Fprintln(w, "# TYPE ukstress_active_dataset gauge")
	fmt.Fprintf(w, "ukstress_active_dataset %d\n", m.activeDataset.Load())
}
