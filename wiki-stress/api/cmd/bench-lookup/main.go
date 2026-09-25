// Command bench-lookup measures real latency and throughput of a running
// stress-api instance over HTTP. It never invents numbers: every figure in
// its JSON output comes from a request actually made during this run,
// against the word sample and API base URL given on the command line.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"net/url"
	"os"
	"runtime"
	"sort"
	"time"
)

type latencySample struct {
	durations []time.Duration
	errors    int
}

func (s *latencySample) record(d time.Duration, err error) {
	if err != nil {
		s.errors++
		return
	}
	s.durations = append(s.durations, d)
}

type latencyStats struct {
	Count          int     `json:"count"`
	Errors         int     `json:"errors"`
	P50Ms          float64 `json:"p50_ms"`
	P95Ms          float64 `json:"p95_ms"`
	P99Ms          float64 `json:"p99_ms"`
	MinMs          float64 `json:"min_ms"`
	MaxMs          float64 `json:"max_ms"`
	ThroughputPerS float64 `json:"throughput_per_second"`
}

func (s *latencySample) stats(wallClock time.Duration) latencyStats {
	sorted := append([]time.Duration(nil), s.durations...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	percentile := func(p float64) float64 {
		if len(sorted) == 0 {
			return 0
		}
		index := int(p * float64(len(sorted)-1))
		return float64(sorted[index]) / float64(time.Millisecond)
	}
	var minMs, maxMs float64
	if len(sorted) > 0 {
		minMs = float64(sorted[0]) / float64(time.Millisecond)
		maxMs = float64(sorted[len(sorted)-1]) / float64(time.Millisecond)
	}
	throughput := 0.0
	if wallClock > 0 {
		throughput = float64(len(sorted)) / wallClock.Seconds()
	}
	return latencyStats{
		Count: len(sorted), Errors: s.errors,
		P50Ms: percentile(0.50), P95Ms: percentile(0.95), P99Ms: percentile(0.99),
		MinMs: minMs, MaxMs: maxMs, ThroughputPerS: throughput,
	}
}

func loadWords(path string) ([]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var words []string
	start := 0
	for i, b := range data {
		if b == '\n' {
			if i > start {
				words = append(words, string(data[start:i]))
			}
			start = i + 1
		}
	}
	if start < len(data) {
		words = append(words, string(data[start:]))
	}
	if len(words) == 0 {
		return nil, fmt.Errorf("no words found in %s", path)
	}
	return words, nil
}

func singleLookup(client *http.Client, baseURL, word string) (time.Duration, error) {
	start := time.Now()
	// A handful of real published words legitimately contain tab (a
	// structural multi-token separator, not a typo) or other characters
	// that are meaningless unescaped inside a URL query string; escaping
	// keeps this a test of the API's request handling rather than an
	// artifact of how this benchmark builds URLs.
	query := url.Values{"word": {word}}.Encode()
	resp, err := client.Get(baseURL + "/v1/lookup?" + query)
	if err != nil {
		return 0, err
	}
	defer func() {
		_, _ = io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
	}()
	elapsed := time.Since(start)
	if resp.StatusCode != http.StatusOK {
		return elapsed, fmt.Errorf("unexpected status %d", resp.StatusCode)
	}
	return elapsed, nil
}

func runSingleLookupPhase(client *http.Client, baseURL string, words []string, n int, rng *rand.Rand) latencyStats {
	sample := &latencySample{}
	start := time.Now()
	for i := 0; i < n; i++ {
		word := words[rng.Intn(len(words))]
		d, err := singleLookup(client, baseURL, word)
		sample.record(d, err)
	}
	return sample.stats(time.Since(start))
}

func batchLookup(client *http.Client, baseURL string, batch []string) (time.Duration, error) {
	body, err := json.Marshal(map[string]any{"words": batch})
	if err != nil {
		return 0, err
	}
	start := time.Now()
	resp, err := client.Post(baseURL+"/v1/lookup:batch", "application/json", bytes.NewReader(body))
	if err != nil {
		return 0, err
	}
	defer func() {
		_, _ = io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
	}()
	elapsed := time.Since(start)
	if resp.StatusCode != http.StatusOK {
		return elapsed, fmt.Errorf("unexpected status %d", resp.StatusCode)
	}
	return elapsed, nil
}

func runBatchPhase(client *http.Client, baseURL string, words []string, batchSize, repeats int, rng *rand.Rand) latencyStats {
	sample := &latencySample{}
	start := time.Now()
	for r := 0; r < repeats; r++ {
		batch := make([]string, batchSize)
		for i := range batch {
			batch[i] = words[rng.Intn(len(words))]
		}
		d, err := batchLookup(client, baseURL, batch)
		sample.record(d, err)
	}
	return sample.stats(time.Since(start))
}

func runConcurrencyPhase(client *http.Client, baseURL string, words []string, concurrency, perWorker int, rng *rand.Rand) latencyStats {
	sample := &latencySample{}
	results := make(chan struct {
		d   time.Duration
		err error
	}, concurrency*perWorker)
	start := time.Now()
	for w := 0; w < concurrency; w++ {
		go func(seed int64) {
			workerRng := rand.New(rand.NewSource(seed))
			for i := 0; i < perWorker; i++ {
				word := words[workerRng.Intn(len(words))]
				d, err := singleLookup(client, baseURL, word)
				results <- struct {
					d   time.Duration
					err error
				}{d, err}
			}
		}(rng.Int63())
	}
	for i := 0; i < concurrency*perWorker; i++ {
		r := <-results
		sample.record(r.d, r.err)
	}
	return sample.stats(time.Since(start))
}

type report struct {
	GeneratedAt string            `json:"generated_at"`
	BaseURL     string            `json:"base_url"`
	Hardware    map[string]string `json:"hardware"`
	WordSample  struct {
		Path        string `json:"path"`
		UniqueWords int    `json:"unique_words"`
	} `json:"word_sample"`
	ColdSingleLookup latencyStats `json:"cold_single_lookup"`
	WarmSingleLookup latencyStats `json:"warm_single_lookup"`
	Batch100         latencyStats `json:"batch_100"`
	Batch1000        latencyStats `json:"batch_1000"`
	Batch10000       latencyStats `json:"batch_10000"`
	Concurrency200   latencyStats `json:"concurrency_200"`
}

func main() {
	baseURL := flag.String("base-url", "http://localhost:8080", "stress-api base URL")
	wordsPath := flag.String("words", "../benchmarks/lookup_words.txt", "path to newline-delimited word sample")
	output := flag.String("output", "", "write JSON report to this path (default: stdout only)")
	flag.Parse()

	words, err := loadWords(*wordsPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "bench-lookup: %v\n", err)
		os.Exit(1)
	}

	client := &http.Client{Timeout: 10 * time.Second}
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	fmt.Fprintln(os.Stderr, "bench-lookup: cold single-lookup phase (50 requests, fresh connection each time)...")
	coldClient := &http.Client{
		Timeout:   10 * time.Second,
		Transport: &http.Transport{DisableKeepAlives: true},
	}
	cold := runSingleLookupPhase(coldClient, *baseURL, words, 50, rng)

	fmt.Fprintln(os.Stderr, "bench-lookup: warm single-lookup phase (500 requests, reused connections)...")
	warm := runSingleLookupPhase(client, *baseURL, words, 500, rng)

	fmt.Fprintln(os.Stderr, "bench-lookup: batch phase (100 words x 20 requests)...")
	batch100 := runBatchPhase(client, *baseURL, words, 100, 20, rng)

	fmt.Fprintln(os.Stderr, "bench-lookup: batch phase (1000 words x 10 requests)...")
	batch1000 := runBatchPhase(client, *baseURL, words, 1000, 10, rng)

	fmt.Fprintln(os.Stderr, "bench-lookup: batch phase (10000 words x 5 requests)...")
	batch10000 := runBatchPhase(client, *baseURL, words, 10000, 5, rng)

	fmt.Fprintln(os.Stderr, "bench-lookup: concurrency phase (200 concurrent workers x 10 requests)...")
	concurrency := runConcurrencyPhase(client, *baseURL, words, 200, 10, rng)

	rep := report{
		GeneratedAt:      time.Now().UTC().Format(time.RFC3339),
		BaseURL:          *baseURL,
		ColdSingleLookup: cold,
		WarmSingleLookup: warm,
		Batch100:         batch100,
		Batch1000:        batch1000,
		Batch10000:       batch10000,
		Concurrency200:   concurrency,
	}
	rep.Hardware = map[string]string{
		"go_version": runtime.Version(),
		"os_arch":    runtime.GOOS + "/" + runtime.GOARCH,
		"num_cpu":    fmt.Sprintf("%d", runtime.NumCPU()),
	}
	rep.WordSample.Path = *wordsPath
	rep.WordSample.UniqueWords = len(words)

	data, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "bench-lookup: marshal report: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(string(data))
	if *output != "" {
		if err := os.WriteFile(*output, data, 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "bench-lookup: write report: %v\n", err)
			os.Exit(1)
		}
	}
}
