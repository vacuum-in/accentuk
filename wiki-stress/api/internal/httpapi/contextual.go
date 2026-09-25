package httpapi

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"time"
)

type coverageEntry struct {
	Signatures []string `json:"signatures"`
}

type coverageManifest struct {
	ModelVersion  string                   `json:"model_version"`
	InventoryHash string                   `json:"inventory_hash"`
	Threshold     float64                  `json:"threshold"`
	Forms         map[string]coverageEntry `json:"forms"`
}

type HTTPContextResolver struct {
	url          string
	client       *http.Client
	manifest     coverageManifest
	manifestHash string
	// morphology parses are slower than a model pass and get their own budget.
	morphologyClient *http.Client
}

// NewHTTPContextResolver validates the immutable serving manifest at startup.
// An expected inventory hash can be supplied from deployment metadata; a
// mismatch disables model startup instead of allowing inventory drift.
func NewHTTPContextResolver(url, manifestPath, expectedInventoryHash string, timeout time.Duration) (*HTTPContextResolver, error) {
	return NewHTTPContextResolverWithTimeouts(url, manifestPath, expectedInventoryHash,
		timeout, 15*time.Second)
}

// NewHTTPContextResolverWithTimeouts gives the parser its own deadline.
func NewHTTPContextResolverWithTimeouts(
	url, manifestPath, expectedInventoryHash string,
	timeout, morphologyTimeout time.Duration,
) (*HTTPContextResolver, error) {
	content, err := os.ReadFile(manifestPath)
	if err != nil {
		return nil, fmt.Errorf("read coverage manifest: %w", err)
	}
	var manifest coverageManifest
	if err := json.Unmarshal(content, &manifest); err != nil {
		return nil, fmt.Errorf("decode coverage manifest: %w", err)
	}
	if manifest.ModelVersion == "" || manifest.InventoryHash == "" || manifest.Threshold <= 0 {
		return nil, errors.New("coverage manifest is incomplete")
	}
	if expectedInventoryHash != "" && manifest.InventoryHash != expectedInventoryHash {
		return nil, errors.New("coverage manifest inventory does not match active inventory")
	}
	digest := sha256.Sum256(content)
	return &HTTPContextResolver{url: url, client: &http.Client{Timeout: timeout},
		manifest: manifest, manifestHash: hex.EncodeToString(digest[:]),
		morphologyClient: &http.Client{Timeout: morphologyTimeout}}, nil
}

func (r *HTTPContextResolver) Metadata() (string, string) {
	return r.manifest.ModelVersion, r.manifestHash
}

type inferenceRequest struct {
	ModelVersion  string        `json:"model_version"`
	InventoryHash string        `json:"inventory_hash"`
	Targets       []ModelTarget `json:"targets"`
}

type inferenceResponse struct {
	ModelVersion  string          `json:"model_version"`
	InventoryHash string          `json:"inventory_hash"`
	Decisions     []ModelDecision `json:"decisions"`
}

func sameStrings(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	a, b := append([]string(nil), left...), append([]string(nil), right...)
	sort.Strings(a)
	sort.Strings(b)
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func scoreKeys(scores map[string]float64) []string {
	keys := make([]string, 0, len(scores))
	for key := range scores {
		keys = append(keys, key)
	}
	return keys
}

// MorphologyTarget is an ambiguous span the contextual model cannot take,
// because the form is outside the serving manifest.
type MorphologyTarget struct {
	Index    int    `json:"index"`
	Sentence string `json:"sentence"`
	Start    int    `json:"start"`
	End      int    `json:"end"`
	Form     string `json:"form"`
}

type morphologyRequest struct {
	Targets []MorphologyTarget `json:"targets"`
}

// MorphologyAnswer is one parser decision, and whether a syntactic rule
// produced it rather than a plain tag match.
type MorphologyAnswer struct {
	Stressed string
	Rule     string
}

type morphologyDecision struct {
	Index    int    `json:"index"`
	Stressed string `json:"stressed"`
	Rule     string `json:"rule"`
}

type morphologyResponse struct {
	Decisions []morphologyDecision `json:"decisions"`
}

// ResolveMorphology answers ambiguities whose readings differ by grammar
// rather than by sense.
//
// These must not go to the cross-encoder. It scores (sentence, gloss) pairs,
// and two glosses that differ only in a case label are nearly identical
// inputs: measured on lang-uk's benchmark, routing 181 such forms to the model
// *lowered* heteronym accuracy by 1.5 points against letting the parser decide.
// A failure here is not fatal — the caller falls back to the dictionary
// default, which is what happened before this tier existed at all.
func (r *HTTPContextResolver) ResolveMorphology(
	ctx context.Context, targets []MorphologyTarget,
) (map[int]MorphologyAnswer, error) {
	if len(targets) == 0 {
		return nil, nil
	}
	payload, err := json.Marshal(morphologyRequest{Targets: targets})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		r.url+"/internal/v1/morphology", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	client := r.morphologyClient
	if client == nil {
		client = r.client
	}
	response, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, response.Body)
		return nil, fmt.Errorf("morphology service returned status %d", response.StatusCode)
	}
	var decoded morphologyResponse
	if err := json.NewDecoder(io.LimitReader(response.Body, 2<<20)).Decode(&decoded); err != nil {
		return nil, fmt.Errorf("decode morphology response: %w", err)
	}
	answers := make(map[int]MorphologyAnswer, len(decoded.Decisions))
	for _, decision := range decoded.Decisions {
		if decision.Stressed != "" {
			answers[decision.Index] = MorphologyAnswer{
				Stressed: decision.Stressed, Rule: decision.Rule}
		}
	}
	return answers, nil
}

func (r *HTTPContextResolver) Resolve(ctx context.Context, targets []ModelTarget) ([]ModelDecision, error) {
	result := make([]ModelDecision, len(targets))
	eligible := make([]ModelTarget, 0, len(targets))
	indexes := make([]int, 0, len(targets))
	for i, target := range targets {
		result[i] = ModelDecision{Index: i, Status: "model_ineligible"}
		coverage, ok := r.manifest.Forms[target.Form]
		if !ok {
			continue
		}
		if !sameStrings(coverage.Signatures, target.Candidates) {
			result[i].Status = "candidate_mismatch"
			continue
		}
		indexes = append(indexes, i)
		eligible = append(eligible, target)
		result[i].Status = "ambiguous"
	}
	if len(eligible) == 0 {
		return result, nil
	}
	payload, err := json.Marshal(inferenceRequest{ModelVersion: r.manifest.ModelVersion,
		InventoryHash: r.manifest.InventoryHash, Targets: eligible})
	if err != nil {
		return result, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, r.url+"/internal/v1/resolve", bytes.NewReader(payload))
	if err != nil {
		return result, err
	}
	req.Header.Set("Content-Type", "application/json")
	response, err := r.client.Do(req)
	if err != nil {
		return result, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, response.Body)
		return result, fmt.Errorf("inference service returned status %d", response.StatusCode)
	}
	var decoded inferenceResponse
	if err := json.NewDecoder(io.LimitReader(response.Body, 2<<20)).Decode(&decoded); err != nil {
		return result, fmt.Errorf("decode inference response: %w", err)
	}
	if decoded.ModelVersion != r.manifest.ModelVersion || decoded.InventoryHash != r.manifest.InventoryHash {
		return result, errors.New("inference model or inventory version mismatch")
	}
	for _, decision := range decoded.Decisions {
		if decision.Index < 0 || decision.Index >= len(indexes) {
			return result, errors.New("inference returned an invalid index")
		}
		original := indexes[decision.Index]
		decision.Index = original
		if !sameStrings(scoreKeys(decision.Scores), targets[original].Candidates) ||
			!contains(targets[original].Candidates, decision.Signature) {
			decision.Status = "candidate_mismatch"
		} else if decision.Margin < r.manifest.Threshold {
			decision.Status = "low_margin"
		} else {
			decision.Status = "selected"
		}
		result[original] = decision
	}
	return result, nil
}

// TokenTarget is an ambiguous span that every earlier tier declined, offered
// to the token classifier before the dictionary default takes it.
type TokenTarget struct {
	Index      int      `json:"index"`
	Sentence   string   `json:"sentence"`
	Start      int      `json:"start"`
	End        int      `json:"end"`
	Form       string   `json:"form"`
	Candidates []string `json:"candidates"`
}

type tokenRequest struct {
	Targets []TokenTarget `json:"targets"`
}

// TokenAnswer is the classifier's reading for one span.
type TokenAnswer struct {
	Signature  string
	Confidence float64
}

type tokenDecision struct {
	Index      int     `json:"index"`
	Signature  string  `json:"signature"`
	Confidence float64 `json:"confidence"`
}

type tokenResponse struct {
	ModelVersion string          `json:"model_version"`
	Decisions    []tokenDecision `json:"decisions"`
}

// ResolveToken asks the token classifier about spans nothing else decided.
//
// The classifier reads the sentence and scores the lexicon's candidate
// readings for the span — no gloss, so it can take the grammatical heteronyms
// the cross-encoder cannot. It answers only for forms on its coverage list,
// which it holds itself: outside that list it is a coin flip where the
// dictionary default is 89% right, inside it 93.8% where the default is 78.5%
// (Common Voice, audio gold). Silence for a target means "not covered", and
// the caller lets the default stand. A failure is not fatal for the same
// reason morphology's is not: the default is what served before this tier.
func (r *HTTPContextResolver) ResolveToken(
	ctx context.Context, targets []TokenTarget,
) (map[int]TokenAnswer, error) {
	if len(targets) == 0 {
		return nil, nil
	}
	payload, err := json.Marshal(tokenRequest{Targets: targets})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		r.url+"/internal/v1/token", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	response, err := r.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, response.Body)
		return nil, fmt.Errorf("token service returned status %d", response.StatusCode)
	}
	var decoded tokenResponse
	if err := json.NewDecoder(io.LimitReader(response.Body, 2<<20)).Decode(&decoded); err != nil {
		return nil, fmt.Errorf("decode token response: %w", err)
	}
	answers := make(map[int]TokenAnswer, len(decoded.Decisions))
	for _, decision := range decoded.Decisions {
		if decision.Signature != "" {
			answers[decision.Index] = TokenAnswer{
				Signature: decision.Signature, Confidence: decision.Confidence}
		}
	}
	return answers, nil
}

// CombineTarget is one ambiguous token with the answer the tier order gave it.
type CombineTarget struct {
	Index      int      `json:"index"`
	Sentence   string   `json:"sentence"`
	Start      int      `json:"start"`
	End        int      `json:"end"`
	Form       string   `json:"form"`
	Text       string   `json:"text"`
	Candidates []string `json:"candidates"`
	Pipeline   string   `json:"pipeline"`
	Status     string   `json:"status"`
}

type combineRequest struct {
	Targets []CombineTarget `json:"targets"`
}

// CombineAnswer is the combiner's reading and whether it departs from the
// tier order's answer.
type CombineAnswer struct {
	Signature   string
	Probability float64
	Departed    bool
}

type combineDecision struct {
	Index       int     `json:"index"`
	Signature   string  `json:"signature"`
	Probability float64 `json:"probability"`
	Departed    bool    `json:"departed"`
}

type combineResponse struct {
	Decisions []combineDecision `json:"decisions"`
}

// Combine asks the learned combiner about every ambiguous token. It runs the
// cross-encoder, the classifier and the tagger itself, so it gets the same
// budget as the morphology call; a failure leaves the tier order's answers.
func (r *HTTPContextResolver) Combine(
	ctx context.Context, targets []CombineTarget,
) (map[int]CombineAnswer, error) {
	if len(targets) == 0 {
		return nil, nil
	}
	payload, err := json.Marshal(combineRequest{Targets: targets})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		r.url+"/internal/v1/combine", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	response, err := r.morphologyClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, response.Body)
		return nil, fmt.Errorf("combine service returned status %d", response.StatusCode)
	}
	var decoded combineResponse
	if err := json.NewDecoder(io.LimitReader(response.Body, 2<<20)).Decode(&decoded); err != nil {
		return nil, fmt.Errorf("decode combine response: %w", err)
	}
	answers := make(map[int]CombineAnswer, len(decoded.Decisions))
	for _, decision := range decoded.Decisions {
		if decision.Signature != "" {
			answers[decision.Index] = CombineAnswer{Signature: decision.Signature,
				Probability: decision.Probability, Departed: decision.Departed}
		}
	}
	return answers, nil
}
