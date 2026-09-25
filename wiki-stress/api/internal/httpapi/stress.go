package httpapi

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"

	"github.com/ukstress/ukstress/api/internal/normalize"
	"golang.org/x/text/unicode/norm"
)

const acute = '\u0301'

var ukrainianVowels = map[rune]bool{
	'а': true, 'е': true, 'є': true, 'и': true, 'і': true,
	'ї': true, 'о': true, 'у': true, 'ю': true, 'я': true,
}

type signatureRepository interface {
	BatchSignatures(context.Context, []string) (map[string][]string, int64, error)
}

type ModelTarget struct {
	Sentence   string   `json:"sentence"`
	Start      int      `json:"start"`
	End        int      `json:"end"`
	Form       string   `json:"form"`
	Candidates []string `json:"candidates"`
}

type ModelDecision struct {
	Index     int                `json:"index"`
	Signature string             `json:"signature,omitempty"`
	Scores    map[string]float64 `json:"scores,omitempty"`
	Margin    float64            `json:"margin"`
	Status    string             `json:"status"`
}

// ContextResolver must apply training-coverage, inventory and calibrated
// threshold checks. The handler independently revalidates its selection.
type ContextResolver interface {
	Resolve(context.Context, []ModelTarget) ([]ModelDecision, error)
	// ResolveMorphology answers ambiguities the model cannot take because the
	// form is outside the serving manifest, and whose readings differ by case
	// or number rather than by sense. Returns stressed surfaces by target index.
	ResolveMorphology(context.Context, []MorphologyTarget) (map[int]MorphologyAnswer, error)
	// ResolveToken answers what every earlier tier declined, from the sentence
	// alone, for forms on the classifier's own coverage list. Returns a
	// signature by target index; a missing index means the form is not covered.
	ResolveToken(context.Context, []TokenTarget) (map[int]TokenAnswer, error)
	// Combine weighs every tier's opinion of each ambiguous token, with the
	// tier order's own answer among them, and returns where it departs.
	Combine(context.Context, []CombineTarget) (map[int]CombineAnswer, error)
	Metadata() (version, manifestHash string)
}

// onAmbiguity names what to emit for a word the pipeline holds two readings
// for and cannot choose between: the model is outside its coverage, its margin
// is under the threshold, or the inference service is down.
//
// `onAmbiguityDefault` serves the lexicon's own first choice — highest
// confidence, then lowest source rank — and reports it as such.
// `onAmbiguityPreserve` emits the word unstressed, which is what this endpoint
// used to do unconditionally.
//
// Measured on the 1000-sentence gold homograph set, preserving scored 0.0000
// on that class by construction — 200 of 1000 rows returned no answer at all —
// against 0.4800 for the dictionary default, and the gap is far wider on
// running text, where the dictionary's first choice is usually also the
// frequent sense. Preserving is still the right behaviour for a caller that
// would rather see the ambiguity than a coin flip, which is why it stays
// available; it is no longer the silent default.
const (
	onAmbiguityDefault  = "default"
	onAmbiguityPreserve = "preserve"
)

type stressRequest struct {
	Text        string `json:"text"`
	OnAmbiguity string `json:"on_ambiguity"`
	// Combiner asks for the learned combiner on this request; absent, the
	// server's COMBINER_ENABLED decides. It is a choice for the caller
	// because it is right for one kind of text and wrong for another:
	// +1.6 on spoken Common Voice sentences, 1 right in 7 on book prose.
	Combiner *bool `json:"combiner,omitempty"`
}

type stressToken struct {
	Start      int                `json:"start"`
	End        int                `json:"end"`
	Text       string             `json:"text"`
	OutputText string             `json:"output_text"`
	Status     string             `json:"status"`
	Candidates []string           `json:"candidates,omitempty"`
	Scores     map[string]float64 `json:"scores,omitempty"`
	Margin     *float64           `json:"margin,omitempty"`
}

type stressResponse struct {
	Text                 string        `json:"text"`
	Tokens               []stressToken `json:"tokens"`
	DatasetID            int64         `json:"dataset_id"`
	ModelVersion         string        `json:"model_version,omitempty"`
	CoverageManifestHash string        `json:"coverage_manifest_hash,omitempty"`
	Warnings             []string      `json:"warnings"`
}

type tokenWork struct {
	stressToken
	normalized string
	byteStart  int
	byteEnd    int
}

// countsAsVowel reports whether a rune occupies a vowel ordinal.
//
// Ordinals must be counted exactly as `stress_signature` counts them, and that
// works on NFD: `й` decomposes to `и` plus a breve, and `и` is a vowel, so the
// generator gives `райо́н` the signature "2" — а, и, о. Counting the composed
// `й` as a consonant here shifted every ordinal after it, so "2" did not fit a
// two-vowel word and `район`, `району`, `війна`, `майдан` were rejected as
// invalid candidates and served unstressed. 58 tokens of `район` alone on
// lang-uk's benchmark.
//
// The rune is decomposed only to decide whether it counts; the original is
// still what gets emitted, so precomposed ї and й survive untouched.
func countsAsVowel(r rune) bool {
	if ukrainianVowels[unicode.ToLower(r)] {
		return true
	}
	for _, decomposed := range norm.NFD.String(string(r)) {
		return ukrainianVowels[unicode.ToLower(decomposed)]
	}
	return false
}

func isUkrainianLetter(r rune) bool {
	r = unicode.ToLower(r)
	return (r >= 'а' && r <= 'я') || r == 'є' || r == 'і' || r == 'ї' || r == 'ґ'
}

func isJoiner(r rune) bool {
	return strings.ContainsRune("-'ʼ’‘ʻ`＇‐‑‒–—−", r)
}

// isSegmentBreak reports whether a rune starts a new signature segment.
//
// This must match `stress_signature` in the Python normalizer, which splits on
// `[\s-]+` — hyphens and whitespace only. Reusing isJoiner here counted the
// apostrophe as a boundary too, so `м'ясо` was read as two tokens: signature
// "0" asked for the first vowel of segment 0, which is just `м` and has none.
// Every apostrophe-bearing form — 35,835 rows, 1.41% of the lexicon, including
// `м'ясо`, `здоров'я`, `сім'я`, `п'ять` — failed with "signature does not fit
// surface" and was served unstressed.
func isSegmentBreak(r rune) bool {
	return r == '-' || unicode.IsSpace(r) || strings.ContainsRune("‐‑‒–—−", r)
}

// wordTokens recognizes complete Ukrainian orthographic words. Joiners are
// retained only between letters, so surrounding punctuation remains outside.
func wordTokens(text string) []tokenWork {
	tokens := make([]tokenWork, 0)
	start := -1
	lastLetterEnd := -1
	for offset, r := range text {
		width := utf8.RuneLen(r)
		if isUkrainianLetter(r) || (start >= 0 && unicode.Is(unicode.Mn, r)) {
			if start < 0 {
				start = offset
			}
			if isUkrainianLetter(r) {
				lastLetterEnd = offset + width
			} else {
				// Combining marks (including an explicit acute and the marks in
				// NFD ї) are part of the orthographic token and its offsets.
				lastLetterEnd = offset + width
			}
			continue
		}
		if start >= 0 && isJoiner(r) {
			continue
		}
		if start >= 0 {
			raw := text[start:lastLetterEnd]
			tokens = append(tokens, tokenWork{stressToken: stressToken{
				Start: utf8.RuneCountInString(text[:start]),
				End:   utf8.RuneCountInString(text[:lastLetterEnd]), Text: raw, OutputText: raw,
			}, byteStart: start, byteEnd: lastLetterEnd})
			start, lastLetterEnd = -1, -1
		}
	}
	if start >= 0 {
		raw := text[start:lastLetterEnd]
		tokens = append(tokens, tokenWork{stressToken: stressToken{
			Start: utf8.RuneCountInString(text[:start]),
			End:   utf8.RuneCountInString(text[:lastLetterEnd]), Text: raw, OutputText: raw,
		}, byteStart: start, byteEnd: lastLetterEnd})
	}
	return tokens
}

func vowelCount(word string) int {
	count := 0
	for _, r := range normalize.LookupKey(word) {
		if ukrainianVowels[unicode.ToLower(r)] {
			count++
		}
	}
	return count
}

func applySignature(surface, signature string) (string, error) {
	if strings.ContainsRune(surface, acute) || signature == "" {
		return "", errors.New("invalid surface or empty signature")
	}
	// `0|1` on a single token is the wordlist's "either stress is acceptable"
	// notation, not two accents to emit. Rendering both produced `де+ржа+ва`
	// and `ді+вчи+на` — not words. A multi-token signature ("1:0|0:2") is
	// different: those parts each carry one accent for their own token, so the
	// split is only collapsed when no part names a segment.
	if strings.Contains(signature, "|") && !strings.Contains(signature, ":") {
		signature = strings.SplitN(signature, "|", 2)[0]
	}
	segments := strings.Split(signature, "|")
	wanted := make(map[[2]int]bool, len(segments))
	for _, part := range segments {
		segment, ordinal := 0, -1
		if strings.Contains(part, ":") {
			if _, err := fmt.Sscanf(part, "%d:%d", &segment, &ordinal); err != nil {
				return "", err
			}
		} else if _, err := fmt.Sscanf(part, "%d", &ordinal); err != nil {
			return "", err
		}
		if segment < 0 || ordinal < 0 {
			return "", errors.New("negative signature position")
		}
		wanted[[2]int{segment, ordinal}] = true
	}

	// Work on the caller's original rune sequence. Normalization is correct for
	// lookup keys but reconstruction must preserve every non-stress code point
	// (notably precomposed ї and the caller's apostrophe/hyphen style).
	runes := []rune(surface)
	var output []rune
	segment, ordinal, inserted := 0, -1, 0
	for i := 0; i < len(runes); i++ {
		r := runes[i]
		if isSegmentBreak(r) {
			segment++
			ordinal = -1
		}
		output = append(output, r)
		if !countsAsVowel(r) {
			continue
		}
		ordinal++
		for i+1 < len(runes) && unicode.Is(unicode.Mn, runes[i+1]) {
			i++
			output = append(output, runes[i])
		}
		if wanted[[2]int{segment, ordinal}] {
			output = append(output, acute)
			inserted++
		}
	}
	if inserted != len(wanted) {
		return "", errors.New("signature does not fit surface")
	}
	return string(output), nil
}

func contains(values []string, value string) bool {
	for _, candidate := range values {
		if candidate == value {
			return true
		}
	}
	return false
}

// applyCompoundFallback stresses hyphenated words the lexicon lacks as a whole.
//
// `адміністративно-територіальна` is not a lexicon entry and never will be —
// ad-hoc compounds are unbounded — but both halves are. Resolving from the
// parts covered 55 of the 89 hyphenated forms missing from lang-uk's benchmark
// vocabulary.
func applyCompoundFallback(work []tokenWork, signatures map[string][]string,
	stress func(string, string) (string, error)) {
	for i := range work {
		if work[i].Status != "not_found" || !strings.Contains(work[i].Text, "-") {
			continue
		}
		parts := strings.Split(work[i].Text, "-")
		out := make([]string, 0, len(parts))
		marked := false
		for _, part := range parts {
			candidates := signatures[normalize.LookupKey(part)]
			if len(candidates) == 0 {
				out = append(out, part)
				continue
			}
			stressed, err := stress(part, candidates[0])
			if err != nil {
				out = append(out, part)
				continue
			}
			out = append(out, stressed)
			marked = true
		}
		if marked {
			work[i].Status, work[i].OutputText = "compound", strings.Join(out, "-")
		}
	}
}

// applyDictionaryDefault fills in the words nothing decided.
//
// A token reaches here as "ambiguous" (the model declined, mismatched, or was
// unreachable) or "model_ineligible" (no resolver, or the form is outside the
// serving manifest). Both used to leave the word unstressed, which is the one
// answer that is certainly wrong; the lexicon's first candidate is at worst the
// wrong sense of a real word.
func applyDictionaryDefault(work []tokenWork) {
	for i := range work {
		if work[i].Status != "ambiguous" && work[i].Status != "model_ineligible" {
			continue
		}
		if len(work[i].Candidates) == 0 {
			continue
		}
		// Candidates arrive in the lexicon's preference order, so the first is
		// the same row the exact-lookup path would have served had the form not
		// been ambiguous.
		stressed, err := applySignature(work[i].Text, work[i].Candidates[0])
		if err != nil {
			work[i].Status = "invalid_candidate"
			continue
		}
		work[i].Status, work[i].OutputText = "dictionary_default", stressed
	}
}

func (s *Server) handleStress(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, s.maxBodyBytes)
	var req stressRequest
	decoder := json.NewDecoder(r.Body)
	if err := decoder.Decode(&req); err != nil || req.Text == "" {
		writeError(w, r, http.StatusBadRequest, "invalid_body", "text must be a non-empty string")
		return
	}
	switch req.OnAmbiguity {
	case "":
		req.OnAmbiguity = onAmbiguityDefault
	case onAmbiguityDefault, onAmbiguityPreserve:
	default:
		writeError(w, r, http.StatusBadRequest, "invalid_body",
			"on_ambiguity must be \"default\" or \"preserve\"")
		return
	}
	if len(req.Text) > int(s.maxBodyBytes) {
		writeError(w, r, http.StatusRequestEntityTooLarge, "text_too_large", "text exceeds the request limit")
		return
	}

	work := wordTokens(req.Text)
	if len(work) > s.maxBatchSize {
		writeError(w, r, http.StatusBadRequest, "too_many_tokens",
			fmt.Sprintf("text exceeds the maximum of %d Ukrainian words", s.maxBatchSize))
		return
	}
	unique := make(map[string]bool)
	forms := make([]string, 0)
	for i := range work {
		if strings.ContainsRune(work[i].Text, acute) {
			work[i].Status = "already_stressed"
			continue
		}
		// A single-vowel word is unambiguous, not unstressable. Ukrainian
		// convention omits the mark in writing, but a TTS front-end still
		// wants to know which vowel carries it, and the lexicon holds the
		// answer. Serving it costs nothing and matches what the evaluation
		// harness measures.
		if vowelCount(work[i].Text) == 0 {
			work[i].Status = "not_required"
			continue
		}
		work[i].normalized = normalize.LookupKey(work[i].Text)
		if !unique[work[i].normalized] {
			unique[work[i].normalized] = true
			forms = append(forms, work[i].normalized)
		}
		// An ad-hoc compound is not a lexicon entry, so ask for its parts too:
		// they are what the fallback stresses when the whole is missing.
		if strings.Contains(work[i].Text, "-") {
			for _, part := range strings.Split(work[i].normalized, "-") {
				if part != "" && !unique[part] {
					unique[part] = true
					forms = append(forms, part)
				}
			}
		}
	}
	sort.Strings(forms)
	signatures, proper, datasetID, err := s.repo.BatchSignaturesWithCase(r.Context(), forms)
	if err != nil {
		s.handleRepositoryError(w, r, err)
		return
	}

	targets := make([]ModelTarget, 0)
	targetIndexes := make([]int, 0)
	for i := range work {
		if work[i].Status != "" {
			continue
		}
		candidates := byCapitalisation(signatures[work[i].normalized], proper, work[i].normalized,
			work[i].Text, sentenceInitial(req.Text, work[i].byteStart))
		work[i].Candidates = candidates
		switch len(candidates) {
		case 0:
			work[i].Status = "not_found"
		case 1:
			stressed, applyErr := applySignature(work[i].Text, candidates[0])
			if applyErr != nil {
				work[i].Status = "invalid_candidate"
			} else {
				work[i].Status, work[i].OutputText = "stressed", stressed
			}
		default:
			work[i].Status = "model_ineligible"
			if s.resolver != nil {
				targetIndexes = append(targetIndexes, i)
				targets = append(targets, ModelTarget{Sentence: req.Text, Start: work[i].Start,
					End: work[i].End, Form: work[i].normalized, Candidates: candidates})
			}
		}
	}

	// The tagger is asked now, alongside the model, rather than after it: the
	// two questions are independent, and asked in turn they were the bulk of a
	// request's time. It is offered every ambiguous token; what it may decide
	// is filtered further down against the same target list as before, built
	// at the same point, so the answers served are the ones the sequential
	// order gave. The classifier stays after both: offered every ambiguous
	// token up front it did two to three times the work, which cost a third
	// of the throughput under concurrent load.
	var early earlyMorphology
	if s.resolver != nil {
		early.ask(r.Context(), s.resolver, morphologyTargets(work, req.Text))
	}

	warnings := make([]string, 0)
	modelVersion, manifestHash := "", ""
	if s.resolver != nil && len(targets) > 0 {
		modelVersion, manifestHash = s.resolver.Metadata()
		decisions, resolveErr := s.resolver.Resolve(r.Context(), targets)
		if resolveErr != nil {
			warnings = append(warnings, "contextual model unavailable; ambiguous words were preserved")
		}
		for _, decision := range decisions {
			if decision.Index < 0 || decision.Index >= len(targetIndexes) {
				warnings = append(warnings, "contextual model returned an invalid target index")
				continue
			}
			i := targetIndexes[decision.Index]
			work[i].Scores = decision.Scores
			if decision.Scores != nil {
				work[i].Margin = &decision.Margin
			}
			if decision.Status != "selected" {
				if decision.Status == "candidate_mismatch" {
					work[i].Status = "invalid_candidate"
				} else if decision.Status == "low_margin" || decision.Status == "ambiguous" {
					work[i].Status = "ambiguous"
				}
				continue
			}
			if !contains(work[i].Candidates, decision.Signature) {
				work[i].Status = "invalid_candidate"
				continue
			}
			stressed, applyErr := applySignature(work[i].Text, decision.Signature)
			if applyErr != nil {
				work[i].Status = "invalid_candidate"
			} else {
				work[i].Status, work[i].OutputText = "stressed", stressed
			}
		}
	}

	// Let the parser decide every ambiguity it can, including the ones the
	// model already answered.
	//
	// This tier used to be offered only tokens still marked "model_ineligible"
	// — the forms outside the serving manifest. That excluded two populations
	// silently: everything the model covered, and everything it declined with a
	// low margin, neither of which is a form a tagger has nothing to say about.
	// Measured on lang-uk, lifting the restriction moves morphology from 250
	// resolved tokens to 594 and is worth +8.29 points of heteronym accuracy
	// (72.96% -> 81.24%), +5.63 macro-F1 and +3.90 sentence accuracy; paired
	// per token it is net +98 (+125/-27), p < 0.00001, and the grammatical
	// class goes from 68.62% to 84.57%. Unambiguous words are untouched.
	//
	// Morphology wins over the model where both answer, which is the order the
	// measurement was taken in: a grammatical alternation is decided by the
	// tags, and the model is guessing at what the parse states outright.
	if s.resolver != nil {
		early.wait()
		allowed := make(map[int]bool)
		for _, t := range morphologyTargets(work, req.Text) {
			allowed[t.Index] = true
		}
		if answers, morphErr := early.answers, early.err; morphErr != nil {
			warnings = append(warnings, "morphology tier unavailable; dictionary defaults were used")
		} else {
			for index, answer := range answers {
				if index < 0 || index >= len(work) || !allowed[index] {
					continue
				}
				// A rule-derived answer overrides a confident model decision.
				// The counted form turns on the preceding numeral, and the
				// corpus the model learnt from labels these tokens the other
				// way — mining it for numeral phrases returns «три се́стри» —
				// so the model is confidently wrong exactly here. A plain tag
				// match still only fills what no tier decided.
				if work[index].Status == "stressed" && answer.Rule == "" {
					continue
				}
				work[index].Status = "morphology"
				if answer.Rule != "" {
					work[index].Status = answer.Rule
				}
				work[index].OutputText = answer.Stressed
			}
		}
	}

	// Before the fallbacks: a governed pronoun is decided by the preposition,
	// not by anything the tiers or the dictionary have to say.
	applyPrepositionalShift(work, req.Text, applySignature)

	applyCompoundFallback(work, signatures, applySignature)
	// Last: only whatever the lexicon and the compound fallback both missed.
	applySuffixFallback(work, s.suffixes, applySignature)
	applyPositionalDefault(work, applySignature)

	// The token classifier takes exactly what the dictionary default would
	// otherwise take, and nothing that any tier above decided. It is gated by
	// its own coverage list on the model side: measured on Common Voice against
	// audio gold, covered forms move from 78.5% to 93.8%, uncovered ones would
	// fall from 89% to 64%. The status is its own so the tier can be scored
	// apart from the default it replaces.
	if s.resolver != nil {
		tokenTargets := tokenTargets(work, req.Text)
		if answers, tokenErr := s.resolver.ResolveToken(r.Context(), tokenTargets); tokenErr != nil {
			warnings = append(warnings, "token tier unavailable; dictionary defaults were used")
		} else {
			for index, answer := range answers {
				if index < 0 || index >= len(work) || !contains(work[index].Candidates, answer.Signature) {
					continue
				}
				stressed, applyErr := applySignature(work[index].Text, answer.Signature)
				if applyErr != nil {
					continue
				}
				work[index].Status, work[index].OutputText = "token_model", stressed
			}
		}
	}

	if req.OnAmbiguity == onAmbiguityDefault {
		applyDictionaryDefault(work)
	}

	// The learned combiner, last: it sees the answer the tier order produced
	// and every tier's opinion, and overrules the answer only when it prefers
	// another reading by its trained margin. Held out, Common Voice moved from
	// 92.74% to 94.77% and lang-uk stayed at 83.96% (RESULTS.md). It was
	// trained on answers made with the dictionary default applied, so it runs
	// only in that mode; a failure keeps the tier order's answers.
	useCombiner := s.combine
	if req.Combiner != nil {
		useCombiner = *req.Combiner
	}
	if useCombiner && s.resolver != nil && req.OnAmbiguity == onAmbiguityDefault {
		if targets := combinerTargets(work, req.Text); len(targets) > 0 {
			answers, combineErr := s.resolver.Combine(r.Context(), targets)
			if combineErr != nil {
				warnings = append(warnings, "combiner unavailable; the tier order's answers were kept")
			}
			for index, answer := range answers {
				if !answer.Departed || index < 0 || index >= len(work) ||
					!contains(work[index].Candidates, answer.Signature) {
					continue
				}
				stressed, applyErr := applySignature(work[index].Text, answer.Signature)
				if applyErr != nil {
					continue
				}
				work[index].Status, work[index].OutputText = "combiner", stressed
			}
		}
	}

	var rebuilt strings.Builder
	position := 0
	tokens := make([]stressToken, len(work))
	for i := range work {
		rebuilt.WriteString(req.Text[position:work[i].byteStart])
		rebuilt.WriteString(work[i].OutputText)
		position = work[i].byteEnd
		tokens[i] = work[i].stressToken
	}
	rebuilt.WriteString(req.Text[position:])
	writeJSON(w, http.StatusOK, stressResponse{Text: rebuilt.String(), Tokens: tokens,
		DatasetID: datasetID, ModelVersion: modelVersion,
		CoverageManifestHash: manifestHash, Warnings: warnings})
}

// byCapitalisation narrows a candidate list that mixes a proper noun's
// readings with a common word's, by the one thing that tells them apart in
// running text: the capital letter. «у ру́сі» (рух) against «Ки́ївській Русі́»
// (Русь); «мені́» against the village «Ме́ні». A lowercase token cannot be the
// proper noun; a capitalised one that does not open a sentence is. A reading
// counts as a proper noun when every stored spelling of it is capitalised.
//
// The list is narrowed only when that leaves one reading: a partial
// narrowing would hand the model a candidate set its inventory does not know
// and the word would go out unstressed. Otherwise it is only reordered, so
// the dictionary default follows the capital letter too.
func byCapitalisation(candidates []string, proper map[string]bool, form, text string, initial bool) []string {
	if len(candidates) < 2 {
		return candidates
	}
	names := make([]string, 0, len(candidates))
	common := make([]string, 0, len(candidates))
	for _, c := range candidates {
		if proper[form+"\x00"+c] {
			names = append(names, c)
		} else {
			common = append(common, c)
		}
	}
	if len(names) == 0 || len(common) == 0 {
		return candidates
	}
	first, _ := utf8.DecodeRuneInString(text)
	switch {
	case !unicode.IsUpper(first):
		if len(common) == 1 {
			return common
		}
		return append(common, names...)
	case !initial:
		if len(names) == 1 {
			return names
		}
		return append(names, common...)
	}
	return candidates
}

// sentenceInitial reports whether the token at byteStart opens a sentence:
// nothing but spaces, quotes, brackets and dashes between it and the start of
// the text or a sentence-final mark. A capital there says nothing.
func sentenceInitial(text string, byteStart int) bool {
	before := []rune(text[:byteStart])
	for i := len(before) - 1; i >= 0; i-- {
		switch r := before[i]; {
		case unicode.IsSpace(r) || strings.ContainsRune("«»\"'„“”‘’()[]—–-", r):
			continue
		case strings.ContainsRune(".!?…:", r):
			return true
		default:
			return false
		}
	}
	return true
}

// combinerTargets offers every token with two or more single-ordinal
// candidates, whatever tier decided it, with the answer it got.
func combinerTargets(work []tokenWork, sentence string) []CombineTarget {
	targets := make([]CombineTarget, 0)
	for i := range work {
		digits := make([]string, 0, len(work[i].Candidates))
		seen := map[string]bool{}
		for _, c := range work[i].Candidates {
			if c != "" && strings.Trim(c, "0123456789") == "" && !seen[c] {
				seen[c] = true
				digits = append(digits, c)
			}
		}
		if len(digits) < 2 {
			continue
		}
		targets = append(targets, CombineTarget{
			Index: i, Sentence: sentence, Start: work[i].Start,
			End: work[i].End, Form: work[i].normalized, Text: work[i].Text,
			Candidates: work[i].Candidates, Pipeline: signatureOfOutput(work[i].OutputText),
			Status: work[i].Status,
		})
	}
	return targets
}

// signatureOfOutput is the vowel ordinal the acute follows in a stressed
// surface, counted as stress_signature counts it; "" when nothing is marked.
func signatureOfOutput(output string) string {
	ordinal := -1
	for _, r := range norm.NFD.String(output) {
		if r == acute {
			if ordinal < 0 {
				return ""
			}
			return fmt.Sprintf("%d", ordinal)
		}
		if ukrainianVowels[unicode.ToLower(r)] {
			ordinal++
		}
	}
	return ""
}

// tokenTargets picks the spans the classifier may answer: two or more
// readings, and either no tier's decision (the dictionary default is about to
// take them) or the cross-encoder's. Morphology, the prepositional rule and
// the reviewed lexicon are left alone.
//
// The cross-encoder's picks were added with tok-v3d. Measured on Common Voice
// against audio gold, the classifier trained on 118 audiobooks reads those
// spans at 94.6% where the cross-encoder reads them at 77.9%; on lang-uk the
// combined rule costs 0.4 points of heteronym accuracy, which is less than
// the dictionary-default-only rule cost with tok-v2. Both numbers are in
// RESULTS.md. A "stressed" token with one candidate is a plain lexicon hit
// and never reaches here.
func tokenTargets(work []tokenWork, text string) []TokenTarget {
	targets := make([]TokenTarget, 0)
	for i := range work {
		if len(work[i].Candidates) < 2 {
			continue
		}
		switch work[i].Status {
		case "ambiguous", "model_ineligible", "stressed":
		default:
			continue
		}
		targets = append(targets, TokenTarget{
			Index: i, Sentence: text, Start: work[i].Start, End: work[i].End,
			Form: work[i].normalized, Candidates: work[i].Candidates})
	}
	return targets
}

// earlyMorphology holds the tagger's answers, asked for in parallel with the
// model and read once the model's own have been applied.
type earlyMorphology struct {
	done    sync.WaitGroup
	answers map[int]MorphologyAnswer
	err     error
}

func (e *earlyMorphology) ask(ctx context.Context, resolver ContextResolver, targets []MorphologyTarget) {
	e.done.Add(1)
	go func() {
		defer e.done.Done()
		e.answers, e.err = resolver.ResolveMorphology(ctx, targets)
	}()
}

func (e *earlyMorphology) wait() { e.done.Wait() }

// morphologyTargets picks the tokens worth asking a parser about.
//
// Every token with more than one reading qualifies, whatever decided it first.
// A single-reading lexicon hit does not: its status is "stressed" too, but
// there is nothing to disambiguate and the parser must not get the chance to
// disagree. Statuses that mean the word is not a word we hold readings for --
// "not_found", "invalid_candidate" -- are left alone for the same reason.
func morphologyTargets(work []tokenWork, text string) []MorphologyTarget {
	targets := make([]MorphologyTarget, 0)
	for i := range work {
		if len(work[i].Candidates) < 2 {
			continue
		}
		switch work[i].Status {
		case "model_ineligible", "ambiguous", "stressed":
		default:
			continue
		}
		targets = append(targets, MorphologyTarget{
			Index: i, Sentence: text, Start: work[i].Start,
			End: work[i].End, Form: work[i].normalized})
	}
	return targets
}
