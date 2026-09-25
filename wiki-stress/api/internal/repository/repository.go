// Package repository implements read-only PostgreSQL lookup operations.
package repository

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

var ErrNoActiveDataset = errors.New("no active dataset")

const exactSQL = `
SELECT stressed_form, stress_signature, lemma_normalized, stressed_lemma,
       part_of_speech, grammatical_tags, is_lemma, is_variant, is_obsolete,
       confidence, source_rank
FROM stress_lookup
WHERE dataset_id = $1 AND form_normalized = $2
ORDER BY is_obsolete, confidence DESC, source_rank, is_lemma DESC, stressed_form
LIMIT $3`

type Candidate struct {
	StressedForm    string   `json:"stressed_form"`
	StressSignature string   `json:"stress_signature"`
	Lemma           string   `json:"lemma"`
	StressedLemma   *string  `json:"stressed_lemma,omitempty"`
	PartOfSpeech    *string  `json:"part_of_speech,omitempty"`
	GrammaticalTags []string `json:"grammatical_tags"`
	IsLemma         bool     `json:"is_lemma"`
	IsVariant       bool     `json:"is_variant"`
	IsObsolete      bool     `json:"is_obsolete"`
	Confidence      float32  `json:"confidence"`
	SourceRank      int16    `json:"-"`
}

type LookupResult struct {
	Input      string      `json:"input"`
	Status     string      `json:"status"`
	Candidates []Candidate `json:"candidates"`
	Ambiguous  bool        `json:"ambiguous"`
	Truncated  bool        `json:"truncated"`
	DatasetID  int64       `json:"dataset_id"`
}

type LemmaForm struct {
	Form            string   `json:"form"`
	StressedForm    string   `json:"stressed_form"`
	GrammaticalTags []string `json:"grammatical_tags"`
	Confidence      float32  `json:"confidence"`
}

// SignatureSet is the serving projection used by full-text stressing.  It is
// deliberately smaller than Candidate: runtime disambiguation is constrained
// to stress signatures already present in the active database.
type SignatureSet struct {
	Form       string   `json:"form"`
	Signatures []string `json:"signatures"`
}

type Repository struct {
	pool          *pgxpool.Pool
	active        atomic.Int64
	healthy       atomic.Bool
	maxResult     int
	inventoryHash string
	// supplementary datasets are read alongside the active one; see
	// config.Config.SupplementaryDatasets for why the lexicon is layered.
	supplementary []int64
	// reviewed names datasets whose rows record a human decision rather
	// than a derivation. The trie default is a good prior — on lang-uk it
	// beats the lexicon order 26 to 12 — but it is still a prior, and a
	// reviewed reading must not lose to it.
	reviewed []int64
	// exclusive names reviewed datasets whose reading is the *only* one, not
	// merely the leading one. A form here is served with a single candidate, so
	// no contextual tier is offered it — which is the point: the cross-encoder
	// answered `маю` with the reading a person had already ruled out.
	exclusive []int64
	// trieDefaults picks which reading leads when the lexicon is ambiguous and
	// nothing else decides. Nil leaves the query's own order in place.
	trieDefaults TrieDefaults
}

func Connect(ctx context.Context, databaseURL string, maxConnections, minConnections int32,
	connectTimeout, statementTimeout time.Duration, maxResults int, inventoryHash string,
) (*Repository, error) {
	cfg, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		return nil, fmt.Errorf("parse database configuration: %w", err)
	}
	cfg.MaxConns = maxConnections
	cfg.MinConns = minConnections
	cfg.ConnConfig.ConnectTimeout = connectTimeout
	cfg.AfterConnect = func(ctx context.Context, conn *pgx.Conn) error {
		if _, err := conn.Exec(ctx, "SET default_transaction_read_only = on"); err != nil {
			return err
		}
		if _, err := conn.Exec(
			ctx,
			"SELECT set_config('statement_timeout', $1, false)",
			statementTimeout.String(),
		); err != nil {
			return err
		}
		_, err := conn.Prepare(ctx, "lookup_exact", exactSQL)
		return err
	}
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("create database pool: %w", err)
	}
	repository := &Repository{pool: pool, maxResult: maxResults, inventoryHash: inventoryHash}
	if err := repository.RefreshActiveDataset(ctx); err != nil && !errors.Is(err, ErrNoActiveDataset) {
		pool.Close()
		return nil, err
	}
	return repository, nil
}

func (r *Repository) Close() { r.pool.Close() }

func (r *Repository) PoolStats() *pgxpool.Stat { return r.pool.Stat() }

func (r *Repository) ActiveDataset() (int64, error) {
	id := r.active.Load()
	if id == 0 {
		return 0, ErrNoActiveDataset
	}
	return id, nil
}

// Healthy reports whether the most recent RefreshActiveDataset call could
// actually reach PostgreSQL. It is intentionally independent of
// ActiveDataset: a connection failure must not be masked by a stale
// cached dataset ID from before the database became unreachable.
func (r *Repository) Healthy() bool {
	return r.healthy.Load()
}

func (r *Repository) RefreshActiveDataset(ctx context.Context) error {
	var id int64
	err := r.pool.QueryRow(ctx,
		"SELECT dataset_id FROM active_dataset WHERE singleton").Scan(&id)
	if errors.Is(err, pgx.ErrNoRows) {
		r.active.Store(0)
		r.healthy.Store(true)
		return ErrNoActiveDataset
	}
	if err != nil {
		r.healthy.Store(false)
		return fmt.Errorf("load active dataset: %w", err)
	}
	r.active.Store(id)
	r.healthy.Store(true)
	return nil
}

func (r *Repository) RunRefresh(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			_ = r.RefreshActiveDataset(ctx)
		}
	}
}

func scanCandidate(rows pgx.Rows) (Candidate, error) {
	var candidate Candidate
	err := rows.Scan(
		&candidate.StressedForm,
		&candidate.StressSignature,
		&candidate.Lemma,
		&candidate.StressedLemma,
		&candidate.PartOfSpeech,
		&candidate.GrammaticalTags,
		&candidate.IsLemma,
		&candidate.IsVariant,
		&candidate.IsObsolete,
		&candidate.Confidence,
		&candidate.SourceRank,
	)
	return candidate, err
}

func (r *Repository) Lookup(ctx context.Context, input, normalized, mode string) (LookupResult, error) {
	datasetID, err := r.ActiveDataset()
	if err != nil {
		return LookupResult{}, err
	}
	tx, err := r.pool.BeginTx(ctx, pgx.TxOptions{AccessMode: pgx.ReadOnly})
	if err != nil {
		return LookupResult{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	rows, err := tx.Query(ctx, "lookup_exact", datasetID, normalized, r.maxResult+1)
	if err != nil {
		return LookupResult{}, err
	}
	defer rows.Close()
	candidates := make([]Candidate, 0)
	for rows.Next() {
		candidate, scanErr := scanCandidate(rows)
		if scanErr != nil {
			return LookupResult{}, scanErr
		}
		candidates = append(candidates, candidate)
	}
	if err := rows.Err(); err != nil {
		return LookupResult{}, err
	}
	if err := tx.Commit(ctx); err != nil {
		return LookupResult{}, err
	}
	truncated := len(candidates) > r.maxResult
	if truncated {
		candidates = candidates[:r.maxResult]
	}
	ambiguous := isAmbiguous(candidates, truncated)
	status := "found"
	if len(candidates) == 0 {
		status = "not_found"
	} else if ambiguous {
		status = "ambiguous"
	}
	if mode == "best" && len(candidates) > 1 {
		candidates = candidates[:1]
	}
	return LookupResult{
		Input: input, Status: status, Candidates: candidates, Ambiguous: ambiguous,
		Truncated: truncated, DatasetID: datasetID,
	}, nil
}

// isAmbiguous reports whether the candidates disagree about where the stress
// falls.
//
// The question this service answers is "where is the stress", so ambiguity is a
// property of the stress signatures, not of the row count. One spelling can
// carry many lexemes that all stress it identically — "а" has 33 rows in the
// published dataset, every one of them "а́" — and counting rows called each of
// those ambiguous, sending a word with exactly one possible pronunciation to
// the contextual model or to abstention. Measured on `merged-wordlist-v1`,
// 89,000 forms (3.8% of the lexicon) were affected, 87,307 of them with
// byte-identical spellings in every row.
//
// A truncated result stays ambiguous: the signatures beyond the limit are
// unknown, so agreement cannot be claimed.
func isAmbiguous(candidates []Candidate, truncated bool) bool {
	if truncated {
		return true
	}
	for _, candidate := range candidates[min(1, len(candidates)):] {
		if candidate.StressSignature != candidates[0].StressSignature {
			return true
		}
	}
	return false
}

func (r *Repository) BatchLookup(ctx context.Context, inputs, normalized []string) ([]LookupResult, error) {
	datasetID, err := r.ActiveDataset()
	if err != nil {
		return nil, err
	}
	rows, err := r.pool.Query(ctx, `
WITH input AS (
    SELECT ordinality, word
    FROM unnest($1::text[]) WITH ORDINALITY AS value(word, ordinality)
), ranked AS (
    SELECT input.ordinality, lookup.stressed_form, lookup.stress_signature,
           lookup.lemma_normalized, lookup.stressed_lemma, lookup.part_of_speech,
           lookup.grammatical_tags, lookup.is_lemma, lookup.is_variant,
           lookup.is_obsolete, lookup.confidence, lookup.source_rank,
           row_number() OVER (
               PARTITION BY input.ordinality
               ORDER BY lookup.is_obsolete, lookup.confidence DESC,
                        lookup.source_rank, lookup.is_lemma DESC, lookup.stressed_form
           ) AS candidate_number,
           -- Distinct signatures, not rows: see isAmbiguous. Counting rows made
           -- every spelling shared by several identically-stressed lexemes look
           -- ambiguous. PostgreSQL rejects count(DISTINCT ...) OVER (...), so
           -- the distinct count is max(dense_rank()) over the signature order,
           -- which needs a second CTE because window functions cannot nest.
           dense_rank() OVER (
               PARTITION BY input.ordinality ORDER BY lookup.stress_signature
           ) AS signature_rank
    FROM input
    LEFT JOIN stress_lookup lookup
      ON lookup.dataset_id = ANY($2) AND lookup.form_normalized = input.word
), counted AS (
    SELECT ranked.*,
           max(signature_rank) OVER (PARTITION BY ordinality) AS candidate_count
    FROM ranked
)
SELECT ordinality, stressed_form, stress_signature, lemma_normalized,
       stressed_lemma, part_of_speech, grammatical_tags, is_lemma, is_variant,
       is_obsolete, confidence, source_rank, candidate_count
FROM counted
WHERE stressed_form IS NOT NULL AND candidate_number <= $3
ORDER BY ordinality, candidate_number`, normalized, datasetID, r.maxResult)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	results := make([]LookupResult, len(inputs))
	for index, input := range inputs {
		results[index] = LookupResult{Input: input, Status: "not_found", DatasetID: datasetID}
	}
	for rows.Next() {
		var ordinal int
		var count int
		var candidate Candidate
		err := rows.Scan(
			&ordinal, &candidate.StressedForm, &candidate.StressSignature, &candidate.Lemma,
			&candidate.StressedLemma, &candidate.PartOfSpeech, &candidate.GrammaticalTags,
			&candidate.IsLemma, &candidate.IsVariant, &candidate.IsObsolete,
			&candidate.Confidence, &candidate.SourceRank, &count,
		)
		if err != nil {
			return nil, err
		}
		result := &results[ordinal-1]
		result.Candidates = append(result.Candidates, candidate)
		result.Ambiguous = count > 1
		result.Truncated = count > r.maxResult
		if result.Ambiguous {
			result.Status = "ambiguous"
		} else {
			result.Status = "found"
		}
	}
	return results, rows.Err()
}

// SetSupplementaryDatasets records dataset ids read alongside the active one.
func (r *Repository) SetSupplementaryDatasets(ids []int64) {
	r.supplementary = ids
}

// SetReviewedDatasets records dataset ids holding human decisions, whose
// reading leads regardless of what the trie prefers.
func (r *Repository) SetReviewedDatasets(ids []int64) {
	r.reviewed = ids
}

// SetExclusiveDatasets records reviewed datasets whose reading is the only
// admissible one. Ordering is not enough for these: a form with two candidates
// reaches the contextual model, and the model can choose the reading the
// review rejected.
func (r *Repository) SetExclusiveDatasets(ids []int64) {
	r.exclusive = ids
}

// datasetIDs returns the active dataset followed by any supplementary ones.
func (r *Repository) datasetIDs(active int64) []int64 {
	if len(r.supplementary) == 0 {
		return []int64{active}
	}
	return append([]int64{active}, r.supplementary...)
}

// BatchSignatures resolves all distinct forms in one set-oriented query and
// collapses duplicate provenance rows by stress signature.
//
// Signatures come back in the lexicon's own preference order — highest
// confidence first, then lowest source_rank — so the caller can serve
// candidates[0] as "what the dictionary says" without a second query. Rows
// contributed by the contextual candidate table carry no confidence of their
// own and sort last: they exist to widen what the model may choose between,
// not to outrank a curated stress.
func (r *Repository) BatchSignatures(ctx context.Context, forms []string) (map[string][]string, int64, error) {
	result, _, datasetID, err := r.batchSignatures(ctx, forms)
	return result, datasetID, err
}

// BatchSignaturesWithCase also reports, per (form, signature), whether the
// reading is a proper noun a person has confirmed: every stored spelling is
// capitalised and a reviewed dataset holds it.
//
// Capitalisation alone is not evidence. The lexicon stores common readings
// capitalised often enough (a wordlist of sentence-initial words) that
// narrowing by it served «бо́ку» as «боку́» 62 times on the top-200 set and
// cost 0.8 points there; only a reviewed decision, such as «Русі́» against
// the «ру́сі» of «рух», may say which reading the capital letter selects.
//
// The merged wordlist injects toponyms and given names at the highest
// confidence, so `Розді́л` (a village) outranks the common noun `ро́зділ` and
// `Ме́ні` displaces `мені́`. For a lowercase token in running text a proper
// noun is the wrong answer whatever its confidence, so the caller demotes it.
func (r *Repository) BatchSignaturesWithCase(
	ctx context.Context, forms []string,
) (map[string][]string, map[string]bool, int64, error) {
	return r.batchSignatures(ctx, forms)
}

func (r *Repository) batchSignatures(
	ctx context.Context, forms []string,
) (map[string][]string, map[string]bool, int64, error) {
	datasetID, err := r.ActiveDataset()
	if err != nil {
		return nil, nil, 0, err
	}
	rows, err := r.pool.Query(ctx, `
WITH input AS (
    SELECT word FROM unnest($1::text[]) AS value(word)
), candidates AS (
    SELECT input.word, lookup.stress_signature,
           lookup.confidence, lookup.source_rank, lookup.stressed_form,
           lookup.dataset_id = ANY($4) AS reviewed,
           lookup.dataset_id = ANY($5) AS exclusive
    FROM input
    JOIN stress_lookup lookup
      ON lookup.dataset_id = ANY($2) AND lookup.form_normalized = input.word
    UNION ALL
    SELECT input.word, contextual.stress_signature, 0::real, 2147483647,
           contextual.stressed_form, false, false
    FROM input
    JOIN contextual_stress_candidate contextual
      ON contextual.inventory_hash = $3
     AND contextual.form_normalized = input.word
)
SELECT word, stress_signature,
       bool_and(stressed_form ~ '^[[:upper:]]') AND bool_or(reviewed) AS proper,
       bool_or(reviewed) AS reviewed,
       bool_or(exclusive) AS exclusive
FROM candidates
GROUP BY word, stress_signature
ORDER BY word, bool_and(stressed_form ~ '^[[:upper:]]'),
         max(confidence) DESC, min(source_rank), stress_signature`,
		forms, r.datasetIDs(datasetID), r.inventoryHash, r.reviewed, r.exclusive)
	if err != nil {
		return nil, nil, 0, err
	}
	defer rows.Close()
	result := make(map[string][]string, len(forms))
	proper := make(map[string]bool)
	// The reading a person decided on, where there is one. Only the first is
	// kept: a form should not carry two reviewed decisions, and if it somehow
	// does, the query's own order picks between them rather than the last row
	// read winning by accident.
	decided := make(map[string]string)
	only := make(map[string]string)
	for rows.Next() {
		var form, signature string
		var isProper, isReviewed, isExclusive bool
		if err := rows.Scan(&form, &signature, &isProper, &isReviewed, &isExclusive); err != nil {
			return nil, nil, 0, err
		}
		result[form] = append(result[form], signature)
		proper[form+"\x00"+signature] = isProper
		if isReviewed {
			if _, seen := decided[form]; !seen {
				decided[form] = signature
			}
		}
		if isExclusive {
			if _, seen := only[form]; !seen {
				only[form] = signature
			}
		}
	}
	if err := rows.Err(); err != nil {
		return nil, nil, 0, err
	}
	for form, signatures := range result {
		signatures = dropSubsumedFreeVariation(signatures)
		// A reviewed reading leads. The trie default is the better guess where
		// nobody has looked — 26 right against the lexicon order's 12 on
		// lang-uk — but it is a guess, and it was overruling decisions made
		// deliberately: `ма́ю` and `того́` were corrected in the lexicon at
		// full confidence and still served as `маю́` and `то́го`.
		// An exclusive reading is the only candidate. Leading the list is not
		// enough — two candidates reach the contextual model, and on `маю` and
		// `того` the model chose the reading a person had ruled out.
		if wanted, ok := only[form]; ok {
			result[form] = []string{wanted}
			continue
		}
		if wanted, ok := decided[form]; ok {
			result[form] = preferTrieDefault(signatures, wanted)
			continue
		}
		result[form] = preferTrieDefault(signatures, r.trieDefaults[form])
	}
	return result, proper, datasetID, nil
}

// dropSubsumedFreeVariation removes a "either stress is acceptable" signature
// when a specific member of it is also present.
//
// `0|1` is one reading, not a rival of `1`. The trie-corrections dataset stores
// the free-variation spelling while the curated wordlist stores the specific
// one, so `держави` arrived with both and looked ambiguous. The free-variation
// row carries the higher confidence, so it sorted first and was then collapsed
// to its first member for output -- serving `де́ржави` while the curated
// `держа́ви` sat right behind it in the same list. It contributes no
// information the specific reading lacks, and which member its collapse picks
// is arbitrary.
func dropSubsumedFreeVariation(signatures []string) []string {
	specific := make(map[string]bool, len(signatures))
	for _, signature := range signatures {
		if !strings.Contains(signature, "|") {
			specific[signature] = true
		}
	}
	if len(specific) == 0 {
		return signatures
	}
	kept := signatures[:0]
	for _, signature := range signatures {
		subsumed := false
		if strings.Contains(signature, "|") {
			for _, member := range strings.Split(signature, "|") {
				if specific[member] {
					subsumed = true
					break
				}
			}
		}
		if !subsumed {
			kept = append(kept, signature)
		}
	}
	return kept
}

func (r *Repository) LemmaForms(ctx context.Context, lemma string, offset, limit int) ([]LemmaForm, error) {
	datasetID, err := r.ActiveDataset()
	if err != nil {
		return nil, err
	}
	rows, err := r.pool.Query(ctx, `
SELECT form_normalized, stressed_form, grammatical_tags, confidence
FROM stress_lookup
WHERE dataset_id = $1 AND lemma_normalized = $2
ORDER BY form_normalized, is_obsolete, confidence DESC, source_rank, stressed_form
OFFSET $3 LIMIT $4`, datasetID, lemma, offset, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	forms := make([]LemmaForm, 0)
	for rows.Next() {
		var form LemmaForm
		if err := rows.Scan(&form.Form, &form.StressedForm, &form.GrammaticalTags, &form.Confidence); err != nil {
			return nil, err
		}
		forms = append(forms, form)
	}
	return forms, rows.Err()
}
