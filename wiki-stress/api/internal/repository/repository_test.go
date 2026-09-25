// Integration tests against a real PostgreSQL instance. Setup is shared
// with internal/httpapi's contract tests via internal/testdb; see that
// package for the skip condition and fixture shape.
package repository_test

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/ukstress/ukstress/api/internal/repository"
	"github.com/ukstress/ukstress/api/internal/testdb"
)

func newRepository(t *testing.T, ctx context.Context, apiURL string) *repository.Repository {
	t.Helper()
	repo, err := repository.Connect(ctx, apiURL, 5, 1, 5*time.Second, 2*time.Second, 50, "")
	if err != nil {
		t.Fatalf("connect repository: %v", err)
	}
	t.Cleanup(repo.Close)
	return repo
}

func setup(t *testing.T) (repo *repository.Repository, ownerURL string, datasetID int64) {
	t.Helper()
	ctx := context.Background()
	ownerURL = testdb.OwnerDatabaseURL(t)
	testdb.EnsureMigrated(t, ctx, ownerURL)
	datasetID = testdb.SeedDataset(t, ctx, ownerURL)
	repo = newRepository(t, ctx, testdb.APIDatabaseURL(t, ownerURL))
	return repo, ownerURL, datasetID
}

// TestHealthyReflectsConnectivityNotStaleCache is a regression test: found
// live against the real deployed API by stopping PostgreSQL and observing
// /health/ready still report "ready" from a cached dataset ID, even though
// the spec requires readiness to fail when PostgreSQL is unreachable.
// Healthy() must flip to false on a connection failure regardless of what
// ActiveDataset() last cached, and recover once PostgreSQL is reachable
// again.
func TestHealthyReflectsConnectivityNotStaleCache(t *testing.T) {
	repo, _, datasetID := setup(t)
	ctx := context.Background()

	if !repo.Healthy() {
		t.Fatal("Healthy() = false immediately after a successful Connect, want true")
	}
	if id, err := repo.ActiveDataset(); err != nil || id != datasetID {
		t.Fatalf("ActiveDataset() = (%d, %v), want (%d, nil)", id, err, datasetID)
	}

	canceledCtx, cancel := context.WithCancel(ctx)
	cancel()
	if err := repo.RefreshActiveDataset(canceledCtx); err == nil {
		t.Fatal("RefreshActiveDataset(canceled context) = nil, want a context error")
	}
	if repo.Healthy() {
		t.Error("Healthy() = true after a failed refresh, want false")
	}
	if id, err := repo.ActiveDataset(); err != nil || id != datasetID {
		t.Errorf("ActiveDataset() after failed refresh = (%d, %v), want the stale cached (%d, nil)",
			id, err, datasetID)
	}

	if err := repo.RefreshActiveDataset(ctx); err != nil {
		t.Fatalf("refresh after recovery: %v", err)
	}
	if !repo.Healthy() {
		t.Error("Healthy() = false after a successful refresh, want true")
	}
}

func TestLookupFoundAmbiguousAndBestMode(t *testing.T) {
	repo, _, datasetID := setup(t)
	ctx := context.Background()

	found, err := repo.Lookup(ctx, "мова", "мова", "exact")
	if err != nil {
		t.Fatalf("lookup found: %v", err)
	}
	if found.Status != "found" || len(found.Candidates) != 1 || found.Ambiguous {
		t.Errorf("lookup(мова) = %+v, want exactly one found, non-ambiguous candidate", found)
	}
	if found.DatasetID != datasetID {
		t.Errorf("lookup(мова).DatasetID = %d, want %d", found.DatasetID, datasetID)
	}

	ambiguous, err := repo.Lookup(ctx, "замок", "замок", "exact")
	if err != nil {
		t.Fatalf("lookup ambiguous: %v", err)
	}
	if ambiguous.Status != "ambiguous" || len(ambiguous.Candidates) != 2 || !ambiguous.Ambiguous {
		t.Errorf("lookup(замок) = %+v, want two ambiguous candidates", ambiguous)
	}
	// Deterministic ordering: lower source_rank first.
	if ambiguous.Candidates[0].StressedForm != "за́мок" || ambiguous.Candidates[1].StressedForm != "замо́к" {
		t.Errorf("lookup(замок) candidate order = %v, want [за́мок, замо́к]", ambiguous.Candidates)
	}

	best, err := repo.Lookup(ctx, "замок", "замок", "best")
	if err != nil {
		t.Fatalf("lookup best: %v", err)
	}
	if len(best.Candidates) != 1 || best.Candidates[0].StressedForm != "за́мок" {
		t.Errorf("lookup(замок, best) = %+v, want single top-ranked candidate", best)
	}
	if !best.Ambiguous {
		t.Error("lookup(замок, best).Ambiguous = false, want true: best mode must not hide that alternatives exist")
	}

	notFound, err := repo.Lookup(ctx, "нема", "нема", "exact")
	if err != nil {
		t.Fatalf("lookup not found: %v", err)
	}
	if notFound.Status != "not_found" || len(notFound.Candidates) != 0 {
		t.Errorf("lookup(нема) = %+v, want not_found with no candidates", notFound)
	}
}

func TestBatchLookupPreservesOrderAndDuplicates(t *testing.T) {
	repo, _, _ := setup(t)
	ctx := context.Background()

	inputs := []string{"мова", "замок", "мова", "нема"}
	results, err := repo.BatchLookup(ctx, inputs, inputs)
	if err != nil {
		t.Fatalf("batch lookup: %v", err)
	}
	if len(results) != len(inputs) {
		t.Fatalf("batch lookup returned %d results, want %d", len(results), len(inputs))
	}
	for i, input := range inputs {
		if results[i].Input != input {
			t.Errorf("results[%d].Input = %q, want %q (order not preserved)", i, results[i].Input, input)
		}
	}
	if results[0].Status != "found" || results[2].Status != "found" {
		t.Errorf("duplicate word мова: results[0]=%+v results[2]=%+v, both want found", results[0], results[2])
	}
	if results[1].Status != "ambiguous" || len(results[1].Candidates) != 2 {
		t.Errorf("results[1] (замок) = %+v, want ambiguous with 2 candidates", results[1])
	}
	if results[3].Status != "not_found" {
		t.Errorf("results[3] (нема) = %+v, want not_found", results[3])
	}
}

func TestLemmaFormsPagination(t *testing.T) {
	repo, _, _ := setup(t)
	ctx := context.Background()

	all, err := repo.LemmaForms(ctx, "замок", 0, 50)
	if err != nil {
		t.Fatalf("lemma forms: %v", err)
	}
	if len(all) != 2 {
		t.Fatalf("lemma forms for замок = %d rows, want 2", len(all))
	}

	page1, err := repo.LemmaForms(ctx, "замок", 0, 1)
	if err != nil {
		t.Fatalf("lemma forms page 1: %v", err)
	}
	page2, err := repo.LemmaForms(ctx, "замок", 1, 1)
	if err != nil {
		t.Fatalf("lemma forms page 2: %v", err)
	}
	if len(page1) != 1 || len(page2) != 1 {
		t.Fatalf("paginated pages = %d, %d rows, want 1 and 1", len(page1), len(page2))
	}
	if page1[0].StressedForm == page2[0].StressedForm {
		t.Errorf("page1 and page2 returned the same row: %+v", page1[0])
	}
	if page1[0].StressedForm != all[0].StressedForm || page2[0].StressedForm != all[1].StressedForm {
		t.Errorf("paginated rows do not match the unpaginated order: page1=%+v page2=%+v all=%+v",
			page1[0], page2[0], all)
	}
}

// TestActiveDatasetSwitchWithoutRestart proves a published dataset switch
// is picked up by RefreshActiveDataset without recreating the Repository,
// and that a request completes entirely against one dataset ID.
func TestActiveDatasetSwitchWithoutRestart(t *testing.T) {
	repo, ownerURL, firstDatasetID := setup(t)
	ctx := context.Background()

	before, err := repo.Lookup(ctx, "мова", "мова", "exact")
	if err != nil {
		t.Fatalf("lookup before switch: %v", err)
	}
	if before.DatasetID != firstDatasetID {
		t.Fatalf("lookup before switch used dataset %d, want %d", before.DatasetID, firstDatasetID)
	}

	secondDatasetID := testdb.SeedDataset(t, ctx, ownerURL)
	if secondDatasetID == firstDatasetID {
		t.Fatal("SeedDataset returned the same dataset ID twice")
	}

	if err := repo.RefreshActiveDataset(ctx); err != nil {
		t.Fatalf("refresh active dataset: %v", err)
	}

	after, err := repo.Lookup(ctx, "мова", "мова", "exact")
	if err != nil {
		t.Fatalf("lookup after switch: %v", err)
	}
	if after.DatasetID != secondDatasetID {
		t.Errorf("lookup after switch used dataset %d, want %d", after.DatasetID, secondDatasetID)
	}
}

// TestConcurrentQueriesDuringDatasetSwitch runs lookups continuously on
// background goroutines while the active dataset is republished and
// refreshed, proving every completed response is internally consistent:
// its candidates and its reported dataset_id always came from the same
// dataset, even mid-switch.
func TestConcurrentQueriesDuringDatasetSwitch(t *testing.T) {
	repo, ownerURL, firstDatasetID := setup(t)
	ctx := context.Background()
	secondDatasetID := testdb.SeedDataset(t, ctx, ownerURL)

	stop := make(chan struct{})
	errCh := make(chan error, 8)
	var wg sync.WaitGroup

	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				result, err := repo.Lookup(ctx, "мова", "мова", "exact")
				if err != nil {
					errCh <- fmt.Errorf("concurrent lookup: %w", err)
					return
				}
				if result.DatasetID != firstDatasetID && result.DatasetID != secondDatasetID {
					errCh <- fmt.Errorf("lookup returned unexpected dataset_id %d", result.DatasetID)
					return
				}
				if result.Status != "found" || len(result.Candidates) != 1 {
					errCh <- fmt.Errorf("lookup returned inconsistent result mid-switch: %+v", result)
					return
				}
			}
		}()
	}

	for i := 0; i < 20; i++ {
		target := firstDatasetID
		if i%2 == 1 {
			target = secondDatasetID
		}
		testdb.SwitchActiveDataset(t, ctx, ownerURL, target)
		if err := repo.RefreshActiveDataset(ctx); err != nil {
			t.Fatalf("refresh active dataset: %v", err)
		}
		time.Sleep(5 * time.Millisecond)
	}

	close(stop)
	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Error(err)
	}
}
