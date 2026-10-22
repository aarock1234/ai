package ai

import (
	"context"
	"sync"
)

// mapConcurrently calls work for every item with at most limit calls running
// at once, and returns the results in input order. A limit of zero or less
// means no limit. The first failure cancels the context passed to the other
// calls; mapConcurrently waits for every started call and returns that first
// failure.
func mapConcurrently[Item, Result any](
	ctx context.Context,
	items []Item,
	limit int,
	work func(ctx context.Context, item Item) (Result, error),
) ([]Result, error) {
	if limit <= 0 || limit > len(items) {
		limit = len(items)
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	var (
		firstFailure error
		failureOnce  sync.Once
		group        sync.WaitGroup
	)

	fail := func(err error) {
		failureOnce.Do(func() {
			firstFailure = err
			cancel()
		})
	}

	results := make([]Result, len(items))
	slots := make(chan struct{}, max(limit, 1))

	for index, item := range items {
		select {
		case slots <- struct{}{}:
		case <-ctx.Done():
		}

		if ctx.Err() != nil {
			fail(context.Cause(ctx))

			break
		}

		group.Go(func() {
			defer func() {
				<-slots
			}()

			result, err := work(ctx, item)
			if err != nil {
				fail(err)

				return
			}

			results[index] = result
		})
	}

	group.Wait()

	if firstFailure != nil {
		return nil, firstFailure
	}

	return results, nil
}
