package repository

import (
	"reflect"
	"testing"
)

func TestDropSubsumedFreeVariation(t *testing.T) {
	for _, testCase := range []struct {
		name string
		in   []string
		want []string
	}{{
		name: "free variation subsumed by a specific member",
		// `держави`: the trie-corrections dataset stores `де́ржа́ви` as `0|1`
		// while the curated wordlist stores `держа́ви` as `1`. These are one
		// reading, and the specific one is the answer to serve.
		in:   []string{"0|1", "1"},
		want: []string{"1"},
	}, {
		name: "free variation kept when no member is present",
		in:   []string{"0|1", "2"},
		want: []string{"0|1", "2"},
	}, {
		name: "genuine ambiguity untouched",
		in:   []string{"0", "1"},
		want: []string{"0", "1"},
	}, {
		name: "free variation alone survives",
		in:   []string{"0|1"},
		want: []string{"0|1"},
	}, {
		name: "multi-token signatures compare whole, not per segment",
		in:   []string{"1:0", "1:0|0:2"},
		want: []string{"1:0"},
	}} {
		t.Run(testCase.name, func(t *testing.T) {
			got := dropSubsumedFreeVariation(append([]string(nil), testCase.in...))
			if !reflect.DeepEqual(got, testCase.want) {
				t.Fatalf("got %q, want %q", got, testCase.want)
			}
		})
	}
}
