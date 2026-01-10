# go-rust-bert

Golang bindings for [rust-bert](https://github.com/guillaume-be/rust-bert), bringing state-of-the-art NLP models to Go.

## Features

- **Sentiment Analysis**: Ready-to-use pipeline for sentiment classification.
- **Self-Contained**: Bundles necessary `libtorch` and Rust dynamic libraries, extracting them at runtime. No manual library installation required on the target machine (after build).
- **Easy Integration**: Pure Go API wrapper.

## Prerequisites

- **macOS** (currently tested on Apple Silicon/ARM64)
- **Rust** (stable toolchain)
- **Go** 1.22+

## Build

To compile the Rust bindings and bundle the required libraries:

```bash
./scripts/compile_rust_mac.sh
```

This script builds the Rust crate, grabs the necessary `libtorch` dylibs, and packages them into `pkg/rustbert/lib/darwin`.

## Usage

```go
package main

import (
	"fmt"
	"log"

	"github.com/go-rust-bert/go-rust-bert/pkg/rustbert"
)

func main() {
	// 1. Initialize the library (extracts bundled shared libraries)
	if err := rustbert.Init(); err != nil {
		log.Fatalf("Failed to initialize: %v", err)
	}

	// 2. Create a Sentiment Analysis model
	// (Downloads model from Hugging Face on first run)
	model, err := rustbert.NewSentimentModel()
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}
	defer model.Close()

	// 3. Make predictions
	result, err := model.Predict("I love writing Go code!")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Label: %s\n", result.Label) // Output: Positive
	fmt.Printf("Score: %f\n", result.Score)
}
```

## Running Tests

```bash
go test -v ./pkg/rustbert/...
```
