# go-rust-bert

Golang bindings for [rust-bert](https://github.com/guillaume-be/rust-bert), bringing state-of-the-art NLP models to Go.

## Features

- **Sentiment Analysis**: Ready-to-use pipeline for sentiment classification.
- **Named Entity Recognition (NER)**: Extract entities (Person, Location, Org) from text.
- **Question Answering**: Extractive QA from context.
- **Summarization**: Abstractive summarization of long texts.
- **Zero-Shot Classification**: Classify text into arbitrary labels without training.
- **Translation**: Translate text between languages (supports Marian and M2M100 models).
- **Text Generation**: Generate text using GPT-2 and similar models.
- **Custom Model Loading**: Load any compatible model from local files using dynamic `ModelType` configuration.
- **Self-Contained**: Bundles necessary `libtorch` and Rust dynamic libraries.

## Prerequisites

- **macOS** (Atomic/ARM64 tested) or **Linux**
- **Rust** (stable toolchain)
- **Go** 1.22+

## Build

To compile the Rust bindings and bundle the libraries:

```bash
./scripts/compile_rust_mac.sh
```

## Usage

### Initialization

```go
package main

import (
	"log"
	"github.com/soundprediction/go-rust-bert/pkg/rustbert"
)

func main() {
	// Extracts bundled libraries to temporary location
	if err := rustbert.Init(); err != nil {
		log.Fatalf("Failed to initialize: %v", err)
	}
    // ... use models
}
```

### Sentiment Analysis

```go
model, _ := rustbert.NewSentimentModel() // Default DistilBERT SST-2
defer model.Close()

result, _ := model.Predict("I love writing Go code!")
fmt.Printf("%s: %f\n", result.Label, result.Score)
```

### Named Entity Recognition (NER)

```go
model, _ := rustbert.NewNERModel() // Default BERT cased
defer model.Close()

entities, _ := model.Predict("My name is John and I live in Paris.")
for _, e := range entities {
    fmt.Printf("%s: %s (%f)\n", e.Label, e.Word, e.Score)
}
```

### Question Answering

```go
model, _ := rustbert.NewQAModel()
defer model.Close()

answers, _ := model.Predict("Where do I live?", "My name is John and I live in Paris.")
for _, a := range answers {
    fmt.Printf("Answer: %s (Score: %f)\n", a.Answer, a.Score)
}
```

### Summarization

```go
model, _ := rustbert.NewSummarizationModel() // Default BART CNN
defer model.Close()

text := "Long text to summarize..."
summaries, _ := model.Summarize(text)
fmt.Println(summaries[0])
```

### Zero-Shot Classification

```go
model, _ := rustbert.NewZeroShotModel() // Default Bart MNLI
defer model.Close()

labels := []string{"politics", "public health", "economics"}
results, _ := model.Predict("The vaccine is efficient.", labels)
for _, r := range results {
    fmt.Printf("%s: %f\n", r.Text, r.Score)
}
```

### Text Generation

```go
model, _ := rustbert.NewTextGenerationModel() // Default GPT-2
defer model.Close()

generated, _ := model.Generate("The meaning of life is", "")
fmt.Println(generated)
```

### Translation

```go
model, _ := rustbert.NewTranslationModel() // Default Marian (Romance languages)
defer model.Close()

translated, _ := model.Translate("Hello world", "en", "fr")
fmt.Println(translated) // Bonjour le monde
```

### Custom Model Loading from Local Files

You can load custom models by downloading the artifacts (manually or via `DownloadArtifacts` helper) and specifying the model type.

Supported `ModelType` constants:
- `ModelTypeBert`
- `ModelTypeDistilBert`
- `ModelTypeRoberta`
- `ModelTypeBart`
- `ModelTypeMarian`
- `ModelTypeGPT2`
- ... and more.

```go
repoID := "distilbert-base-uncased-finetuned-sst-2-english"
// Auto-download helper
modelPath, configPath, vocabPath, mergesPath, _ := rustbert.DownloadArtifacts(repoID, "")

// Initialize with specific ModelType
model, _ := rustbert.NewSentimentModelFromFiles(
    modelPath, 
    configPath, 
    vocabPath, 
    mergesPath, 
    rustbert.ModelTypeDistilBert,
)
defer model.Close()

model.Predict("Custom loaded model works!")
```

## Running Tests

```bash
go test -v ./pkg/rustbert/...
```
