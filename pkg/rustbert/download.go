package rustbert

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/gomlx/go-huggingface/hub"
)

// DownloadArtifacts downloads the necessary files for a model from Hugging Face.
// repoID: e.g. "distilbert-base-uncased-finetuned-sst-2-english"
// cacheDir: directory to store the model. If empty, uses ~/.cache/rustbert
//
// Returns paths to: model, config, vocab, merges (optional)
func DownloadArtifacts(repoID, cacheDir string) (string, string, string, string, error) {
	if cacheDir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", "", "", "", fmt.Errorf("failed to get user home dir: %w", err)
		}
		cacheDir = filepath.Join(home, ".cache", "rustbert")
	}

	repo := hub.New(repoID).WithCacheDir(cacheDir)

	// Download model weights
	// rust-bert typically supports rust_model.ot
	modelPath, err := repo.DownloadFile("rust_model.ot")
	if err != nil {
		// Fallback or just fail? rust-bert usually needs rust_model.ot specifically for Torch backend.
		return "", "", "", "", fmt.Errorf("failed to download rust_model.ot: %w", err)
	}

	// Download config
	configPath, err := repo.DownloadFile("config.json")
	if err != nil {
		return "", "", "", "", fmt.Errorf("failed to download config.json: %w", err)
	}

	// Download vocab
	// Could be vocab.txt (BERT) or vocab.json (GPT/RoBERTa)
	// We might need to try both or user needs to specify.
	// For generic download, let's try likely candidates.
	vocabPath, err := repo.DownloadFile("vocab.txt")
	if err != nil {
		// Try vocab.json
		if vp, err2 := repo.DownloadFile("vocab.json"); err2 == nil {
			vocabPath = vp
			err = nil
		} else {
			// For some models (like RoBERTa), vocab.json is standard.
			return "", "", "", "", fmt.Errorf("failed to download vocab.txt or vocab.json: %w", err)
		}
	}

	// Download merges (optional, for BPE like RoBERTa/GPT)
	mergesPath := ""
	if mp, err := repo.DownloadFile("merges.txt"); err == nil {
		mergesPath = mp
	}

	return modelPath, configPath, vocabPath, mergesPath, nil
}
