package rustbert

/*
#cgo LDFLAGS: -ldl
#include <stdlib.h>
#include <dlfcn.h>
#include <stdio.h>

void* open_lib(const char* path) {
    return dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
}

void* get_sym(void* handle, const char* name) {
    return dlsym(handle, name);
}

char* get_dlerror() {
    return dlerror();
}
*/
import "C"
import (
	"compress/gzip"
	"embed"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"unsafe"
)

//go:generate ../../scripts/compile_rust_linux.sh
//go:embed lib
var libFS embed.FS

var (
	initialized bool
	dlHandle    unsafe.Pointer
)

func Init() error {
	if initialized {
		return nil
	}

	goOS := runtime.GOOS
	goArch := runtime.GOARCH

	var libPath string
	var libName string

	switch goOS {
	case "darwin":
		libPath = "lib/darwin"
		libName = "librust_bert_binding.dylib.gz"
	case "linux":
		if goArch == "amd64" {
			libPath = "lib/linux-amd64"
		} else if goArch == "arm64" {
			libPath = "lib/linux-arm64"
		} else {
			return fmt.Errorf("unsupported linux architecture: %s", goArch)
		}
		libName = "librust_bert_binding.so.gz"
	default:
		return fmt.Errorf("unsupported OS: %s", goOS)
	}

	tmpDir, err := os.MkdirTemp("", "go-rust-bert-lib")
	if err != nil {
		return fmt.Errorf("failed to create temp dir: %w", err)
	}

	// Extract Libs
	libNameBinding := strings.TrimSuffix(libName, ".gz")
	destBinding := filepath.Join(tmpDir, libNameBinding)
	if err := extractAndDecompress(filepath.Join(libPath, libName), destBinding); err != nil {
		return err
	}

	// Extract libtorch dependencies - platform specific
	var libs []string
	var optionalLibs []string
	switch goOS {
	case "darwin":
		libs = []string{"libc10.dylib.gz", "libtorch_cpu.dylib.gz", "libtorch.dylib.gz", "libomp.dylib.gz"}
	case "linux":
		libs = []string{"libc10.so.gz", "libtorch_cpu.so.gz", "libtorch.so.gz"}
		// gomp library will be discovered dynamically
	}
	for _, l := range libs {
		dest := filepath.Join(tmpDir, strings.TrimSuffix(l, ".gz"))
		if err := extractAndDecompress(filepath.Join(libPath, l), dest); err != nil {
			return fmt.Errorf("failed to extract %s: %w", l, err)
		}
	}
	// Extract optional libs (don't fail if missing)
	for _, l := range optionalLibs {
		dest := filepath.Join(tmpDir, strings.TrimSuffix(l, ".gz"))
		_ = extractAndDecompress(filepath.Join(libPath, l), dest)
	}

	// On Linux, discover and extract any libgomp*.so*.gz files (name includes hash)
	if goOS == "linux" {
		entries, _ := libFS.ReadDir(libPath)
		for _, entry := range entries {
			name := entry.Name()
			if strings.HasPrefix(name, "libgomp") && strings.HasSuffix(name, ".gz") {
				dest := filepath.Join(tmpDir, strings.TrimSuffix(name, ".gz"))
				_ = extractAndDecompress(filepath.Join(libPath, name), dest)
			}
		}
	}

	// Load dependencies first with RTLD_GLOBAL so they're available to main lib
	// Order matters: load dependencies before dependents
	var depLibNames []string
	switch goOS {
	case "darwin":
		depLibNames = []string{"libomp.dylib", "libc10.dylib", "libtorch.dylib", "libtorch_cpu.dylib"}
	case "linux":
		depLibNames = []string{"libc10.so", "libtorch.so", "libtorch_cpu.so"}
		// Also load any libgomp libraries
		files, _ := os.ReadDir(tmpDir)
		for _, f := range files {
			if strings.HasPrefix(f.Name(), "libgomp") && strings.HasSuffix(f.Name(), ".so.1") {
				depLibNames = append([]string{f.Name()}, depLibNames...)
			}
		}
	}
	for _, depLib := range depLibNames {
		depPath := filepath.Join(tmpDir, depLib)
		cDepPath := C.CString(depPath)
		depHandle := C.open_lib(cDepPath)
		C.free(unsafe.Pointer(cDepPath))
		if depHandle == nil {
			cErr := C.get_dlerror()
			return fmt.Errorf("dlopen dependency %s failed: %s", depLib, C.GoString(cErr))
		}
	}

	// DLOPEN main binding library
	cPath := C.CString(destBinding)
	defer C.free(unsafe.Pointer(cPath))
	dlHandle = C.open_lib(cPath)
	if dlHandle == nil {
		cErr := C.get_dlerror()
		return fmt.Errorf("dlopen failed: %s", C.GoString(cErr))
	}

	// Load Symbols
	loadSym := func(name string) (unsafe.Pointer, error) {
		cName := C.CString(name)
		defer C.free(unsafe.Pointer(cName))
		sym := C.get_sym(dlHandle, cName)
		if sym == nil {
			return nil, fmt.Errorf("symbol not found: %s", name)
		}
		return sym, nil
	}

	if fnNewSentimentModel, err = loadSym("new_sentiment_model"); err != nil {
		return err
	}
	if fnNewSentimentModelFromFiles, err = loadSym("new_sentiment_model_from_files"); err != nil {
		return err
	}
	if fnPredictSentiment, err = loadSym("predict_sentiment"); err != nil {
		return err
	}
	if fnFreeSentimentModel, err = loadSym("free_sentiment_model"); err != nil {
		return err
	}
	if fnFreeSentimentResult, err = loadSym("free_sentiment_result"); err != nil {
		return err
	}

	// POS Tagging
	if fnNewPOSModel, err = loadSym("new_pos_model"); err != nil {
		return err
	}
	if fnPredictPOS, err = loadSym("predict_pos"); err != nil {
		return err
	}
	if fnFreePOSModel, err = loadSym("free_pos_model"); err != nil {
		return err
	}
	if fnFreePOSResult, err = loadSym("free_pos_result"); err != nil {
		return err
	}

	// NER
	if fnNewNERModel, err = loadSym("new_ner_model"); err != nil {
		return err
	}
	if fnNewNERModelFromFiles, err = loadSym("new_ner_model_from_files"); err != nil {
		return err
	}
	if fnPredictNER, err = loadSym("predict_ner"); err != nil {
		return err
	}
	if fnFreeNERModel, err = loadSym("free_ner_model"); err != nil {
		return err
	}
	if fnFreeNERResult, err = loadSym("free_ner_result"); err != nil {
		return err
	}

	// Question Answering
	if fnNewQAModel, err = loadSym("new_qa_model"); err != nil {
		return err
	}
	if fnNewQAModelFromFiles, err = loadSym("new_qa_model_from_files"); err != nil {
		return err
	}
	if fnPredictQA, err = loadSym("predict_qa"); err != nil {
		return err
	}
	if fnFreeQAModel, err = loadSym("free_qa_model"); err != nil {
		return err
	}
	if fnFreeQAResult, err = loadSym("free_qa_result"); err != nil {
		return err
	}

	// Summarization
	if fnNewSummarizationModel, err = loadSym("new_summarization_model"); err != nil {
		return err
	}
	if fnNewSummarizationModelFromFiles, err = loadSym("new_summarization_model_from_files"); err != nil {
		return err
	}
	if fnSummarize, err = loadSym("summarize"); err != nil {
		return err
	}
	if fnFreeSummarizationModel, err = loadSym("free_summarization_model"); err != nil {
		return err
	}
	if fnFreeSummarizationResult, err = loadSym("free_summarization_result"); err != nil {
		return err
	}

	// Zero-Shot Classification
	if fnNewZeroShotModel, err = loadSym("new_zero_shot_model"); err != nil {
		return err
	}
	if fnNewZeroShotModelFromFiles, err = loadSym("new_zero_shot_model_from_files"); err != nil {
		return err
	}
	if fnPredictZeroShot, err = loadSym("predict_zero_shot"); err != nil {
		return err
	}
	if fnFreeZeroShotModel, err = loadSym("free_zero_shot_model"); err != nil {
		return err
	}
	if fnFreeZeroShotResult, err = loadSym("free_zero_shot_result"); err != nil {
		return err
	}

	// Translation
	if fnNewTranslationModel, err = loadSym("new_translation_model"); err != nil {
		return err
	}
	if fnNewTranslationModelFromFiles, err = loadSym("new_translation_model_from_files"); err != nil {
		return err
	}
	if fnTranslate, err = loadSym("translate"); err != nil {
		return err
	}
	if fnFreeTranslationModel, err = loadSym("free_translation_model"); err != nil {
		return err
	}

	// Text Generation
	if fnNewTextGenerationModel, err = loadSym("new_text_generation_model"); err != nil {
		return err
	}
	if fnNewTextGenerationModelFromFiles, err = loadSym("new_text_generation_model_from_files"); err != nil {
		return err
	}
	if fnGenerateText, err = loadSym("generate_text"); err != nil {
		return err
	}
	if fnFreeTextGenerationModel, err = loadSym("free_text_generation_model"); err != nil {
		return err
	}

	initialized = true
	return nil
}

func extractAndDecompress(srcPath, destPath string) error {
	var r io.Reader
	var closers []io.Closer

	// CLEANUP helper
	defer func() {
		for _, c := range closers {
			c.Close()
		}
	}()

	// 1. Try opening direct file
	f, err := libFS.Open(srcPath)
	if err == nil {
		closers = append(closers, f)
		r = f
	} else {
		// 2. Try split files (partaa, partab, ...)
		// We assume standard split suffixes: aa, ab, ac...
		var readers []io.Reader
		
		// Attempt to find parts
		// We support up to 'az' which is plenty for 100MB chunks of a few GB file
		// ASCII 97='a'
		for i := 0; i < 26; i++ { // first char
			for j := 0; j < 26; j++ { // second char
				suffix := fmt.Sprintf(".part%c%c", 'a'+i, 'a'+j)
				partPath := srcPath + suffix
				
				pf, err := libFS.Open(partPath)
				if err != nil {
					// Stop at first missing part. 
					// However, ensure we found at least one if we are relying on split
					if len(readers) == 0 {
						// No parts found, and main file missing -> error
						return fmt.Errorf("open embedded %s: %w", srcPath, err)
					}
					// End of parts
					goto PartsDone
				}
				closers = append(closers, pf)
				readers = append(readers, pf)
			}
		}
	PartsDone:
		if len(readers) > 0 {
			r = io.MultiReader(readers...)
		}
	}

	if r == nil {
		return fmt.Errorf("failed to open source %s (or parts)", srcPath)
	}

	// 3. Decompress if needed
	if strings.HasSuffix(srcPath, ".gz") {
		gz, err := gzip.NewReader(r)
		if err != nil {
			return fmt.Errorf("gzip reader %s: %w", srcPath, err)
		}
		defer gz.Close()
		r = gz
	}

	out, err := os.OpenFile(destPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0755)
	if err != nil {
		return fmt.Errorf("create dest %s: %w", destPath, err)
	}
	defer out.Close()

	if _, err := io.Copy(out, r); err != nil {
		return fmt.Errorf("copy %s: %w", srcPath, err)
	}
	return nil
}

func init() {
	if err := Init(); err != nil {
		// Log but don't panic, let the user decide if they want to handle it
		fmt.Fprintf(os.Stderr, "WARNING: go-rust-bert failed to initialize: %v\n", err)
	}
}
