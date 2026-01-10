package rustbert

/*
#include <stdlib.h>

// --- Sentiment Analysis ---

typedef struct {
    void* model;
} SentimentModelWrapper;

typedef struct {
    char* label;
    float score;
} SentimentResult;

// --- POS Tagging ---

typedef struct {
    void* model;
} POSModelWrapper;

typedef struct {
    char* word;
    float score;
    char* label;
} POSTag;

typedef struct {
    POSTag* tags;
    size_t count;
} POSResult;

// --- NER ---

typedef struct {
    void* model;
} NERModelWrapper;

typedef struct {
    char* word;
    float score;
    char* label;
    size_t offset_begin;
    size_t offset_end;
} Entity;

typedef struct {
    Entity* entities;
    size_t count;
} NERResult;

// --- Question Answering ---

typedef struct {
    void* model;
} QAModelWrapper;

typedef struct {
    float score;
    size_t start;
    size_t end;
    char* answer;
} QAAnswer;

typedef struct {
    QAAnswer* answers;
    size_t count;
} QAResult;

// --- Summarization ---

typedef struct {
    void* model;
} SummarizationModelWrapper;

typedef struct {
    char** summaries;
    size_t count;
} SummarizationResult;

// --- Zero-Shot Classification ---

typedef struct {
    void* model;
} ZeroShotClassificationModelWrapper;

typedef struct {
    char* text;
    double score;
} ZeroShotLabel;

typedef struct {
    ZeroShotLabel* labels;
    size_t count;
} ZeroShotResult;

// --- Translation ---

typedef struct {
    void* model;
} TranslationModelWrapper;

// --- Text Generation ---

typedef struct {
    void* model;
} TextGenerationModelWrapper;

// Function pointer typedefs
typedef SentimentModelWrapper* (*new_sentiment_model_t)();
typedef SentimentResult* (*predict_sentiment_t)(SentimentModelWrapper*, const char*);
typedef void (*free_sentiment_model_t)(SentimentModelWrapper*);
typedef void (*free_sentiment_result_t)(SentimentResult*);

typedef POSModelWrapper* (*new_pos_model_t)();
typedef POSResult* (*predict_pos_t)(POSModelWrapper*, const char*);
typedef void (*free_pos_model_t)(POSModelWrapper*);
typedef void (*free_pos_result_t)(POSResult*);

typedef NERModelWrapper* (*new_ner_model_t)();
typedef NERResult* (*predict_ner_t)(NERModelWrapper*, const char*);
typedef void (*free_ner_model_t)(NERModelWrapper*);
typedef void (*free_ner_result_t)(NERResult*);

typedef QAModelWrapper* (*new_qa_model_t)();
typedef QAResult* (*predict_qa_t)(QAModelWrapper*, const char*, const char*);
typedef void (*free_qa_model_t)(QAModelWrapper*);
typedef void (*free_qa_result_t)(QAResult*);

typedef SummarizationModelWrapper* (*new_summarization_model_t)();
typedef SummarizationResult* (*summarize_t)(SummarizationModelWrapper*, const char*);
typedef void (*free_summarization_model_t)(SummarizationModelWrapper*);
typedef void (*free_summarization_result_t)(SummarizationResult*);

typedef ZeroShotClassificationModelWrapper* (*new_zero_shot_model_t)();
typedef ZeroShotResult* (*predict_zero_shot_t)(ZeroShotClassificationModelWrapper*, const char*, const char**, size_t);
typedef void (*free_zero_shot_model_t)(ZeroShotClassificationModelWrapper*);
typedef void (*free_zero_shot_result_t)(ZeroShotResult*);

typedef TranslationModelWrapper* (*new_translation_model_t)();
typedef char* (*translate_t)(TranslationModelWrapper*, const char*, const char*, const char*);
typedef void (*free_translation_model_t)(TranslationModelWrapper*);

typedef SentimentModelWrapper* (*new_sentiment_model_from_files_t)(const char*, const char*, const char*, const char*, int);

typedef TextGenerationModelWrapper* (*new_text_generation_model_t)();
typedef char* (*generate_text_t)(TextGenerationModelWrapper*, const char*, const char*);
typedef void (*free_text_generation_model_t)(TextGenerationModelWrapper*);

typedef NERModelWrapper* (*new_ner_model_from_files_t)(const char*, const char*, const char*, const char*, int);
typedef QAModelWrapper* (*new_qa_model_from_files_t)(const char*, const char*, const char*, const char*, int);
typedef SummarizationModelWrapper* (*new_summarization_model_from_files_t)(const char*, const char*, const char*, const char*, int);
typedef ZeroShotClassificationModelWrapper* (*new_zero_shot_model_from_files_t)(const char*, const char*, const char*, const char*, int);
typedef TranslationModelWrapper* (*new_translation_model_from_files_t)(const char*, const char*, const char*, const char*, int);
typedef TextGenerationModelWrapper* (*new_text_generation_model_from_files_t)(const char*, const char*, const char*, const char*, int);

// Helpers to call function pointers from C
SentimentModelWrapper* call_new_sentiment_model(void* f) {
    return ((new_sentiment_model_t)f)();
}

SentimentModelWrapper* call_new_sentiment_model_from_files(void* f, const char* m, const char* c, const char* v, const char* me, int t) {
    return ((new_sentiment_model_from_files_t)f)(m, c, v, me, t);
}

SentimentResult* call_predict_sentiment(void* f, SentimentModelWrapper* w, const char* text) {
    return ((predict_sentiment_t)f)(w, text);
}

void call_free_sentiment_model(void* f, SentimentModelWrapper* w) {
    ((free_sentiment_model_t)f)(w);
}

void call_free_sentiment_result(void* f, SentimentResult* r) {
    ((free_sentiment_result_t)f)(r);
}

POSModelWrapper* call_new_pos_model(void* f) {
    return ((new_pos_model_t)f)();
}

POSResult* call_predict_pos(void* f, POSModelWrapper* w, const char* text) {
    return ((predict_pos_t)f)(w, text);
}

void call_free_pos_model(void* f, POSModelWrapper* w) {
    ((free_pos_model_t)f)(w);
}

void call_free_pos_result(void* f, POSResult* r) {
    ((free_pos_result_t)f)(r);
}

NERModelWrapper* call_new_ner_model(void* f) {
    return ((new_ner_model_t)f)();
}

NERResult* call_predict_ner(void* f, NERModelWrapper* w, const char* text) {
    return ((predict_ner_t)f)(w, text);
}

void call_free_ner_model(void* f, NERModelWrapper* w) {
    ((free_ner_model_t)f)(w);
}

void call_free_ner_result(void* f, NERResult* r) {
    ((free_ner_result_t)f)(r);
}

QAModelWrapper* call_new_qa_model(void* f) {
    return ((new_qa_model_t)f)();
}

QAResult* call_predict_qa(void* f, QAModelWrapper* w, const char* question, const char* context) {
    return ((predict_qa_t)f)(w, question, context);
}

void call_free_qa_model(void* f, QAModelWrapper* w) {
    ((free_qa_model_t)f)(w);
}

void call_free_qa_result(void* f, QAResult* r) {
    ((free_qa_result_t)f)(r);
}

SummarizationModelWrapper* call_new_summarization_model(void* f) {
    return ((new_summarization_model_t)f)();
}

SummarizationResult* call_summarize(void* f, SummarizationModelWrapper* w, const char* text) {
    return ((summarize_t)f)(w, text);
}

void call_free_summarization_model(void* f, SummarizationModelWrapper* w) {
    ((free_summarization_model_t)f)(w);
}

void call_free_summarization_result(void* f, SummarizationResult* r) {
    ((free_summarization_result_t)f)(r);
}

ZeroShotClassificationModelWrapper* call_new_zero_shot_model(void* f) {
    return ((new_zero_shot_model_t)f)();
}

ZeroShotResult* call_predict_zero_shot(
    void* f,
    ZeroShotClassificationModelWrapper* w,
    const char* text,
    const char** labels,
    size_t labels_count
) {
    return ((predict_zero_shot_t)f)(w, text, labels, labels_count);
}

void call_free_zero_shot_model(void* f, ZeroShotClassificationModelWrapper* w) {
    ((free_zero_shot_model_t)f)(w);
}

void call_free_zero_shot_result(void* f, ZeroShotResult* r) {
    ((free_zero_shot_result_t)f)(r);
}

TranslationModelWrapper* call_new_translation_model(void* f) {
    return ((new_translation_model_t)f)();
}

char* call_translate(
    void* f,
    TranslationModelWrapper* w,
    const char* text,
    const char* source_lang,
    const char* target_lang
) {
    return ((translate_t)f)(w, text, source_lang, target_lang);
}

void call_free_translation_model(void* f, TranslationModelWrapper* w) {
    ((free_translation_model_t)f)(w);
}

TextGenerationModelWrapper* call_new_text_generation_model(void* f) {
    return ((new_text_generation_model_t)f)();
}

char* call_generate_text(
    void* f,
    TextGenerationModelWrapper* w,
    const char* prompt,
    const char* prefix
) {
    return ((generate_text_t)f)(w, prompt, prefix);
}

void call_free_text_generation_model(void* f, TextGenerationModelWrapper* w) {
    ((free_text_generation_model_t)f)(w);
}

// Helpers for custom loaders
void* call_new_ner_model_from_files(void* f, const char* m, const char* c, const char* v, const char* me, int t) {
    return ((new_ner_model_from_files_t)f)(m, c, v, me, t);
}

void* call_new_qa_model_from_files(void* f, const char* m, const char* c, const char* v, const char* me, int t) {
    return ((new_qa_model_from_files_t)f)(m, c, v, me, t);
}

void* call_new_summarization_model_from_files(void* f, const char* m, const char* c, const char* v, const char* me, int t) {
    return ((new_summarization_model_from_files_t)f)(m, c, v, me, t);
}

void* call_new_zero_shot_model_from_files(void* f, const char* m, const char* c, const char* v, const char* me, int t) {
    return ((new_zero_shot_model_from_files_t)f)(m, c, v, me, t);
}

void* call_new_translation_model_from_files(void* f, const char* m, const char* c, const char* v, const char* me, int t) {
    return ((new_translation_model_from_files_t)f)(m, c, v, me, t);
}

void* call_new_text_generation_model_from_files(void* f, const char* m, const char* c, const char* v, const char* me, int t) {
    return ((new_text_generation_model_from_files_t)f)(m, c, v, me, t);
}
*/
import "C"

import (
	"errors"
	"runtime"
	"unsafe"
)

// ModelType constants matching Rust implementation
const (
	ModelTypeBert       = 0
	ModelTypeDistilBert = 1
	ModelTypeRoberta    = 2
	ModelTypeXLMRoberta = 3
	ModelTypeElectra    = 4
	ModelTypeAlbert     = 5
	ModelTypeXLNet      = 6
	ModelTypeBart       = 7
	ModelTypeMarian     = 8
	ModelTypeT5         = 9
	ModelTypeGPT2       = 10
)

var (
	fnNewSentimentModel          unsafe.Pointer
	fnNewSentimentModelFromFiles unsafe.Pointer
	fnPredictSentiment           unsafe.Pointer
	fnFreeSentimentModel         unsafe.Pointer
	fnFreeSentimentResult        unsafe.Pointer

	fnNewPOSModel   unsafe.Pointer
	fnPredictPOS    unsafe.Pointer
	fnFreePOSModel  unsafe.Pointer
	fnFreePOSResult unsafe.Pointer

	fnNewNERModel          unsafe.Pointer
	fnNewNERModelFromFiles unsafe.Pointer
	fnPredictNER           unsafe.Pointer
	fnFreeNERModel         unsafe.Pointer
	fnFreeNERResult        unsafe.Pointer

	fnNewQAModel          unsafe.Pointer
	fnNewQAModelFromFiles unsafe.Pointer
	fnPredictQA           unsafe.Pointer
	fnFreeQAModel         unsafe.Pointer
	fnFreeQAResult        unsafe.Pointer

	fnNewSummarizationModel          unsafe.Pointer
	fnNewSummarizationModelFromFiles unsafe.Pointer
	fnSummarize                      unsafe.Pointer
	fnFreeSummarizationModel         unsafe.Pointer
	fnFreeSummarizationResult        unsafe.Pointer

	fnNewZeroShotModel          unsafe.Pointer
	fnNewZeroShotModelFromFiles unsafe.Pointer
	fnPredictZeroShot           unsafe.Pointer
	fnFreeZeroShotModel         unsafe.Pointer
	fnFreeZeroShotResult        unsafe.Pointer

	fnNewTranslationModel          unsafe.Pointer
	fnNewTranslationModelFromFiles unsafe.Pointer
	fnTranslate                    unsafe.Pointer
	fnFreeTranslationModel         unsafe.Pointer

	fnNewTextGenerationModel          unsafe.Pointer
	fnNewTextGenerationModelFromFiles unsafe.Pointer
	fnGenerateText                    unsafe.Pointer
	fnFreeTextGenerationModel         unsafe.Pointer
)

// Helper for calling *from_files functions which all have same signature
func callNewModelFromFiles(fn unsafe.Pointer, helper func(unsafe.Pointer, *C.char, *C.char, *C.char, *C.char, C.int) unsafe.Pointer, modelPath, configPath, vocabPath, mergesPath string, modelType int) (unsafe.Pointer, error) {
	if !initialized {
		return nil, errors.New("library not initialized")
	}

	cModel := C.CString(modelPath)
	cConfig := C.CString(configPath)
	cVocab := C.CString(vocabPath)
	defer C.free(unsafe.Pointer(cModel))
	defer C.free(unsafe.Pointer(cConfig))
	defer C.free(unsafe.Pointer(cVocab))

	var cMerges *C.char
	if mergesPath != "" {
		cMerges = C.CString(mergesPath)
		defer C.free(unsafe.Pointer(cMerges))
	}

	ptr := helper(fn, cModel, cConfig, cVocab, cMerges, C.int(modelType))
	if ptr == nil {
		return nil, errors.New("failed to create custom model")
	}
	return ptr, nil
}

// SentimentModel is a wrapper around the Rust sentiment analysis model
type SentimentModel struct {
	ptr *C.SentimentModelWrapper
}

// SentimentResult represents the output of sentiment analysis
type SentimentResult struct {
	Label string
	Score float64
}

// NewSentimentModel creates a new sentiment analysis model
func NewSentimentModel() (*SentimentModel, error) {
	if !initialized {
		return nil, errors.New("library not initialized")
	}

	ptr := C.call_new_sentiment_model(fnNewSentimentModel)
	if ptr == nil {
		return nil, errors.New("failed to create sentiment model")
	}
	return &SentimentModel{ptr: ptr}, nil
}

// NewSentimentModelFromFiles creates a new SentimentModel using local files.
// mergesPath is optional (pass "" if not used).
func NewSentimentModelFromFiles(modelPath, configPath, vocabPath, mergesPath string, modelType int) (*SentimentModel, error) {
	ptr, err := callNewModelFromFiles(fnNewSentimentModelFromFiles, func(fn unsafe.Pointer, m, c, v, me *C.char, t C.int) unsafe.Pointer {
		return unsafe.Pointer(C.call_new_sentiment_model_from_files(fn, m, c, v, me, t))
	}, modelPath, configPath, vocabPath, mergesPath, modelType)
	if err != nil {
		return nil, err
	}
	return &SentimentModel{ptr: (*C.SentimentModelWrapper)(ptr)}, nil
}

func NewNERModelFromFiles(modelPath, configPath, vocabPath, mergesPath string, modelType int) (*NERModel, error) {
	ptr, err := callNewModelFromFiles(fnNewNERModelFromFiles, func(fn unsafe.Pointer, m, c, v, me *C.char, t C.int) unsafe.Pointer {
		return unsafe.Pointer(C.call_new_ner_model_from_files(fn, m, c, v, me, t))
	}, modelPath, configPath, vocabPath, mergesPath, modelType)
	if err != nil {
		return nil, err
	}
	return &NERModel{ptr: (*C.NERModelWrapper)(ptr)}, nil
}

func NewQAModelFromFiles(modelPath, configPath, vocabPath, mergesPath string, modelType int) (*QAModel, error) {
	ptr, err := callNewModelFromFiles(fnNewQAModelFromFiles, func(fn unsafe.Pointer, m, c, v, me *C.char, t C.int) unsafe.Pointer {
		return unsafe.Pointer(C.call_new_qa_model_from_files(fn, m, c, v, me, t))
	}, modelPath, configPath, vocabPath, mergesPath, modelType)
	if err != nil {
		return nil, err
	}
	return &QAModel{ptr: (*C.QAModelWrapper)(ptr)}, nil
}

func NewSummarizationModelFromFiles(modelPath, configPath, vocabPath, mergesPath string, modelType int) (*SummarizationModel, error) {
	ptr, err := callNewModelFromFiles(fnNewSummarizationModelFromFiles, func(fn unsafe.Pointer, m, c, v, me *C.char, t C.int) unsafe.Pointer {
		return unsafe.Pointer(C.call_new_summarization_model_from_files(fn, m, c, v, me, t))
	}, modelPath, configPath, vocabPath, mergesPath, modelType)
	if err != nil {
		return nil, err
	}
	return &SummarizationModel{ptr: (*C.SummarizationModelWrapper)(ptr)}, nil
}

func NewZeroShotModelFromFiles(modelPath, configPath, vocabPath, mergesPath string, modelType int) (*ZeroShotModel, error) {
	ptr, err := callNewModelFromFiles(fnNewZeroShotModelFromFiles, func(fn unsafe.Pointer, m, c, v, me *C.char, t C.int) unsafe.Pointer {
		return unsafe.Pointer(C.call_new_zero_shot_model_from_files(fn, m, c, v, me, t))
	}, modelPath, configPath, vocabPath, mergesPath, modelType)
	if err != nil {
		return nil, err
	}
	return &ZeroShotModel{ptr: (*C.ZeroShotClassificationModelWrapper)(ptr)}, nil
}

func NewTranslationModelFromFiles(modelPath, configPath, vocabPath, mergesPath string, modelType int) (*TranslationModel, error) {
	ptr, err := callNewModelFromFiles(fnNewTranslationModelFromFiles, func(fn unsafe.Pointer, m, c, v, me *C.char, t C.int) unsafe.Pointer {
		return unsafe.Pointer(C.call_new_translation_model_from_files(fn, m, c, v, me, t))
	}, modelPath, configPath, vocabPath, mergesPath, modelType)
	if err != nil {
		return nil, err
	}
	return &TranslationModel{ptr: (*C.TranslationModelWrapper)(ptr)}, nil
}

func NewTextGenerationModelFromFiles(modelPath, configPath, vocabPath, mergesPath string, modelType int) (*TextGenerationModel, error) {
	ptr, err := callNewModelFromFiles(fnNewTextGenerationModelFromFiles, func(fn unsafe.Pointer, m, c, v, me *C.char, t C.int) unsafe.Pointer {
		return unsafe.Pointer(C.call_new_text_generation_model_from_files(fn, m, c, v, me, t))
	}, modelPath, configPath, vocabPath, mergesPath, modelType)
	if err != nil {
		return nil, err
	}
	return &TextGenerationModel{ptr: (*C.TextGenerationModelWrapper)(ptr)}, nil
}

// Predict performs sentiment analysis on the given text
func (m *SentimentModel) Predict(text string) (*SentimentResult, error) {
	if m.ptr == nil {
		return nil, errors.New("model is closed")
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	res := C.call_predict_sentiment(fnPredictSentiment, m.ptr, cText)
	if res == nil {
		return nil, errors.New("prediction failed")
	}
	defer C.call_free_sentiment_result(fnFreeSentimentResult, res)

	return &SentimentResult{
		Label: C.GoString(res.label),
		Score: float64(res.score),
	}, nil
}

// Close frees the underlying Rust model
func (m *SentimentModel) Close() {
	if m.ptr != nil {
		C.call_free_sentiment_model(fnFreeSentimentModel, m.ptr)
		m.ptr = nil
	}
}

// SetFinalizer ensures the model is closed when garbage collected (optional but good practice)
func (m *SentimentModel) SetFinalizer() {
	runtime.SetFinalizer(m, func(obj *SentimentModel) {
		obj.Close()
	})
}

// --- POS Tagging ---

// POSModel is a wrapper around the Rust POS tagging model
type POSModel struct {
	ptr *C.POSModelWrapper
}

// POSTag represents a single Part-of-Speech tag
type POSTag struct {
	Word  string
	Score float64
	Label string
}

// NewPOSModel creates a new POS tagging model
func NewPOSModel() (*POSModel, error) {
	if !initialized {
		return nil, errors.New("library not initialized")
	}

	ptr := C.call_new_pos_model(fnNewPOSModel)
	if ptr == nil {
		return nil, errors.New("failed to create POS model")
	}
	return &POSModel{ptr: ptr}, nil
}

// Predict performs POS tagging on the given text
func (m *POSModel) Predict(text string) ([]POSTag, error) {
	if m.ptr == nil {
		return nil, errors.New("model is closed")
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	res := C.call_predict_pos(fnPredictPOS, m.ptr, cText)
	if res == nil {
		return nil, errors.New("prediction failed")
	}
	defer C.call_free_pos_result(fnFreePOSResult, res)

	count := int(res.count)
	tags := make([]POSTag, count)

	// Access C array
	// unsafe.Slice requires Go 1.17+
	cTags := unsafe.Slice(res.tags, count)

	for i := 0; i < count; i++ {
		tags[i] = POSTag{
			Word:  C.GoString(cTags[i].word),
			Score: float64(cTags[i].score),
			Label: C.GoString(cTags[i].label),
		}
	}

	return tags, nil
}

// Close frees the underlying Rust model
func (m *POSModel) Close() {
	if m.ptr != nil {
		C.call_free_pos_model(fnFreePOSModel, m.ptr)
		m.ptr = nil
	}
}

// --- NER ---

// NERModel is a wrapper around the Rust NER model
type NERModel struct {
	ptr *C.NERModelWrapper
}

// Entity represents an extracted named entity
type Entity struct {
	Word   string
	Score  float64
	Label  string
	Offset struct {
		Begin int
		End   int
	}
}

// NewNERModel creates a new NER model
func NewNERModel() (*NERModel, error) {
	if !initialized {
		return nil, errors.New("library not initialized")
	}

	ptr := C.call_new_ner_model(fnNewNERModel)
	if ptr == nil {
		return nil, errors.New("failed to create NER model")
	}
	return &NERModel{ptr: ptr}, nil
}

// Predict performs named entity recognition on the given text
func (m *NERModel) Predict(text string) ([]Entity, error) {
	if m.ptr == nil {
		return nil, errors.New("model is closed")
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	res := C.call_predict_ner(fnPredictNER, m.ptr, cText)
	if res == nil {
		return nil, errors.New("prediction failed")
	}
	defer C.call_free_ner_result(fnFreeNERResult, res)

	count := int(res.count)
	entities := make([]Entity, count)

	cEntities := unsafe.Slice(res.entities, count)

	for i := 0; i < count; i++ {
		entities[i] = Entity{
			Word:  C.GoString(cEntities[i].word),
			Score: float64(cEntities[i].score),
			Label: C.GoString(cEntities[i].label),
			Offset: struct{ Begin, End int }{
				Begin: int(cEntities[i].offset_begin),
				End:   int(cEntities[i].offset_end),
			},
		}
	}

	return entities, nil
}

// Close frees the underlying Rust model
func (m *NERModel) Close() {
	if m.ptr != nil {
		C.call_free_ner_model(fnFreeNERModel, m.ptr)
		m.ptr = nil
	}
}

// --- Question Answering ---

// QAModel is a wrapper around the Rust QA model
type QAModel struct {
	ptr *C.QAModelWrapper
}

// Answer represents an extracted answer
type Answer struct {
	Score  float64
	Start  int
	End    int
	Answer string
}

// NewQAModel creates a new Question Answering model
func NewQAModel() (*QAModel, error) {
	if !initialized {
		return nil, errors.New("library not initialized")
	}

	ptr := C.call_new_qa_model(fnNewQAModel)
	if ptr == nil {
		return nil, errors.New("failed to create QA model")
	}
	return &QAModel{ptr: ptr}, nil
}

// Predict performs question answering
func (m *QAModel) Predict(question, context string) ([]Answer, error) {
	if m.ptr == nil {
		return nil, errors.New("model is closed")
	}

	cQuestion := C.CString(question)
	defer C.free(unsafe.Pointer(cQuestion))
	cContext := C.CString(context)
	defer C.free(unsafe.Pointer(cContext))

	res := C.call_predict_qa(fnPredictQA, m.ptr, cQuestion, cContext)
	if res == nil {
		return nil, errors.New("prediction failed")
	}
	defer C.call_free_qa_result(fnFreeQAResult, res)

	count := int(res.count)
	answers := make([]Answer, count)

	cAnswers := unsafe.Slice(res.answers, count)

	for i := 0; i < count; i++ {
		answers[i] = Answer{
			Score:  float64(cAnswers[i].score),
			Start:  int(cAnswers[i].start),
			End:    int(cAnswers[i].end),
			Answer: C.GoString(cAnswers[i].answer),
		}
	}

	return answers, nil
}

// Close frees the underlying Rust model
func (m *QAModel) Close() {
	if m.ptr != nil {
		C.call_free_qa_model(fnFreeQAModel, m.ptr)
		m.ptr = nil
	}
}

// --- Summarization ---

// SummarizationModel is a wrapper around the Rust Summarization model
type SummarizationModel struct {
	ptr *C.SummarizationModelWrapper
}

// NewSummarizationModel creates a new Summarization model
func NewSummarizationModel() (*SummarizationModel, error) {
	if !initialized {
		return nil, errors.New("library not initialized")
	}

	ptr := C.call_new_summarization_model(fnNewSummarizationModel)
	if ptr == nil {
		return nil, errors.New("failed to create Summarization model")
	}
	return &SummarizationModel{ptr: ptr}, nil
}

// Summarize performs text summarization
func (m *SummarizationModel) Summarize(text string) ([]string, error) {
	if m.ptr == nil {
		return nil, errors.New("model is closed")
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	res := C.call_summarize(fnSummarize, m.ptr, cText)
	if res == nil {
		return nil, errors.New("summarization failed")
	}
	defer C.call_free_summarization_result(fnFreeSummarizationResult, res)

	count := int(res.count)
	if count == 0 {
		return []string{}, nil
	}

	summaries := make([]string, count)
	cSummaries := unsafe.Slice(res.summaries, count)

	for i := 0; i < count; i++ {
		summaries[i] = C.GoString(cSummaries[i])
	}

	return summaries, nil
}

// Close frees the underlying Rust model
func (m *SummarizationModel) Close() {
	if m.ptr != nil {
		C.call_free_summarization_model(fnFreeSummarizationModel, m.ptr)
		m.ptr = nil
	}
}

// --- Zero-Shot Classification ---

// ZeroShotLabel represents a classification label and its score
type ZeroShotLabel struct {
	Text  string
	Score float64
}

// ZeroShotModel is a wrapper around the Rust Zero-Shot Classification model
type ZeroShotModel struct {
	ptr *C.ZeroShotClassificationModelWrapper
}

// NewZeroShotModel creates a new Zero-Shot Classification model
func NewZeroShotModel() (*ZeroShotModel, error) {
	if !initialized {
		return nil, errors.New("library not initialized")
	}

	ptr := C.call_new_zero_shot_model(fnNewZeroShotModel)
	if ptr == nil {
		return nil, errors.New("failed to create Zero-Shot model")
	}
	return &ZeroShotModel{ptr: ptr}, nil
}

// Predict performs zero-shot classification
func (m *ZeroShotModel) Predict(text string, labels []string) ([]ZeroShotLabel, error) {
	if m.ptr == nil {
		return nil, errors.New("model is closed")
	}

	if len(labels) == 0 {
		return nil, errors.New("labels cannot be empty")
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	cLabels := make([]*C.char, len(labels))
	for i, label := range labels {
		cLabels[i] = C.CString(label)
		defer C.free(unsafe.Pointer(cLabels[i]))
	}

	res := C.call_predict_zero_shot(
		fnPredictZeroShot,
		m.ptr,
		cText,
		&cLabels[0],
		C.size_t(len(labels)),
	)
	if res == nil {
		return nil, errors.New("zero-shot prediction failed")
	}
	defer C.call_free_zero_shot_result(fnFreeZeroShotResult, res)

	count := int(res.count)
	if count == 0 {
		return []ZeroShotLabel{}, nil
	}

	results := make([]ZeroShotLabel, count)
	cResults := unsafe.Slice(res.labels, count)

	for i := 0; i < count; i++ {
		results[i] = ZeroShotLabel{
			Text:  C.GoString(cResults[i].text),
			Score: float64(cResults[i].score),
		}
	}

	return results, nil
}

// Close frees the underlying Rust model
func (m *ZeroShotModel) Close() {
	if m.ptr != nil {
		C.call_free_zero_shot_model(fnFreeZeroShotModel, m.ptr)
		m.ptr = nil
	}
}

// --- Translation ---

// TranslationModel is a wrapper around the Rust Translation model
type TranslationModel struct {
	ptr *C.TranslationModelWrapper
}

// NewTranslationModel creates a new Translation model
func NewTranslationModel() (*TranslationModel, error) {
	if !initialized {
		return nil, errors.New("library not initialized")
	}

	ptr := C.call_new_translation_model(fnNewTranslationModel)
	if ptr == nil {
		return nil, errors.New("failed to create Translation model")
	}
	return &TranslationModel{ptr: ptr}, nil
}

// Translate performs translation of text
// sourceLang can be empty string for auto-detection or default behavior (though M2M100 usually needs it, generic binding supports optional)
func (m *TranslationModel) Translate(text string, sourceLang string, targetLang string) (string, error) {
	if m.ptr == nil {
		return "", errors.New("model is closed")
	}

	if targetLang == "" {
		return "", errors.New("target language cannot be empty")
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	cTarget := C.CString(targetLang)
	defer C.free(unsafe.Pointer(cTarget))

	var cSource *C.char
	if sourceLang != "" {
		cSource = C.CString(sourceLang)
		defer C.free(unsafe.Pointer(cSource))
	}

	cRes := C.call_translate(
		fnTranslate,
		m.ptr,
		cText,
		cSource,
		cTarget,
	)
	if cRes == nil {
		return "", errors.New("translation failed")
	}
	defer C.free(unsafe.Pointer(cRes))

	return C.GoString(cRes), nil
}

// Close frees the underlying Rust model
func (m *TranslationModel) Close() {
	if m.ptr != nil {
		C.call_free_translation_model(fnFreeTranslationModel, m.ptr)
		m.ptr = nil
	}
}

// --- Text Generation ---

// TextGenerationModel is a wrapper around the Rust Text Generation model
type TextGenerationModel struct {
	ptr *C.TextGenerationModelWrapper
}

// NewTextGenerationModel creates a new TextGeneration model (GPT2 Medium by default)
func NewTextGenerationModel() (*TextGenerationModel, error) {
	if !initialized {
		return nil, errors.New("library not initialized")
	}

	ptr := C.call_new_text_generation_model(fnNewTextGenerationModel)
	if ptr == nil {
		return nil, errors.New("failed to create Text Generation model")
	}
	return &TextGenerationModel{ptr: ptr}, nil
}

// Generate generates text based on prompt
// prefix can be empty string.
func (m *TextGenerationModel) Generate(prompt string, prefix string) (string, error) {
	if m.ptr == nil {
		return "", errors.New("model is closed")
	}

	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	var cPrefix *C.char
	if prefix != "" {
		cPrefix = C.CString(prefix)
		defer C.free(unsafe.Pointer(cPrefix))
	}

	// We pass dummy values for params we aren't supporting dynamically yet
	cRes := C.call_generate_text(
		fnGenerateText,
		m.ptr,
		cPrompt,
		cPrefix,
	)
	if cRes == nil {
		return "", errors.New("generation failed")
	}
	defer C.free(unsafe.Pointer(cRes))

	return C.GoString(cRes), nil
}

// Close frees the underlying Rust model
func (m *TextGenerationModel) Close() {
	if m.ptr != nil {
		C.call_free_text_generation_model(fnFreeTextGenerationModel, m.ptr)
		m.ptr = nil
	}
}
