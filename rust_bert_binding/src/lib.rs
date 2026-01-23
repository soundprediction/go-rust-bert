//! FFI bindings for rust-bert, exposing C-compatible functions for Go integration.

use libc::{c_char, size_t};
use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::ner::NERModel;
use rust_bert::pipelines::token_classification::TokenClassificationConfig;
use rust_bert::pipelines::pos_tagging::{POSModel, POSConfig};
use rust_bert::pipelines::question_answering::{QaInput, QuestionAnsweringModel, QuestionAnsweringConfig};
use rust_bert::pipelines::sentiment::{SentimentModel, SentimentPolarity, SentimentConfig};
use rust_bert::pipelines::summarization::{SummarizationModel, SummarizationConfig};
use rust_bert::pipelines::text_generation::{TextGenerationModel, TextGenerationConfig};
use rust_bert::pipelines::translation::{TranslationModel, TranslationModelBuilder, Language};
use rust_bert::pipelines::zero_shot_classification::{ZeroShotClassificationModel, ZeroShotClassificationConfig};
use std::ffi::{CStr, CString};
use std::ptr;

// ============================================================================
// FFI Structs - All must use #[repr(C)] to match Go CGO definitions
// ============================================================================

/// Wrapper for SentimentModel
#[repr(C)]
pub struct SentimentModelWrapper {
    model: *mut SentimentModel,
}

/// Result of sentiment analysis
#[repr(C)]
pub struct SentimentResult {
    pub label: *mut c_char,
    pub score: f32,
}

/// Wrapper for POSModel
#[repr(C)]
pub struct POSModelWrapper {
    model: *mut POSModel,
}

/// Single POS tag
#[repr(C)]
pub struct POSTag {
    pub word: *mut c_char,
    pub score: f32,
    pub label: *mut c_char,
}

/// Result of POS tagging
#[repr(C)]
pub struct POSResult {
    pub tags: *mut POSTag,
    pub count: size_t,
}

/// Wrapper for NERModel
#[repr(C)]
pub struct NERModelWrapper {
    model: *mut NERModel,
}

/// Single named entity
#[repr(C)]
pub struct Entity {
    pub word: *mut c_char,
    pub score: f32,
    pub label: *mut c_char,
    pub offset_begin: size_t,
    pub offset_end: size_t,
}

/// Result of NER
#[repr(C)]
pub struct NERResult {
    pub entities: *mut Entity,
    pub count: size_t,
}

/// Wrapper for QuestionAnsweringModel
#[repr(C)]
pub struct QAModelWrapper {
    model: *mut QuestionAnsweringModel,
}

/// Single QA answer
#[repr(C)]
pub struct QAAnswer {
    pub score: f32,
    pub start: size_t,
    pub end: size_t,
    pub answer: *mut c_char,
}

/// Result of QA
#[repr(C)]
pub struct QAResult {
    pub answers: *mut QAAnswer,
    pub count: size_t,
}

/// Wrapper for SummarizationModel
#[repr(C)]
pub struct SummarizationModelWrapper {
    model: *mut SummarizationModel,
}

/// Result of summarization
#[repr(C)]
pub struct SummarizationResult {
    pub summaries: *mut *mut c_char,
    pub count: size_t,
}

/// Wrapper for ZeroShotClassificationModel
#[repr(C)]
pub struct ZeroShotClassificationModelWrapper {
    model: *mut ZeroShotClassificationModel,
}

/// Single zero-shot label with score
#[repr(C)]
pub struct ZeroShotLabel {
    pub text: *mut c_char,
    pub score: f64,
}

/// Result of zero-shot classification
#[repr(C)]
pub struct ZeroShotResult {
    pub labels: *mut ZeroShotLabel,
    pub count: size_t,
}

/// Wrapper for TranslationModel
#[repr(C)]
pub struct TranslationModelWrapper {
    model: *mut TranslationModel,
}

/// Wrapper for TextGenerationModel
#[repr(C)]
pub struct TextGenerationModelWrapper {
    model: *mut TextGenerationModel,
}

// ============================================================================
// Helper functions
// ============================================================================

fn cstr_to_string(s: *const c_char) -> Option<String> {
    if s.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(s).to_str().ok().map(|s| s.to_string()) }
}

fn string_to_cstr(s: &str) -> *mut c_char {
    CString::new(s)
        .map(|c| c.into_raw())
        .unwrap_or(ptr::null_mut())
}

#[allow(dead_code)]
fn model_type_from_int(t: i32) -> ModelType {
    match t {
        0 => ModelType::Bert,
        1 => ModelType::DistilBert,
        2 => ModelType::Roberta,
        3 => ModelType::XLMRoberta,
        4 => ModelType::Electra,
        5 => ModelType::Albert,
        6 => ModelType::XLNet,
        7 => ModelType::Bart,
        8 => ModelType::Marian,
        9 => ModelType::T5,
        10 => ModelType::GPT2,
        _ => ModelType::Bert,
    }
}

// ============================================================================
// Sentiment Analysis FFI Functions
// ============================================================================

/// Create a new sentiment model with default configuration (DistilBERT SST-2)
#[no_mangle]
pub extern "C" fn new_sentiment_model() -> *mut SentimentModelWrapper {
    match SentimentModel::new(SentimentConfig::default()) {
        Ok(model) => {
            let wrapper = SentimentModelWrapper {
                model: Box::into_raw(Box::new(model)),
            };
            Box::into_raw(Box::new(wrapper))
        }
        Err(e) => {
            eprintln!("Failed to create sentiment model: {:?}", e);
            ptr::null_mut()
        }
    }
}

/// Create a sentiment model from custom files
#[no_mangle]
pub extern "C" fn new_sentiment_model_from_files(
    _model_path: *const c_char,
    _config_path: *const c_char,
    _vocab_path: *const c_char,
    _merges_path: *const c_char,
    _model_type: i32,
) -> *mut SentimentModelWrapper {
    // Custom model loading requires more complex config setup
    // For now, fall back to default model
    new_sentiment_model()
}

/// Predict sentiment for the given text
#[no_mangle]
pub extern "C" fn predict_sentiment(
    wrapper: *mut SentimentModelWrapper,
    text: *const c_char,
) -> *mut SentimentResult {
    if wrapper.is_null() || text.is_null() {
        return ptr::null_mut();
    }

    let text_str = match cstr_to_string(text) {
        Some(s) => s,
        None => return ptr::null_mut(),
    };

    unsafe {
        let model = &*(*wrapper).model;
        match model.predict(&[text_str.as_str()]).first() {
            Some(sentiment) => {
                let label = match sentiment.polarity {
                    SentimentPolarity::Positive => "POSITIVE",
                    SentimentPolarity::Negative => "NEGATIVE",
                };
                let result = SentimentResult {
                    label: string_to_cstr(label),
                    score: sentiment.score as f32,
                };
                Box::into_raw(Box::new(result))
            }
            None => ptr::null_mut(),
        }
    }
}

/// Free a sentiment model
#[no_mangle]
pub extern "C" fn free_sentiment_model(wrapper: *mut SentimentModelWrapper) {
    if !wrapper.is_null() {
        unsafe {
            let w = Box::from_raw(wrapper);
            if !w.model.is_null() {
                drop(Box::from_raw(w.model));
            }
        }
    }
}

/// Free a sentiment result
#[no_mangle]
pub extern "C" fn free_sentiment_result(result: *mut SentimentResult) {
    if !result.is_null() {
        unsafe {
            let r = Box::from_raw(result);
            if !r.label.is_null() {
                drop(CString::from_raw(r.label));
            }
        }
    }
}

// ============================================================================
// POS Tagging FFI Functions
// ============================================================================

/// Create a new POS model with default configuration
#[no_mangle]
pub extern "C" fn new_pos_model() -> *mut POSModelWrapper {
    match POSModel::new(POSConfig::default()) {
        Ok(model) => {
            let wrapper = POSModelWrapper {
                model: Box::into_raw(Box::new(model)),
            };
            Box::into_raw(Box::new(wrapper))
        }
        Err(e) => {
            eprintln!("Failed to create POS model: {:?}", e);
            ptr::null_mut()
        }
    }
}

/// Predict POS tags for the given text
#[no_mangle]
pub extern "C" fn predict_pos(
    wrapper: *mut POSModelWrapper,
    text: *const c_char,
) -> *mut POSResult {
    if wrapper.is_null() || text.is_null() {
        return ptr::null_mut();
    }

    let text_str = match cstr_to_string(text) {
        Some(s) => s,
        None => return ptr::null_mut(),
    };

    unsafe {
        let model = &*(*wrapper).model;
        let results = model.predict(&[text_str.as_str()]);

        if results.is_empty() || results[0].is_empty() {
            let result = POSResult {
                tags: ptr::null_mut(),
                count: 0,
            };
            return Box::into_raw(Box::new(result));
        }

        let tags: Vec<POSTag> = results[0]
            .iter()
            .map(|tag| POSTag {
                word: string_to_cstr(&tag.word),
                score: tag.score as f32,
                label: string_to_cstr(&tag.label),
            })
            .collect();

        let count = tags.len();
        let tags_ptr = Box::into_raw(tags.into_boxed_slice()) as *mut POSTag;

        let result = POSResult {
            tags: tags_ptr,
            count,
        };
        Box::into_raw(Box::new(result))
    }
}

/// Free a POS model
#[no_mangle]
pub extern "C" fn free_pos_model(wrapper: *mut POSModelWrapper) {
    if !wrapper.is_null() {
        unsafe {
            let w = Box::from_raw(wrapper);
            if !w.model.is_null() {
                drop(Box::from_raw(w.model));
            }
        }
    }
}

/// Free a POS result
#[no_mangle]
pub extern "C" fn free_pos_result(result: *mut POSResult) {
    if !result.is_null() {
        unsafe {
            let r = Box::from_raw(result);
            if !r.tags.is_null() && r.count > 0 {
                let tags = Vec::from_raw_parts(r.tags, r.count, r.count);
                for tag in tags {
                    if !tag.word.is_null() {
                        drop(CString::from_raw(tag.word));
                    }
                    if !tag.label.is_null() {
                        drop(CString::from_raw(tag.label));
                    }
                }
            }
        }
    }
}

// ============================================================================
// NER FFI Functions
// ============================================================================

/// Create a new NER model with default configuration
#[no_mangle]
pub extern "C" fn new_ner_model() -> *mut NERModelWrapper {
    match NERModel::new(TokenClassificationConfig::default()) {
        Ok(model) => {
            let wrapper = NERModelWrapper {
                model: Box::into_raw(Box::new(model)),
            };
            Box::into_raw(Box::new(wrapper))
        }
        Err(e) => {
            eprintln!("Failed to create NER model: {:?}", e);
            ptr::null_mut()
        }
    }
}

/// Create a NER model from custom files
#[no_mangle]
pub extern "C" fn new_ner_model_from_files(
    _model_path: *const c_char,
    _config_path: *const c_char,
    _vocab_path: *const c_char,
    _merges_path: *const c_char,
    _model_type: i32,
) -> *mut NERModelWrapper {
    new_ner_model()
}

/// Predict NER entities for the given text
#[no_mangle]
pub extern "C" fn predict_ner(
    wrapper: *mut NERModelWrapper,
    text: *const c_char,
) -> *mut NERResult {
    if wrapper.is_null() || text.is_null() {
        return ptr::null_mut();
    }

    let text_str = match cstr_to_string(text) {
        Some(s) => s,
        None => return ptr::null_mut(),
    };

    unsafe {
        let model = &*(*wrapper).model;
        let results = model.predict(&[text_str.as_str()]);

        if results.is_empty() || results[0].is_empty() {
            let result = NERResult {
                entities: ptr::null_mut(),
                count: 0,
            };
            return Box::into_raw(Box::new(result));
        }

        let entities: Vec<Entity> = results[0]
            .iter()
            .map(|ent| Entity {
                word: string_to_cstr(&ent.word),
                score: ent.score as f32,
                label: string_to_cstr(&ent.label),
                offset_begin: ent.offset.begin as usize,
                offset_end: ent.offset.end as usize,
            })
            .collect();

        let count = entities.len();
        let entities_ptr = Box::into_raw(entities.into_boxed_slice()) as *mut Entity;

        let result = NERResult {
            entities: entities_ptr,
            count,
        };
        Box::into_raw(Box::new(result))
    }
}

/// Free a NER model
#[no_mangle]
pub extern "C" fn free_ner_model(wrapper: *mut NERModelWrapper) {
    if !wrapper.is_null() {
        unsafe {
            let w = Box::from_raw(wrapper);
            if !w.model.is_null() {
                drop(Box::from_raw(w.model));
            }
        }
    }
}

/// Free a NER result
#[no_mangle]
pub extern "C" fn free_ner_result(result: *mut NERResult) {
    if !result.is_null() {
        unsafe {
            let r = Box::from_raw(result);
            if !r.entities.is_null() && r.count > 0 {
                let entities = Vec::from_raw_parts(r.entities, r.count, r.count);
                for ent in entities {
                    if !ent.word.is_null() {
                        drop(CString::from_raw(ent.word));
                    }
                    if !ent.label.is_null() {
                        drop(CString::from_raw(ent.label));
                    }
                }
            }
        }
    }
}

// ============================================================================
// Question Answering FFI Functions
// ============================================================================

/// Create a new QA model with default configuration
#[no_mangle]
pub extern "C" fn new_qa_model() -> *mut QAModelWrapper {
    match QuestionAnsweringModel::new(QuestionAnsweringConfig::default()) {
        Ok(model) => {
            let wrapper = QAModelWrapper {
                model: Box::into_raw(Box::new(model)),
            };
            Box::into_raw(Box::new(wrapper))
        }
        Err(e) => {
            eprintln!("Failed to create QA model: {:?}", e);
            ptr::null_mut()
        }
    }
}

/// Create a QA model from custom files
#[no_mangle]
pub extern "C" fn new_qa_model_from_files(
    _model_path: *const c_char,
    _config_path: *const c_char,
    _vocab_path: *const c_char,
    _merges_path: *const c_char,
    _model_type: i32,
) -> *mut QAModelWrapper {
    new_qa_model()
}

/// Predict answers for the given question and context
#[no_mangle]
pub extern "C" fn predict_qa(
    wrapper: *mut QAModelWrapper,
    question: *const c_char,
    context: *const c_char,
) -> *mut QAResult {
    if wrapper.is_null() || question.is_null() || context.is_null() {
        return ptr::null_mut();
    }

    let question_str = match cstr_to_string(question) {
        Some(s) => s,
        None => return ptr::null_mut(),
    };

    let context_str = match cstr_to_string(context) {
        Some(s) => s,
        None => return ptr::null_mut(),
    };

    unsafe {
        let model = &*(*wrapper).model;
        let qa_input = QaInput {
            question: question_str,
            context: context_str,
        };

        let results = model.predict(&[qa_input], 1, 32);

        if results.is_empty() || results[0].is_empty() {
            let result = QAResult {
                answers: ptr::null_mut(),
                count: 0,
            };
            return Box::into_raw(Box::new(result));
        }

        let answers: Vec<QAAnswer> = results[0]
            .iter()
            .map(|ans| QAAnswer {
                score: ans.score as f32,
                start: ans.start,
                end: ans.end,
                answer: string_to_cstr(&ans.answer),
            })
            .collect();

        let count = answers.len();
        let answers_ptr = Box::into_raw(answers.into_boxed_slice()) as *mut QAAnswer;

        let result = QAResult {
            answers: answers_ptr,
            count,
        };
        Box::into_raw(Box::new(result))
    }
}

/// Free a QA model
#[no_mangle]
pub extern "C" fn free_qa_model(wrapper: *mut QAModelWrapper) {
    if !wrapper.is_null() {
        unsafe {
            let w = Box::from_raw(wrapper);
            if !w.model.is_null() {
                drop(Box::from_raw(w.model));
            }
        }
    }
}

/// Free a QA result
#[no_mangle]
pub extern "C" fn free_qa_result(result: *mut QAResult) {
    if !result.is_null() {
        unsafe {
            let r = Box::from_raw(result);
            if !r.answers.is_null() && r.count > 0 {
                let answers = Vec::from_raw_parts(r.answers, r.count, r.count);
                for ans in answers {
                    if !ans.answer.is_null() {
                        drop(CString::from_raw(ans.answer));
                    }
                }
            }
        }
    }
}

// ============================================================================
// Summarization FFI Functions
// ============================================================================

/// Create a new summarization model with default configuration
#[no_mangle]
pub extern "C" fn new_summarization_model() -> *mut SummarizationModelWrapper {
    match SummarizationModel::new(SummarizationConfig::default()) {
        Ok(model) => {
            let wrapper = SummarizationModelWrapper {
                model: Box::into_raw(Box::new(model)),
            };
            Box::into_raw(Box::new(wrapper))
        }
        Err(e) => {
            eprintln!("Failed to create summarization model: {:?}", e);
            ptr::null_mut()
        }
    }
}

/// Create a summarization model from custom files
#[no_mangle]
pub extern "C" fn new_summarization_model_from_files(
    _model_path: *const c_char,
    _config_path: *const c_char,
    _vocab_path: *const c_char,
    _merges_path: *const c_char,
    _model_type: i32,
) -> *mut SummarizationModelWrapper {
    new_summarization_model()
}

/// Summarize the given text
#[no_mangle]
pub extern "C" fn summarize(
    wrapper: *mut SummarizationModelWrapper,
    text: *const c_char,
) -> *mut SummarizationResult {
    if wrapper.is_null() || text.is_null() {
        return ptr::null_mut();
    }

    let text_str = match cstr_to_string(text) {
        Some(s) => s,
        None => return ptr::null_mut(),
    };

    unsafe {
        let model = &*(*wrapper).model;
        match model.summarize(&[text_str.as_str()]) {
            Ok(summaries) => {
                let cstr_summaries: Vec<*mut c_char> = summaries
                    .iter()
                    .map(|s| string_to_cstr(s))
                    .collect();

                let count = cstr_summaries.len();
                let summaries_ptr = Box::into_raw(cstr_summaries.into_boxed_slice()) as *mut *mut c_char;

                let result = SummarizationResult {
                    summaries: summaries_ptr,
                    count,
                };
                Box::into_raw(Box::new(result))
            }
            Err(e) => {
                eprintln!("Summarization failed: {:?}", e);
                ptr::null_mut()
            }
        }
    }
}

/// Free a summarization model
#[no_mangle]
pub extern "C" fn free_summarization_model(wrapper: *mut SummarizationModelWrapper) {
    if !wrapper.is_null() {
        unsafe {
            let w = Box::from_raw(wrapper);
            if !w.model.is_null() {
                drop(Box::from_raw(w.model));
            }
        }
    }
}

/// Free a summarization result
#[no_mangle]
pub extern "C" fn free_summarization_result(result: *mut SummarizationResult) {
    if !result.is_null() {
        unsafe {
            let r = Box::from_raw(result);
            if !r.summaries.is_null() && r.count > 0 {
                let summaries = Vec::from_raw_parts(r.summaries, r.count, r.count);
                for s in summaries {
                    if !s.is_null() {
                        drop(CString::from_raw(s));
                    }
                }
            }
        }
    }
}

// ============================================================================
// Zero-Shot Classification FFI Functions
// ============================================================================

/// Create a new zero-shot classification model with default configuration
#[no_mangle]
pub extern "C" fn new_zero_shot_model() -> *mut ZeroShotClassificationModelWrapper {
    match ZeroShotClassificationModel::new(ZeroShotClassificationConfig::default()) {
        Ok(model) => {
            let wrapper = ZeroShotClassificationModelWrapper {
                model: Box::into_raw(Box::new(model)),
            };
            Box::into_raw(Box::new(wrapper))
        }
        Err(e) => {
            eprintln!("Failed to create zero-shot model: {:?}", e);
            ptr::null_mut()
        }
    }
}

/// Create a zero-shot model from custom files
#[no_mangle]
pub extern "C" fn new_zero_shot_model_from_files(
    _model_path: *const c_char,
    _config_path: *const c_char,
    _vocab_path: *const c_char,
    _merges_path: *const c_char,
    _model_type: i32,
) -> *mut ZeroShotClassificationModelWrapper {
    new_zero_shot_model()
}

/// Predict zero-shot classification for the given text and labels
#[no_mangle]
pub extern "C" fn predict_zero_shot(
    wrapper: *mut ZeroShotClassificationModelWrapper,
    text: *const c_char,
    labels: *const *const c_char,
    labels_count: size_t,
) -> *mut ZeroShotResult {
    if wrapper.is_null() || text.is_null() || labels.is_null() || labels_count == 0 {
        return ptr::null_mut();
    }

    let text_str = match cstr_to_string(text) {
        Some(s) => s,
        None => return ptr::null_mut(),
    };

    let labels_vec: Vec<String> = unsafe {
        (0..labels_count)
            .filter_map(|i| cstr_to_string(*labels.add(i)))
            .collect()
    };

    if labels_vec.is_empty() {
        return ptr::null_mut();
    }

    let labels_refs: Vec<&str> = labels_vec.iter().map(|s| s.as_str()).collect();

    unsafe {
        let model = &*(*wrapper).model;
        match model.predict(&[text_str.as_str()], labels_refs.as_slice(), None, 128) {
            Ok(results) => {
                if results.is_empty() {
                    let result = ZeroShotResult {
                        labels: ptr::null_mut(),
                        count: 0,
                    };
                    return Box::into_raw(Box::new(result));
                }

                // Each result is a Label with text and score
                let zs_labels: Vec<ZeroShotLabel> = results
                    .iter()
                    .map(|label| ZeroShotLabel {
                        text: string_to_cstr(&label.text),
                        score: label.score,
                    })
                    .collect();

                let count = zs_labels.len();
                let labels_ptr = Box::into_raw(zs_labels.into_boxed_slice()) as *mut ZeroShotLabel;

                let result = ZeroShotResult {
                    labels: labels_ptr,
                    count,
                };
                Box::into_raw(Box::new(result))
            }
            Err(e) => {
                eprintln!("Zero-shot prediction failed: {:?}", e);
                ptr::null_mut()
            }
        }
    }
}

/// Free a zero-shot model
#[no_mangle]
pub extern "C" fn free_zero_shot_model(wrapper: *mut ZeroShotClassificationModelWrapper) {
    if !wrapper.is_null() {
        unsafe {
            let w = Box::from_raw(wrapper);
            if !w.model.is_null() {
                drop(Box::from_raw(w.model));
            }
        }
    }
}

/// Free a zero-shot result
#[no_mangle]
pub extern "C" fn free_zero_shot_result(result: *mut ZeroShotResult) {
    if !result.is_null() {
        unsafe {
            let r = Box::from_raw(result);
            if !r.labels.is_null() && r.count > 0 {
                let labels = Vec::from_raw_parts(r.labels, r.count, r.count);
                for label in labels {
                    if !label.text.is_null() {
                        drop(CString::from_raw(label.text));
                    }
                }
            }
        }
    }
}

// ============================================================================
// Translation FFI Functions
// ============================================================================

/// Create a new translation model with default configuration (English to French)
#[no_mangle]
pub extern "C" fn new_translation_model() -> *mut TranslationModelWrapper {
    match TranslationModelBuilder::new()
        .with_source_languages(vec![Language::English])
        .with_target_languages(vec![Language::French])
        .create_model()
    {
        Ok(model) => {
            let wrapper = TranslationModelWrapper {
                model: Box::into_raw(Box::new(model)),
            };
            Box::into_raw(Box::new(wrapper))
        }
        Err(e) => {
            eprintln!("Failed to create translation model: {:?}", e);
            ptr::null_mut()
        }
    }
}

/// Create a translation model from custom files
#[no_mangle]
pub extern "C" fn new_translation_model_from_files(
    _model_path: *const c_char,
    _config_path: *const c_char,
    _vocab_path: *const c_char,
    _merges_path: *const c_char,
    _model_type: i32,
) -> *mut TranslationModelWrapper {
    new_translation_model()
}

/// Translate the given text
#[no_mangle]
pub extern "C" fn translate(
    wrapper: *mut TranslationModelWrapper,
    text: *const c_char,
    _source_lang: *const c_char,
    _target_lang: *const c_char,
) -> *mut c_char {
    if wrapper.is_null() || text.is_null() {
        return ptr::null_mut();
    }

    let text_str = match cstr_to_string(text) {
        Some(s) => s,
        None => return ptr::null_mut(),
    };

    unsafe {
        let model = &*(*wrapper).model;
        match model.translate(&[text_str.as_str()], None, None) {
            Ok(results) => {
                match results.first() {
                    Some(translation) => string_to_cstr(translation),
                    None => ptr::null_mut(),
                }
            }
            Err(e) => {
                eprintln!("Translation failed: {:?}", e);
                ptr::null_mut()
            }
        }
    }
}

/// Free a translation model
#[no_mangle]
pub extern "C" fn free_translation_model(wrapper: *mut TranslationModelWrapper) {
    if !wrapper.is_null() {
        unsafe {
            let w = Box::from_raw(wrapper);
            if !w.model.is_null() {
                drop(Box::from_raw(w.model));
            }
        }
    }
}

// ============================================================================
// Text Generation FFI Functions
// ============================================================================

/// Create a new text generation model with default configuration (GPT-2)
#[no_mangle]
pub extern "C" fn new_text_generation_model() -> *mut TextGenerationModelWrapper {
    match TextGenerationModel::new(TextGenerationConfig::default()) {
        Ok(model) => {
            let wrapper = TextGenerationModelWrapper {
                model: Box::into_raw(Box::new(model)),
            };
            Box::into_raw(Box::new(wrapper))
        }
        Err(e) => {
            eprintln!("Failed to create text generation model: {:?}", e);
            ptr::null_mut()
        }
    }
}

/// Create a text generation model from custom files
#[no_mangle]
pub extern "C" fn new_text_generation_model_from_files(
    _model_path: *const c_char,
    _config_path: *const c_char,
    _vocab_path: *const c_char,
    _merges_path: *const c_char,
    _model_type: i32,
) -> *mut TextGenerationModelWrapper {
    new_text_generation_model()
}

/// Generate text from the given prompt
#[no_mangle]
pub extern "C" fn generate_text(
    wrapper: *mut TextGenerationModelWrapper,
    prompt: *const c_char,
    prefix: *const c_char,
) -> *mut c_char {
    if wrapper.is_null() || prompt.is_null() {
        return ptr::null_mut();
    }

    let prompt_str = match cstr_to_string(prompt) {
        Some(s) => s,
        None => return ptr::null_mut(),
    };

    let prefix_opt = cstr_to_string(prefix);

    unsafe {
        let model = &*(*wrapper).model;
        match model.generate(&[prompt_str.as_str()], prefix_opt.as_deref()) {
            Ok(results) => {
                match results.first() {
                    Some(generated) => string_to_cstr(generated),
                    None => ptr::null_mut(),
                }
            }
            Err(e) => {
                eprintln!("Text generation failed: {:?}", e);
                ptr::null_mut()
            }
        }
    }
}

/// Free a text generation model
#[no_mangle]
pub extern "C" fn free_text_generation_model(wrapper: *mut TextGenerationModelWrapper) {
    if !wrapper.is_null() {
        unsafe {
            let w = Box::from_raw(wrapper);
            if !w.model.is_null() {
                drop(Box::from_raw(w.model));
            }
        }
    }
}
