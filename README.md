# 🤗 Hugging Face Pipelines – Complete Guide

Welcome to the **Hugging Face Pipelines Guide** 🚀  
This repository is a **comprehensive, step-by-step resource** covering **31 Hugging Face pipelines** with examples, models, training techniques, and best practices.  

Whether you are a **beginner** exploring NLP or a **researcher** building production-ready AI, this guide will help you master Hugging Face.

---

## 📘 1. Introduction

Hugging Face is one of the most popular ecosystems for **Natural Language Processing (NLP)**, **Computer Vision (CV)**, **Audio Processing**, and **Multimodal AI**.  

At the core of Hugging Face is the `pipeline()` API, which:  
- ✅ Simplifies inference by hiding preprocessing, tokenization, and post-processing  
- ✅ Provides **one-line access** to powerful models  
- ✅ Supports multiple domains: text, audio, image, and multimodal tasks  

Example of a simple pipeline:

```python
from transformers import pipeline

# Sentiment analysis in one line
classifier = pipeline("sentiment-analysis")
result = classifier("I love Hugging Face!")
print(result)
Output:

python
Copy code
[{'label': 'POSITIVE', 'score': 0.9998}]
⚙️ 2. Installation & Environment Setup
🔹 Step 1: Create a Conda Environment
It’s a good practice to keep Hugging Face in a separate environment:

bash
Copy code
conda create -n hf_pipelines python=3.10 -y
conda activate hf_pipelines
🔹 Step 2: Install Hugging Face Libraries
Install the core libraries:

bash
Copy code
pip install transformers datasets huggingface_hub
For GPU (PyTorch CUDA):

bash
Copy code
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
For TensorFlow users:

bash
Copy code
pip install tensorflow
For audio pipelines:

bash
Copy code
pip install librosa soundfile
For vision pipelines:

bash
Copy code
pip install pillow opencv-python
🔹 Step 3: Verify Installation
Run:

python
Copy code
from transformers import pipeline
print(pipeline("sentiment-analysis")("I love AI!"))
If it outputs a POSITIVE/NEGATIVE label, installation is successful ✅.

📊 3. Pipeline Parameters (General)
Every Hugging Face pipeline supports common parameters:

Category	Parameter	Description
Model	model	Model identifier or path
tokenizer	Custom tokenizer instance
device	-1 for CPU, 0+ for GPU
framework	"pt" (PyTorch) / "tf" (TensorFlow)
Processing	batch_size	Samples per batch
truncation	Cut long sequences
padding	Pad shorter sequences
max_length	Max input length
Generation	temperature	Randomness control
top_k	Top-k sampling
top_p	Nucleus sampling
repetition_penalty	Penalize repeated tokens
Output	return_all_scores	Return all scores
aggregation_strategy	For NER tasks
handle_impossible_answer	For QA tasks

📚 4. Pipelines
Below we cover all 31 Hugging Face pipelines with:

📌 Description (what & why)

🧩 Suggested models

🛠️ Training techniques

🏗️ Library support

⭐ Best choice recommendation

💻 Full code example

📝 1. Text Classification
What: Assign label(s) to text (e.g., positive/negative).
Why: Sentiment analysis, intent detection, moderation.

Suggested Models:

distilbert-base-uncased-finetuned-sst-2-english → Fast, lightweight

roberta-large-mnli → Strong zero-shot capabilities

bert-base-cased → Solid baseline for fine-tuning

Training Techniques: Fine-tuning, LoRA/PEFT, Zero-shot with NLI models
Libraries: PyTorch ✅ | TensorFlow ✅
Best Choice: distilbert for speed ⚡, roberta-large-mnli for flexibility

python
Copy code
from transformers import pipeline

clf = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1
)

texts = ["I love this product!", "The service was terrible."]
results = clf(texts, batch_size=8, truncation=True, padding=True, max_length=128)
print(results)
🎭 2. Token Classification (NER)
What: Identify entities in text (names, locations, orgs).
Why: Information extraction, knowledge graphs, search.

Suggested Models:

dbmdz/bert-large-cased-finetuned-conll03-english → Classic NER model

dslim/bert-base-NER → Lightweight, accurate

xlm-roberta-large-finetuned-conll03-english → Multilingual

Training Techniques: Fine-tuning on entity datasets (CoNLL-2003, OntoNotes)
Libraries: PyTorch ✅ | TensorFlow ✅
Best Choice: dslim/bert-base-NER for general, xlm-roberta for multilingual

python
Copy code
ner = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)

text = "Hugging Face is based in New York and was founded by Julien."
print(ner(text))
❓ 3. Question Answering
What: Extract answer span from context text.
Why: Chatbots, document QA, search.

Suggested Models:

distilbert-base-cased-distilled-squad → Fast & lightweight

bert-large-uncased-whole-word-masking-finetuned-squad → Strong accuracy

deepset/roberta-base-squad2 → Handles unanswerable questions

Training Techniques: Fine-tuning on SQuAD, domain adaptation
Libraries: PyTorch ✅ | TensorFlow ✅
Best Choice: roberta-base-squad2 for real-world QA

python
Copy code
qa = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2"
)

context = "Hugging Face is a company based in New York founded in 2016."
question = "Where is Hugging Face based?"
print(qa(question=question, context=context, handle_impossible_answer=True))
🧩 4. Fill-Mask
What: Predict masked word(s) in text.
Why: Cloze tasks, word suggestion, data augmentation.

Suggested Models:

bert-base-uncased → Classic fill-mask model

roberta-base → More accurate predictions

albert-base-v2 → Lightweight

Training Techniques: Pretraining (MLM), fine-tuning on domain text
Libraries: PyTorch ✅ | TensorFlow ✅
Best Choice: roberta-base for general-purpose MLM

python
Copy code
fill = pipeline("fill-mask", model="roberta-base")
print(fill("Hugging Face is creating <mask> for everyone."))
📖 5. Summarization
What: Generate a shorter version of text.
Why: Article/news summarization, reports.

Suggested Models:

facebook/bart-large-cnn → Best for news-style summaries

t5-small → Lightweight and fast

google/pegasus-xsum → Abstractive summaries

Training Techniques: Fine-tuning on summarization datasets (CNN/DailyMail, XSum)
Libraries: PyTorch ✅ | TensorFlow ✅
Best Choice: bart-large-cnn for balance of speed + quality

python
Copy code
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

text = """
Hugging Face is a company creating machine learning tools.
It is known for its Transformers library, which is widely used for NLP tasks.
"""
print(summarizer(text, max_length=40, min_length=10, do_sample=False))
🌍 6. Translation
What: Translate text between languages.
Why: Cross-lingual applications, chatbots, global reach.

Suggested Models:

Helsinki-NLP/opus-mt-en-de → English → German

Helsinki-NLP/opus-mt-mul-en → Multilingual → English

facebook/m2m100_418M → Many-to-many

Training Techniques: Fine-tuning on parallel corpora
Libraries: PyTorch ✅ | TensorFlow ✅
Best Choice: M2M100 for multilingual

python
Copy code
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-de")
print(translator("I love Hugging Face!", max_length=40))
📰 7. Text Generation
What: Generate free-form text.
Why: Story generation, chatbots, creative writing.

Suggested Models:

gpt2 → Classic text generation

EleutherAI/gpt-neo-1.3B → Strong open-source GPT

tiiuae/falcon-7b-instruct → Larger, better instruction following

Training Techniques: Pretraining, instruction-tuning, RLHF
Libraries: PyTorch ✅ | TensorFlow ✅
Best Choice: falcon-7b-instruct for modern tasks

python
Copy code
generator = pipeline("text-generation", model="gpt2")

print(generator("Hugging Face is", 
                max_length=50, 
                temperature=0.7, 
                top_k=50, 
                top_p=0.9))
🔍 8. Feature Extraction
What: Convert text into vector embeddings.
Why: Semantic search, clustering, similarity.

Suggested Models:

sentence-transformers/all-MiniLM-L6-v2 → Fast, small

bert-base-uncased → General embeddings

roberta-base → Higher accuracy

Training Techniques: Contrastive learning, SimCSE, fine-tuning on STS
Libraries: PyTorch ✅ | TensorFlow ✅
Best Choice: sentence-transformers/all-MiniLM-L6-v2

python
Copy code
extractor = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")
print(extractor("Hugging Face is awesome!", padding=True, truncation=True))
📊 9. Zero-Shot Classification
What: Classify text into categories without training.
Why: Quick prototyping, unseen label classification.

Suggested Models:

facebook/bart-large-mnli → Standard zero-shot model

joeddav/xlm-roberta-large-xnli → Multilingual

facebook/deberta-v3-large-mnli → Stronger accuracy

Training Techniques: NLI pretraining, transfer learning
Libraries: PyTorch ✅ | TensorFlow ✅
Best Choice: bart-large-mnli

python
Copy code
zero = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "I love to play football."
labels = ["sports", "politics", "technology"]
print(zero(text, candidate_labels=labels))
🎬 10. Video Classification
What: Classify videos into categories.
Why: Activity recognition, moderation, recommendation.

Suggested Models:

MCG-NJU/videomae-base-finetuned-kinetics → Strong activity classification

facebook/xclip-base-patch32 → Multimodal video understanding

google/vivit-b-16x2 → Transformer-based video

Training Techniques: Pretraining on Kinetics-400, fine-tuning
Libraries: PyTorch ✅
Best Choice: videomae-base for general use

python
Copy code
video_clf = pipeline("video-classification", model="MCG-NJU/videomae-base-finetuned-kinetics")
print(video_clf("example_video.mp4"))
---

## 🎨 11. Image Classification

**What:** Assign labels to images.  
**Why:** Object recognition, tagging, moderation.  

**Suggested Models:**  
- `google/vit-base-patch16-224` → Vision Transformer baseline  
- `microsoft/resnet-50` → Classic CNN  
- `facebook/convnext-base-224` → High accuracy  

**Training Techniques:** Fine-tuning on ImageNet, transfer learning  
**Libraries:** PyTorch ✅ | TensorFlow ✅  
**Best Choice:** `vit-base` for modern tasks  

```python
from transformers import pipeline

img_clf = pipeline("image-classification", model="google/vit-base-patch16-224")
print(img_clf("example.jpg"))
🖼️ 12. Image Segmentation
What: Classify each pixel in an image.
Why: Medical imaging, scene parsing, AR/VR.

Suggested Models:

facebook/detr-resnet-50-panoptic → Panoptic segmentation

nvidia/segformer-b0-finetuned-ade-512-512 → Efficient

facebook/mask2former-swin-base-coco → Advanced

Training Techniques: Fine-tuning on COCO, ADE20k
Libraries: PyTorch ✅
Best Choice: segformer for efficiency

python
Copy code
segmenter = pipeline("image-segmentation", model="nvidia/segformer-b0-finetuned-ade-512-512")
print(segmenter("example.jpg"))
🖼️➡️🖼️ 13. Image-to-Image
What: Transform input image into another image.
Why: Style transfer, image editing.

Suggested Models:

CompVis/stable-diffusion-v1-4 → General image generation

runwayml/stable-diffusion-inpainting → Inpainting/editing

lllyasviel/control_v11p_sd15_canny → Guided generation

Training Techniques: Diffusion model fine-tuning, DreamBooth, LoRA
Libraries: PyTorch ✅
Best Choice: stable-diffusion

python
Copy code
img2img = pipeline("image-to-image", model="CompVis/stable-diffusion-v1-4")
print(img2img("input.png", prompt="Convert this into Van Gogh style"))
🖼️➡️📝 14. Image-to-Text (Captioning)
What: Generate text description for images.
Why: Accessibility, search, recommendation.

Suggested Models:

nlpconnect/vit-gpt2-image-captioning → Popular captioning

Salesforce/blip-image-captioning-base → Stronger captions

microsoft/git-large-coco → Advanced

Training Techniques: Pretraining on image-text datasets (COCO, Flickr30k)
Libraries: PyTorch ✅
Best Choice: BLIP for modern captioning

python
Copy code
captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
print(captioner("example.jpg"))
🔑 15. Keypoint Matching
What: Match keypoints between two images.
Why: Object tracking, AR, 3D reconstruction.

Suggested Models:

naver-clova-ix/dino-vitb16 → Keypoint embeddings

facebook/superpoint → Lightweight feature extractor

google/matchformer → Advanced matching

Training Techniques: Contrastive training, supervised keypoint matching
Libraries: PyTorch ✅
Best Choice: dino-vitb16

python
Copy code
matcher = pipeline("keypoint-matching", model="naver-clova-ix/dino-vitb16")
print(matcher({"image": "img1.jpg"}, {"image": "img2.jpg"}))
🎭 16. Mask Generation
What: Predict masks for objects in images.
Why: Object segmentation, image editing.

Suggested Models:

facebook/sam-vit-base → Segment Anything Model (SAM)

facebook/mask2former-swin-base-coco → Advanced segmentation

facebook/detr-resnet-50 → Basic segmentation

Training Techniques: Weakly supervised training, large-scale pretraining
Libraries: PyTorch ✅
Best Choice: SAM

python
Copy code
masker = pipeline("mask-generation", model="facebook/sam-vit-base")
print(masker("example.jpg"))
🎯 17. Object Detection
What: Detect objects + bounding boxes in images.
Why: Surveillance, self-driving cars, retail analytics.

Suggested Models:

facebook/detr-resnet-50 → Transformer-based

hustvl/yolos-small → Lightweight

google/owlvit-base-patch32 → Zero-shot detection

Training Techniques: Fine-tuning on COCO, DETR training
Libraries: PyTorch ✅
Best Choice: detr-resnet-50 for general, owlvit for zero-shot

python
Copy code
detector = pipeline("object-detection", model="facebook/detr-resnet-50")
print(detector("example.jpg"))
🧠 18. Automatic Speech Recognition (ASR)
What: Convert speech to text.
Why: Transcription, voice assistants, accessibility.

Suggested Models:

openai/whisper-small → Multilingual, accurate

facebook/wav2vec2-base-960h → English ASR

jonatasgrosman/wav2vec2-large-xlsr-53-english → Strong accuracy

Training Techniques: CTC loss, fine-tuning on LibriSpeech, Whisper pretraining
Libraries: PyTorch ✅
Best Choice: Whisper

python
Copy code
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")
print(asr("example.wav"))
🎵 19. Audio Classification
What: Classify audio into categories.
Why: Music classification, environmental sounds.

Suggested Models:

superb/hubert-large-superb-ks → Keyword spotting

audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim → Emotion detection

MIT/ast-finetuned-audioset-10-10-0.4593 → Audio tagging

Training Techniques: Fine-tuning, contrastive audio training
Libraries: PyTorch ✅
Best Choice: HuBERT

python
Copy code
audio_clf = pipeline("audio-classification", model="superb/hubert-large-superb-ks")
print(audio_clf("example.wav"))
🗣️ 20. Text-to-Speech (TTS)
What: Convert text into spoken audio.
Why: Voice assistants, accessibility, chatbots.

Suggested Models:

facebook/fastspeech2-en-ljspeech → Fast TTS

espnet/kan-bayashi_ljspeech_vits → Natural voice

microsoft/speecht5_tts → Strong prosody control

Training Techniques: Tacotron2-style seq2seq, VITS training, neural vocoders
Libraries: PyTorch ✅
Best Choice: SpeechT5

python
Copy code
tts = pipeline("text-to-speech", model="microsoft/speecht5_tts")
audio = tts("Hello Hugging Face, this is a TTS test.")
with open("output.wav", "wb") as f:
    f.write(audio["audio"])
---

# 📊 21. Document Question Answering

**What:** Answer questions from scanned docs/PDFs.  
**Why:** Automating forms, contracts, receipts.  

**Suggested Models:**  
- `impira/layoutlm-document-qa` → LayoutLM  
- `microsoft/layoutlmv3-base` → Strong document understanding  
- `naver-clova-ix/donut-base` → OCR-free  

**Training Techniques:** Pretraining on DocVQA, FUNSD  
**Libraries:** PyTorch ✅  
**Best Choice:** `LayoutLMv3`  

```python
docqa = pipeline("document-question-answering", model="impira/layoutlm-document-qa")
print(docqa("invoice.pdf", question="What is the total amount?"))
📝 22. Text2Text Generation
What: Convert one text into another (translation, rewriting).
Why: Summarization, style transfer.

Suggested Models:

google/mt5-small → Multilingual text2text

facebook/bart-large → Strong English generation

t5-base → General-purpose

Training Techniques: Seq2seq with teacher forcing
Libraries: PyTorch ✅ | TensorFlow ✅
Best Choice: BART for English, mT5 for multilingual

python
Copy code
t2t = pipeline("text2text-generation", model="facebook/bart-large")
print(t2t("Translate this text to French: Hello World"))
📖 23. Conversational AI
What: Multi-turn dialogues with memory.
Why: Chatbots, customer support.

Suggested Models:

microsoft/DialoGPT-medium → Conversational baseline

facebook/blenderbot-400M-distill → Balanced chatbot

meta-llama/Llama-2-7b-chat-hf → Advanced dialogue

Training Techniques: RLHF, supervised fine-tuning
Libraries: PyTorch ✅
Best Choice: Blenderbot small, Llama-2-chat for advanced

python
Copy code
from transformers import pipeline, Conversation

chatbot = pipeline("conversational", model="facebook/blenderbot-400M-distill")
conv = Conversation("Hello, how are you?")
print(chatbot(conv))
🧾 24. Table Question Answering
What: Answer questions about structured tables.
Why: Data extraction, BI automation.

Suggested Models:

google/tapas-large-finetuned-wtq → TAPAS model

microsoft/tapas-base-finetuned-sqa → Small version

naver-clova-ix/tapas-base → Efficient

Training Techniques: Pretraining on WikiTableQuestions
Libraries: PyTorch ✅
Best Choice: TAPAS-large

python
Copy code
tableqa = pipeline("table-question-answering", model="google/tapas-large-finetuned-wtq")
print(tableqa(table={"Country":["France","Germany"],"Pop":[67,83]}, query="Which country has higher population?"))
🤖 25. Zero-Shot Image Classification
What: Classify images without training examples.
Why: Fast prototyping, open-vocab tasks.

Suggested Models:

openai/clip-vit-base-patch32 → CLIP model

laion/CLIP-ViT-B-16-laion2B-s34B-b79K → Large CLIP

google/owlvit-base-patch32 → Open-vocab

Training Techniques: Contrastive training (text-image pairs)
Libraries: PyTorch ✅
Best Choice: CLIP

python
Copy code
clip = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")
print(clip("dog.jpg", candidate_labels=["cat","dog","car"]))
🎤 26. Speaker Diarization
What: Detect “who spoke when” in audio.
Why: Meetings, transcription services.

Suggested Models:

pyannote/speaker-diarization → Best in class

speechbrain/spkrec-ecapa-voxceleb → Speaker recognition

voxceleb/diarization → Dataset-based

Training Techniques: Speaker embedding training, clustering
Libraries: PyTorch ✅
Best Choice: pyannote

python
Copy code
diarizer = pipeline("speaker-diarization", model="pyannote/speaker-diarization")
print(diarizer("meeting.wav"))
🧠 27. Feature Extraction
What: Extract embeddings/vectors from text or images.
Why: Semantic search, clustering, retrieval.

Suggested Models:

sentence-transformers/all-MiniLM-L6-v2 → Compact

bert-base-uncased → Classic

openai/clip-vit-base-patch32 → For vision+text

Training Techniques: Contrastive, triplet loss
Libraries: PyTorch ✅
Best Choice: all-MiniLM-L6-v2

python
Copy code
extractor = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")
print(extractor("Hugging Face is awesome!"))
🧪 28. Fill-Mask (Masked Language Modeling)
What: Predict missing words in a sentence.
Why: Pretraining, cloze tests.

Suggested Models:

bert-base-uncased → English

roberta-base → Stronger variant

distilroberta-base → Efficient

Training Techniques: Masked language modeling
Libraries: PyTorch ✅ | TensorFlow ✅
Best Choice: RoBERTa

python
Copy code
masker = pipeline("fill-mask", model="bert-base-uncased")
print(masker("The capital of France is [MASK]."))
🗄️ 29. Sentence Similarity
What: Compare similarity between two texts.
Why: Search, recommendations, deduplication.

Suggested Models:

sentence-transformers/all-MiniLM-L6-v2 → Popular

sentence-transformers/multi-qa-MiniLM-L6-cos-v1 → QA-focused

bert-base-nli-mean-tokens → Classic

Training Techniques: Contrastive, NLI-based fine-tuning
Libraries: PyTorch ✅
Best Choice: MiniLM

python
Copy code
sim = pipeline("sentence-similarity", model="sentence-transformers/all-MiniLM-L6-v2")
print(sim("I love machine learning", "Deep learning is great"))
⚡ 30. Translation
What: Translate text between languages.
Why: Localization, multilingual apps.

Suggested Models:

Helsinki-NLP/opus-mt-en-fr → English–French

facebook/m2m100_418M → 100+ languages

google/mt5-base → Multilingual

Training Techniques: Seq2seq translation training
Libraries: PyTorch ✅ | TensorFlow ✅
Best Choice: M2M100 for multilingual

python
Copy code
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
print(translator("Hello World", src="en", tgt="fr"))
🕵️ 31. Named Entity Recognition (NER)
What: Extract entities (names, dates, places).
Why: Information extraction, NLP pipelines.

Suggested Models:

dbmdz/bert-large-cased-finetuned-conll03-english → Strong NER

dslim/bert-base-NER → Efficient

xlm-roberta-large-finetuned-conll03-english → Multilingual

Training Techniques: Token classification, BIO tagging
Libraries: PyTorch ✅ | TensorFlow ✅
Best Choice: BERT-large-conll03

python
Copy code
ner = pipeline("ner", model="dslim/bert-base-NER")
print(ner("Hugging Face was founded in New York."))
📌 Summary Tables
✅ Pipelines vs Libraries
Pipeline Type	PyTorch	TensorFlow	ONNX
Text-based NLP 📝	✅	✅	✅
Vision (Images) 🖼️	✅	⚪	⚪
Audio 🎤	✅	⚪	⚪
Multimodal 🔗	✅	⚪	⚪

🏆 Best Models by Domain
Domain	Recommended Model	Why Best?
Text Classification 📝	distilbert-base-uncased	Efficient & strong
Summarization 📚	facebook/bart-large-cnn	State-of-art
Translation 🌍	facebook/m2m100_418M	Multilingual
ASR 🎤	openai/whisper-small	Accurate + multilingual
Image Classification 🖼️	google/vit-base-patch16-224	Transformer-based
Object Detection 🎯	facebook/detr-resnet-50	Transformer detection
NER 🕵️	dslim/bert-base-NER	High accuracy

🎉 Final Notes
🧰 Hugging Face Pipelines provide easy, production-ready APIs

⚡ Supports 30+ modalities (text, vision, audio, multimodal)

🏗️ Choose models wisely → balance speed, accuracy, memory

🔬 Fine-tuning & transfer learning improve results

💡 Always monitor bias, fairness, and ethics in AI
