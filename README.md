[Link to Video Presentation](https://www.youtube.com/watch?v=CHCZ0BLWR5Y)

# Multi-Metric Text Similarity Evaluation for Contextual Sentence Reconstruction
Cole Lamers
University of Wisconsin – Whitewater
Submitted in Partial Fulfillment
of the Requirements for the Degree of
Masters of Science in Computer Science
Advisor:
Dr. Hien Nguyen
08/11/2025

## Abstract
* This project presents a lightweight text appending and caption evaluation algorithm using lexical and
structural similarity metrics such as Jaccard Cosine Dice and ROUGE-L. The approach combines a
sliding window technique with fuzzy matching to balance local context with tolerance for transcription
noise. This is particularly effective in automatic speech recognition (ASR) outputs. The system
evaluates concatenated outputs against the intended output using ROUGE-L scoring as a means of
determining the overall accuracy. The algorithm implements a proportional and tiered threshold
modifier that allows the text concatenation to be adaptive to: appending overwriting ignoring or
detecting false-positives depending on the scores from the threshold values measured. Several test
cases were compared using a broad range of threshold modifiers. A real-time real-world test was also
conducted using Stephen Colbert’s White House Correspondents’ Dinner speech due to it's length and
the variations within the audio processed. This method demonstrates the potential for a real-time string
appending process that is fast adaptable accurate and deterministic without the computational overhead
of a full-scale LLM.

# Dependencies required for capturing audio to text, feeding to llm, and then responding back with audio
* [Whisper.cpp](https://github.com/ggml-org/whisper.cpp)
> Whisper.cpp is required if you intend on taking text input to parse with the algorithm.

* [llama.cpp](https://github.com/ggml-org/llama.cpp)
> llama.cpp is only used if you intend on feeding text from your whisper.cpp input to the llm.

* [Piper](https://github.com/OHF-Voice/piper1-gpl)
> Piper is only required if you intend on converting text to speech.
