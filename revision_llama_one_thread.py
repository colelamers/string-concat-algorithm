#!/usr/bin/env python3
import subprocess
import time
import os
import sys
import re
import threading
import string
import math

# Load Environment variables. This is hardcoded at the moment.
# Intellij Environment variables:
# PYTHONUNBUFFERED=1;LD_LIBRARY_PATH=../../PycharmProjects/AI_TTS/whisper.cpp-master/build/src:../../PycharmProjects/AI_TTS/whisper.cpp-master/build/ggml/src:../../PycharmProjects/AI_TTS/whisper.cpp-master/build/ggml/src/ggml-cuda:../../PycharmProjects/AI_TTS/llama.cpp-master/build/bin:../../PycharmProjects/AI_TTS/piper-master/build/pi/lib
# Script Method:
# LD_LIBRARY_PATH=../../PycharmProjects/AI_TTS/whisper.cpp-master/build/src:../../PycharmProjects/AI_TTS/whisper.cpp-master/build/ggml/src:../../PycharmProjects/AI_TTS/whisper.cpp-master/build/ggml/src/ggml-cuda:../../PycharmProjects/AI_TTS/llama.cpp-master/build/bin:../../PycharmProjects/AI_TTS/piper-master/build/pi/lib:$LD_LIBRARY_PATH python3 llama_one_thread.py

_threshold_base = .15
VERY_LOW_THRESHOLD = max(0.0, min(_threshold_base * 1, 1.0)) # Has similar words and maybe length but dissimilar
LOW_THRESHOLD = max(0.0, min(_threshold_base * 2, 1.0)) # Close as a match, likely just a revision of text
MED_THRESHOLD = max(0.0, min(_threshold_base * 3, 1.0)) # Likely match, some possibly it's a false positive
HIGH_THRESHOLD = max(0.0, min(_threshold_base * 4, 1.0)) # Very probable as a match, low false positive potential
VERY_HIGH_THRESHOLD = max(0.0, min(_threshold_base * 5, 1.0)) # Extremely probable that it is a match

ld_paths = [
    "../../PycharmProjects/AI_TTS/whisper.cpp-master/build/src",
    "../../PycharmProjects/AI_TTS/whisper.cpp-master/build/ggml/src",
    "../../PycharmProjects/AI_TTS/whisper.cpp-master/build/ggml/src/ggml-cuda",
    "../../PycharmProjects/AI_TTS/llama.cpp-master/build/bin",
    "../../PycharmProjects/AI_TTS/piper-master/build/pi/lib"
]
current_ld = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = ":".join(ld_paths + [current_ld])
SIMILARITY_THRESHOLD = .75

ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
whisper_fill_ins_to_skip = (
    "",
    " ",
    " .",
    " . ",
    " .\n",
    " .\x1b[2K\n",
    "(clapping)",
    "(explosion)",
    "(explosion)\x1b[2K",
    "(explosions)",
    "(explosions)\x1b[2K",
    "(fire crackling)",
    "(giggles)",
    "(humming)",
    "(laughing)",
    "(mimics whooshing)",
    "(water",
    "(water splashing)",
    "[2K",
    "[2K]",
    "[BLANK_AUDIO]",
    "[BLANK_AUDIO]\x1b[2K",
    "[MUSIC]",
    "*sad",
    "music*",
    "[SPEAKING",
    "[Start",
    "*laughs*",
    "*Loud",
    "noise*",
    ".",
    "Device",
    "Device 0:",
    "ggml_cuda_init:",
    "init:",
    "main:",
    "whisper_backend_init_gpu:",
    "whisper_init_from_file_with_params_no_state:",
    "whisper_init_state:",
    "whisper_init_with_params_no_state:",
    "whisper_model_load:",
    "\x1b",
    "\x1b[2K",
    "[",
    "Silence",
    "]"
)

def whisper_init(model):
    cmd = [
        "./whisper.cpp-master/build/bin/whisper-stream",
        "-m", "whisper.cpp-master/models/ggml-large-v3-turbo-q8_0.bin",
        #"-m", "whisper.cpp-master/models/ggml-medium.en-q5_0.bin",
        "-t", "8",
    ]

    if model.lower() == "de":
        cmd += ["-l", "de"]

    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

def llama_init():
    return subprocess.Popen([
            "./llama.cpp-master/build/bin/llama-cli",
            "-m", "llama.cpp-master/models/7b-Domain-RL-Meta.Q3_K_M.gguf",
            "--n-gpu-layers", "16",
            "--top-k", "40",
            "--top-p", "0.9",
            "--n-predict", "300",
            #"--no-warmup",
            "--no-context-shift",
            "-c", "500",
            "--interactive"
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

def piper_init(model):
    cur_model = "en_US-ryan-high.onnx"
    if model.lower() == "de":
        cur_model = "de_DE-thorsten_emotional-medium.onnx"

    return subprocess.Popen([
            "./piper-master/build/piper",
            "--model", "./piper-master/models/" + cur_model,
            "--output-raw"
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=False
    )

def aplay_init(piper_proc):
    return subprocess.Popen([
            "aplay",
            "-r", "22050",
            "-f", "S16_LE",
            "-t", "raw",
            "-"
        ],
        stdin=piper_proc.stdout,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=False
    )

def tail_last_line(filename):
    try:
        return subprocess.check_output(['tail', '-n', '1', filename], text=True).strip()
    except subprocess.CalledProcessError:
        return ""

def strip_markdown(reply):
    reply = re.sub(r'\*\*(.*?)\*\*', r'\1', reply)
    reply = re.sub(r'\*(.*?)\*', r'\1', reply)
    reply = re.sub(r'__(.*?)__', r'\1', reply)
    reply = re.sub(r'^\s*\d+\.\s+', '', reply, flags=re.MULTILINE)
    reply = re.sub(r'^\s*[-*]\s+', '', reply, flags=re.MULTILINE)
    reply = re.sub(r'^\s*#{1,6}\s+', '', reply, flags=re.MULTILINE)
    reply = re.sub(r'^\s*:\s*', '', reply, flags=re.MULTILINE)
    return reply.strip()

def parse_for_end_sentence(reply):
    reply = reply.replace("> ", "").replace("EOF by user", "").strip()
    reply = strip_markdown(reply)

    parts = re.split(r'(?<=[.?!])\s+', reply)
    sentences = ""
    for p in parts:
        p_clean = p.strip()
        if len(p_clean) >= 10 and len(p_clean.split()) >= 3 and not p_clean.endswith(":"):
            sentences += " " + p_clean
            if len(sentences.strip()) > 200:
                break
    return sentences.strip()

def print_out_call_response(call, response):
    print("\n----------------------------------------------------------------------------------------------")
    print("Speaker: " + call)
    print("---------------------------------------------------------------------------------------------")
    print("AI: " + response)
    print("---------------------------------------------------------------------------------------------")

def tokenize(words):
    text = " ".join(words).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return set(re.findall(r'\b\w+\b', text))

def jaccard_similarity(list_str_1, list_str_2):
    set1 = tokenize(list_str_1)
    set2 = tokenize(list_str_2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return 0.0 if union == 0 else intersection / union

def cosine_similarity(list_str_1, list_str_2):
    set1 = tokenize(list_str_1)
    set2 = tokenize(list_str_2)
    intersection = len(set1 & set2)
    len1 = len(set1)
    len2 = len(set2)
    return 0.0 if len1 == 0 or len2 == 0 else intersection / math.sqrt(len1 * len2)

def dice_coefficient(list_str_1, list_str_2):
    set1 = tokenize(list_str_1)
    set2 = tokenize(list_str_2)
    intersection = len(set1 & set2)
    total = len(set1) + len(set2)
    return 0.0 if total == 0 else 2 * intersection / total

def rouge_l(list_str_1, list_str_2):
    m = len(list_str_1)
    n = len(list_str_2)

    # Initialize LCS table
    lcs_table = []
    for i in range(m + 1):
        row = []
        for j in range(n + 1):
            row.append(0)
        lcs_table.append(row)

    # Fill LCS table
    for i in range(m):
        for j in range(n):
            if list_str_1[i].lower() == list_str_2[j].lower():
                lcs_table[i + 1][j + 1] = lcs_table[i][j] + 1
            else:
                lcs_table[i + 1][j + 1] = max(lcs_table[i][j + 1], lcs_table[i + 1][j])

    lcs_length = lcs_table[m][n]

    # Compute precision, recall, and F1
    precision = lcs_length / max(1, len(list_str_2))
    recall = lcs_length / max(1, len(list_str_1))
    f1 = precision + recall
    if f1 > 0:
        return (2 * precision * recall) / f1
    else:
        return 0.0

# Min window size of 4 because it's common to have two similar words but not
# be an actual similar section. So we want the window to be similar
# Tells us if two strings have any overlap, and if so, how much.
def sliding_window_comparison(prev_list, cur_list, is_jaccard, prev_win_size=4, cur_win_size=4):
    if not prev_list and not cur_list:
        return []

    prev_win_size = min(prev_win_size, len(prev_list)) if len(prev_list) > 1 else 1
    cur_win_size = min(cur_win_size, len(cur_list)) if len(cur_list) > 1 else 1
    window_matches_found = 0

    for i in range(len(prev_list) - prev_win_size + 1):
        for j in range(len(cur_list) - cur_win_size + 1):
            if is_jaccard:
                sim = jaccard_similarity(prev_list[i:i + prev_win_size], cur_list[j:j + cur_win_size])
            else:
                # This helps with comparing lengths of lists
                sim = cosine_similarity(prev_list[i:i + prev_win_size], cur_list[j:j + cur_win_size])
            # Can only be 1 per ith iteration. So even if it's a false
            # positive, because a singular false positive isn't enough
            # to throw it off and we can count it.
            # .75 AS THE SIMILARITY THRESHOLD
            if sim >= .75:
                window_matches_found += 1
                break

    # If high matches found, extremely similar.
    # If low matches found, implies it could be similar or there was an overlap.
    # If it's 0, it does not have an exact match anywhere within the windows.
    return window_matches_found / max(1, len(prev_list) - prev_win_size + 1)

def find_matching_indexes(prev_list, next_list):
    j_match_lists = [[] for _ in range(len(prev_list))]
    i_matches = []
    for i in range(len(prev_list)):
        found = False
        for j in range(len(next_list)):
            #print(f"j-{j}:{next_list[j]}")
            #print(f"i-{i}:{prev_list[i]}")
            if next_list[j].lower() == prev_list[i].lower():
                j_match_lists[i].append(j)
                found = True
        if found:
            i_matches.append(i)
    return j_match_lists, i_matches

# This function finds the longest streak of numbers that go up by 1 (like 5, 6, 7, 8),
# where each number in the streak comes from a different list.
# The order of the lists is based on `prev_matches`, and the lists themselves are in `new_matches`.
# It returns the start and end positions (from `prev_matches`) of where the longest streak begins and ends.
# If no streak is found, it returns (-1, -1).
def find_strict_consecutive_chain(prev_matches, new_matches):
    longest_chain_length = 0
    longest_chain_start = -1
    longest_chain_end = -1

    # Loop through each starting index in prev_matches
    for i in range(len(prev_matches)):
        prev_index = prev_matches[i]

        # Try every starting value in the current new_matches[prev_index] list
        for j in new_matches[prev_index]:
            current_chain_length = 1
            current_value = j

            # Attempt to extend the chain as long as consecutive values continue
            for k in range(1, len(prev_matches) - i):
                next_prev_index = prev_matches[i + k]
                next_values = new_matches[next_prev_index]

                # Check if the next consecutive value exists in the next list
                if current_value + 1 in next_values:
                    current_value += 1
                    current_chain_length += 1
                else:
                    break

            if current_chain_length > longest_chain_length:
                longest_chain_length = current_chain_length
                longest_chain_start = prev_matches[i]
                longest_chain_end = prev_matches[i + current_chain_length - 1]

    if longest_chain_length > 0:
        return (longest_chain_start, longest_chain_end)
    else:
        return (-1, -1)

# Min window size of 4 because it's common to have two similar words but not
# be an actual similar section. So we want the window to be similar
# Tells us if two strings have any overlap, and if so, how much.
def sliding_window_comparison(prev_list, cur_list, is_jaccard, prev_win_size=4, cur_win_size=4):
    if not prev_list and not cur_list:
        return []

    prev_win_size = min(prev_win_size, len(prev_list)) if len(prev_list) > 1 else 1
    cur_win_size = min(cur_win_size, len(cur_list)) if len(cur_list) > 1 else 1
    window_matches_found = 0

    for i in range(len(prev_list) - prev_win_size + 1):
        for j in range(len(cur_list) - cur_win_size + 1):
            if is_jaccard:
                sim = jaccard_similarity(prev_list[i:i + prev_win_size], cur_list[j:j + cur_win_size])
            else:
                # This helps with comparing lengths of lists
                sim = cosine_similarity(prev_list[i:i + prev_win_size], cur_list[j:j + cur_win_size])
            # Can only be 1 per ith iteration. So even if it's a false
            # positive, because a singular false positive isn't enough
            # to throw it off and we can count it.
            #
            if sim >= VERY_HIGH_THRESHOLD:
                window_matches_found += 1
                break

    # If high matches found, extremely similar.
    # If low matches found, implies it could be similar or there was an overlap.
    # If it's 0, it does not have an exact match anywhere within the windows.
    return window_matches_found / max(1, len(prev_list) - prev_win_size + 1)

def find_matching_indexes(prev_list, next_list):
    j_match_lists = [[] for _ in range(len(prev_list))]
    i_matches = []
    for i in range(len(prev_list)):
        found = False
        for j in range(len(next_list)):
            #print(f"j-{j}:{next_list[j]}")
            #print(f"i-{i}:{prev_list[i]}")
            if next_list[j].lower() == prev_list[i].lower():
                j_match_lists[i].append(j)
                found = True
        if found:
            i_matches.append(i)
    return j_match_lists, i_matches

# Used for matching index evaluation between prev and next lists
# The purpose of this lets me find the lower and upper index of
# the similar text between both lists and so I can easily append
# over the appropriate indexes.
def find_strict_consecutive_chain(prev_matches, new_matches):
    longest_chain_len = 0
    longest_chain_range = None

    for start in range(len(prev_matches)):
        prev_index = prev_matches[start]
        for i in new_matches[prev_index]:
            chain_len = 1
            current_val = i

            for j in range(1, len(prev_matches) - start):
                next_prev_index = prev_matches[start + j]
                next_matches = new_matches[next_prev_index]

                if current_val + 1 in next_matches:
                    current_val += 1
                    chain_len += 1
                else:
                    break

            if chain_len > longest_chain_len:
                longest_chain_len = chain_len
                longest_chain_range = (
                    prev_matches[start],
                    prev_matches[start + chain_len - 1]
                )

    return longest_chain_range if longest_chain_range else (-1, -1)
def print_to_file(str):
    with open("temp_file.txt", "a") as file:
        print(str, file=file)
def whisper_readin_process(whisper_proc):
    empty_count = 0
    concat_list = []
    prev_sentence = []
    # Whisper read in process
    while True:
        line = whisper_proc.stdout.readline()
        clean_text = ansi_escape.sub('', line).strip()
        new_sentence = [part for part in clean_text.split(" ") if part.strip()]

        # 5 is arbitrary, 3 seems too quick
        if empty_count > 5:
            break

        if re.search(r'[\u4e00-\u9fff]', clean_text):
            empty_count += 1
            continue
        # No one word inputs because whisper typically contains lots of
        # "ghost" words detected of only 1 word/array length
        if len(new_sentence) <= 1:
            empty_count += 1
            time.sleep(1)
            continue

        # Issue with Whisper "Thank you." as a ghost sound picked up even
        # though nothing is said
        if len(new_sentence) == 2:
            if new_sentence[0] == "Thank" and new_sentence[1] == "you.":
                empty_count += 1
                time.sleep(1)
                continue

        # Part of some default output from whisper, just skip
        if new_sentence[0] in whisper_fill_ins_to_skip:
            continue

        print_to_file(f"\nconcat_list:   {concat_list}")
        print_to_file(f"prev_sentence: {prev_sentence}")
        print_to_file(f"new_sentence:  {new_sentence}")
        jaccard = sliding_window_comparison(prev_sentence, new_sentence, True, 4, 4)  # True == Jaccard
        cosine = sliding_window_comparison(prev_sentence, new_sentence, False, 4, 4)  # False == Cosine
        rogue = rouge_l(prev_sentence, new_sentence)
        dice = dice_coefficient(prev_sentence, new_sentence)
        j_match_lists, i_matches = find_matching_indexes(prev_sentence, new_sentence)

        # Really only for debugging
        # print(f"prev_sentence matches: {i_matches}") # prev_sentence indexes found
        # print(f"new_sentence matches:  {j_match_lists}") # split_setnence indexes
        print_to_file(f"jaccard: {jaccard}")  # Semi-fuzzy match  ignores order
        print_to_file(f"cosine:  {cosine}")  # fuzzy on match    fuzzy on length    ignores order
        print_to_file(f"dice:  {dice}")  # Do the contents overl
        print_to_file(f"rogue_l:  {rogue}")

        #          | Fuzziness on Matching	   | Fuzziness on Order | Sensitivity to Length	         | Notes
        # ---------------------------------------------------------------------------------------------------------
        # Jaccard  | Medium (semi-strict)	   | Ignores order	    | High (based on union)	         | Penalizes non-overlap strongly
        # Cosine   | High (fuzzy + vectorized) | Ignores order	    | High (via vector length)	     | Measures alignment, not sequence
        # Dice	   | High (fuzzy)	           | Ignores order	    | Medium (averages lengths)	     | Counts overlap, not order
        # ROUGE-L  | Medium (sequence-aware)   | Respects order	    | Medium (depends on LCS length) | Prefers long, ordered matches

        # If indexes > 0, then they might have similarities so check further.
        # Otherwise, just concatenate. No matching words heavily implies they are not similar.
        match_list_contain_data = (len(j_match_lists) > 0 and len(i_matches) > 0)

        # Low rogue_l but high everything else actually implies a false positive!
        # Basically, they have all matching words, but contextually they could be totally
        # incoherrent. rogue_l catches for this. Basically jaccard-cosine-dice are high but very low
        # rogue_l means it is most definitely something that should be concatenated, not overwritten.
        rogue_false_positive = (jaccard > MED_THRESHOLD
                                and cosine > HIGH_THRESHOLD
                                and dice > HIGH_THRESHOLD
                                and rogue < LOW_THRESHOLD)

        # Basically there is enough similarity detected that it's not an exact match, but it's likely part of
        # speech that needs to be appended into one coherrent string
        append_at_location = (jaccard > VERY_LOW_THRESHOLD
                              and cosine > LOW_THRESHOLD
                              and dice > LOW_THRESHOLD
                              and rogue > LOW_THRESHOLD)

        # High match, basically this catches when whisper or a string is very close together and
        # we are assuming that the little difference was irrelevant because they are so similar.
        overwrite_prev_string = (jaccard > HIGH_THRESHOLD
                                 and cosine > VERY_HIGH_THRESHOLD
                                 and dice > VERY_HIGH_THRESHOLD
                                 and rogue > MED_THRESHOLD)

        # There happenes to be a matching first word and last word of both sentences, so
        # it's fairly safe to assume we can append on that index
        append_on_last_word = False
        if len(prev_sentence) > 0:
            append_on_last_word = prev_sentence[len(prev_sentence) - 1] == new_sentence[0]

        i_index = len(concat_list) - len(prev_sentence)

        if rogue_false_positive or not match_list_contain_data:
            # Concat, rogue caught similar tokens, but definitely not the same contextual string
            # or if nto that, the match list's dont' have any data
            concat_list += new_sentence
        elif overwrite_prev_string:  # highly probable match
            concat_list = concat_list[:i_index] + new_sentence[0:]
        elif append_at_location:  # very low probable match
            lower, upper = find_strict_consecutive_chain(i_matches, j_match_lists)
            concat_list = concat_list[:i_index] + (prev_sentence[:lower] + new_sentence[j_match_lists[lower][0]:])
        elif append_on_last_word:
            # Check last word of sentence matches first word of sentence to combine on ends
            concat_list = concat_list[:i_index] + (prev_sentence[:len(prev_sentence) - 1] + new_sentence[0:])
        else:
            concat_list += new_sentence
            # concat_list += new_sentence

        # Needed to monitor the last appended text so overrides
        # in indexes do not occur in know/valid parts of the sentence.
        # This prevents arbitrary percentage of size of string checks.
        # Then retain it when performing the jaccard check again.
        prev_sentence = new_sentence
        empty_count = 0
    # return " ".join(concat_list)
    return concat_list

def llama_readin_process(captured_text):
    # Reason for init and communicate on each function call is that
    # we can reliably control for potential errors or Out of Memory type of exceptions
    # that can occur. That way, if the sub process fails, just the function call
    # fails and you can try again. Otherwise, if it's a global process, there are
    # errors where the whole thing can fail and then is un-recoverable and requires
    # reinitialization
    llama_proc = llama_init()
    reply = ""
    try:
        print("CQ")
        stdout, stderr = llama_proc.communicate(input=captured_text, timeout=60)
        reply = stdout.strip()
    except subprocess.TimeoutExpired:
        if stderr:
            print(stderr)
        llama_proc.kill()
        print("LT (timeout)")
        time.sleep(1)

    print("AZ")
    reply = stdout.lstrip('\n').lstrip('> ').strip()
    if reply.startswith(captured_text.strip()):
        reply = reply[len(captured_text.strip()):]

    print("YZ")
    reply = parse_for_end_sentence(reply)
    print_out_call_response(captured_text, reply)
    llama_proc.kill()
    llama_proc.wait()

    return reply

def run_whisper_llama(model):
    whisper_proc = whisper_init(model)
    # Chat loop
    captured_text = []
    while True:
        if whisper_proc.poll() is None:
            # Process is still running
            pass
        else:
            # Process has exited or failed, reinitialize
            whisper_proc = whisper_init(model)

        t_text = whisper_readin_process(whisper_proc)
        #t_text = " ".join(t_text)
        captured_text += t_text
        print_to_file(f"Whisper-Append: {captured_text}")
'''
# This code is commented out only for the use of taking in live data perpetually without reading back
        print("CE")
        whisper_proc.terminate()
        whisper_proc.wait()
        whisper_proc.stdout.close()
        reply = llama_readin_process(captured_text)

        print("BA")
        with piper_init(model) as piper_proc:
            piper_proc.stdin.write(reply.encode("utf-8"))
            piper_proc.stdin.flush()
            piper_proc.stdin.close()
            print("BB")

            try:
                print("KA")
                with aplay_init(piper_proc) as aplay_proc:
                    aplay_proc.wait()
            except KeyboardInterrupt:
                print("Interrupted. Terminating processes.")
                aplay_proc.terminate()
                piper_proc.terminate()
            finally:
                piper_proc.stdout.close()
            print("BC")
            piper_proc.wait()
'''
if __name__ == "__main__":
    # todo 1; somehow kickoff whisper right away and then skip the
    #  first action and then capture the first "real input"
    piper_model_lang_code = sys.argv[1] if len(sys.argv) > 1 else "en"

    whisper_llama_thread = threading.Thread(target=run_whisper_llama, args=(piper_model_lang_code,), daemon=True)
    whisper_llama_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        sys.exit(0)
