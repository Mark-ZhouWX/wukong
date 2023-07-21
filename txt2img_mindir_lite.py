import os
import numpy as np
import unicodedata
import argparse
from functools import lru_cache
from PIL import Image
# import mindspore as ms
# import mindspore.nn as nn
import sys
import mindspore_lite as mslite
import cv2
import time


from test_task_rrdb import test_rrdb_om_srx4

workspace = os.path.dirname(os.path.abspath(__file__))
print("workspace", workspace, flush=True)
sys.path.append(workspace)


@lru_cache()
def default_wordpiece():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocab_zh.txt")


def is_control(char):
    """Checks whether `char` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def is_punctuation(char):
    """Checks whether `char` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (33 <= cp <= 47) or (58 <= cp <= 64) \
            or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def is_whitespace(char):
    """Checks whether `char` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
            (0x4E00 <= cp <= 0x9FFF)
            or (0x3400 <= cp <= 0x4DBF)  #
            or (0x20000 <= cp <= 0x2A6DF)  #
            or (0x2A700 <= cp <= 0x2B73F)  #
            or (0x2B740 <= cp <= 0x2B81F)  #
            or (0x2B820 <= cp <= 0x2CEAF)  #
            or (0xF900 <= cp <= 0xFAFF)
            or (0x2F800 <= cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)


class WordpieceTokenizer(object):
    def __init__(self, vocab_path: str = default_wordpiece()):
        with open(vocab_path) as vocab_file:
            vocab = [line.strip() for line in vocab_file]
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.max_input_chars_per_word = 100
        self.tokenize_chinese_chars = True
        self.unk_token = "[UNK]"
        
        SOT_TEXT = "<|startoftext|>"
        EOT_TEXT = "<|endoftext|>"
        CONTEXT_LEN = 77
        self.never_split = [self.unk_token, SOT_TEXT, EOT_TEXT]

    @staticmethod
    def __whitespace_tokenize(text):
        """Runs basic whitespace cleaning and splitting on a piece of text."""
        text = text.strip()
        if not text:
            return []
        tokens = text.split()
        return tokens

    def __split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        if self.never_split and text in self.never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    @staticmethod
    def __clean_text(text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or is_control(char):
                continue
            if is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    @staticmethod
    def __tokenize_chinese_chars(text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def __wordpiece_tokenize(self, text):
        output_tokens = []
        for token in self.__whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.encoder:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens

    def __basic_tokenize(self, text):
        # union() returns a new set by concatenating the two sets.
        text = self.__clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        if self.tokenize_chinese_chars:
            text = self.__tokenize_chinese_chars(text)
        orig_tokens = self.__whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if token not in self.never_split:
                token = token.lower()
                token = strip_accents(token)
            split_tokens.extend(self.__split_on_punc(token))
        output_tokens = self.__whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def text_tokenize(self, text):
        split_tokens = []
        for token in self.__basic_tokenize(text):
            if token in self.never_split:
                split_tokens.append(token)
            else:
                split_tokens += self.__wordpiece_tokenize(token)
        return split_tokens

    def encode(self, text):
        tokens = self.text_tokenize(text)
        return [self.encoder.get(token, self.unk_token) for token in tokens]

    def decode(self, tokens):
        segments = [self.decoder.get(token, self.unk_token)
                    for token in tokens]
        text = ""
        for segment in segments:
            if segment in self.never_split:
                text += segment
            else:
                text += segment.lstrip("##")
        return text


def tokenize(texts, tokenizer):

    if isinstance(texts, str):
        texts = [texts]
    
    SOT_TEXT = "[CLS]"
    EOT_TEXT = "[SEP]"
    CONTEXT_LEN = 77
    
    sot_token = tokenizer.encoder[SOT_TEXT]
    eot_token = tokenizer.encoder[EOT_TEXT]
    all_tokens = [[sot_token] +
                  tokenizer.encode(text) + [eot_token] for text in texts]
    result = np.zeros((len(all_tokens), CONTEXT_LEN)).astype(np.int64)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > CONTEXT_LEN:
            tokens = tokens[:CONTEXT_LEN - 1] + [eot_token]

        result[i, : len(tokens)] = tokens

    return result


def preprocess(batch_size, prompt, tokenizer):
    unconditional_condition_token = tokenize(batch_size * [""], tokenizer)
    condition_token = tokenize(batch_size * [prompt], tokenizer)
    print(unconditional_condition_token[0])
    print(condition_token[0])
    return (unconditional_condition_token, condition_token)


def image_adjust(image_sr, output_mode):
    print(f"{output_mode}: input shape {image_sr.shape}")
    if output_mode == 0:
        image_res = cv2.resize(image_sr, (1280, 768))
        image_save = image_res[24:744,:]
    elif output_mode == 1:
        image_res = cv2.resize(image_sr, (768, 1280))
        image_save = image_res[:, 24:744]
    elif output_mode == 2:
        image_res = cv2.resize(image_sr, (1280, 1024))
        image_save = image_res[62:962, 40:1240]
    elif output_mode == 3:
        image_res = cv2.resize(image_sr, (1024, 1280))
        image_save = image_res[40:1240, 62:962]
    elif output_mode == 4:
        image_save = cv2.resize(image_sr, (1000, 1000))
    print(f"{output_mode}: output_shape {image_save.shape}")
    return image_save


def postprocess(images, save_path, output_mode):
    # images = images.asnumpy()
    output = list()
    if output_mode == 0:
        save_path_file  = os.path.join(save_path, "1280_720.png")
        save_path_file_sr = os.path.join(save_path, "1280_720_sr.png")
    elif output_mode == 1:
        save_path_file  = os.path.join(save_path, "720_1280.png")
        save_path_file_sr = os.path.join(save_path, "720_1280_sr.png")
    elif output_mode == 2:
        save_path_file  = os.path.join(save_path, "1200_900.png")
        save_path_file_sr = os.path.join(save_path, "1200_900_sr.png")
    elif output_mode == 3:
        save_path_file  = os.path.join(save_path, "900_1200.png")
        save_path_file_sr = os.path.join(save_path, "900_1200_sr.png")
    elif output_mode == 4:
        save_path_file  = os.path.join(save_path, "1000_1000.png")
        save_path_file_sr = os.path.join(save_path, "1000_1000_sr.png")
    print("save_path: =================================================")
    print(save_path_file)
    print(save_path_file_sr)
    for image in images:
        image = 255. * image.transpose(1, 2, 0)
        output.append(image)
        if save_path:
            img = Image.fromarray(image.astype(np.uint8))
            base_count = len(os.listdir(save_path))
            img.save(save_path_file)
        print("=========================================")
        # exit()
        print(image.shape)
        # exit()
        image_sr = test_rrdb_om_srx4(
            image,
            "./models/rrdb_srx4_fp32_new.om"
            )
        image_save = image_adjust(image_sr, output_mode)
        # print(f"{output_mode}: output_shape {image_save.shape}")
        image_save = cv2.cvtColor(image_save, cv2.COLOR_BGR2RGB)
        ret = cv2.imwrite(save_path_file_sr, image_save)
        assert ret
    return output

def load_model(mindir_path, context, output_mode):
    model_path_0 = os.path.join(mindir_path, "wukong_youhua_384_640_out_graph.mindir")
    model_path_1 = os.path.join(mindir_path, "wukong_youhua_640_384_out_graph.mindir")
    model_path_2 = os.path.join(mindir_path, "wukong_youhua_512_640_out_graph.mindir")
    model_path_3 = os.path.join(mindir_path, "wukong_youhua_640_512_out_graph.mindir")
    model_path_4 = os.path.join(mindir_path, "wukong_youhua_512_512_out_graph.mindir")

    model0 = mslite.Model()
    model1 = mslite.Model()
    model2 = mslite.Model()
    model3 = mslite.Model()
    model4 = mslite.Model()
    model0.build_from_file(model_path_0, mslite.ModelType.MINDIR, context)
    model1.build_from_file(model_path_1, mslite.ModelType.MINDIR, context)
    model2.build_from_file(model_path_2, mslite.ModelType.MINDIR, context)
    model3.build_from_file(model_path_3, mslite.ModelType.MINDIR, context)
    model4.build_from_file(model_path_4, mslite.ModelType.MINDIR, context)
    model_list = []
    model_list.append(model0)
    model_list.append(model1)
    model_list.append(model2)
    model_list.append(model3)
    model_list.append(model4)
    return model_list   # model0

def main(args):
    work_dir = os.path.dirname(os.path.abspath(__file__))
    device_id = int(os.getenv("DEVICE_ID", 0))
    batch_size = 1

    if os.path.isabs(args.mindir_path):
        mindir_path = args.mindir_path
    else:
        mindir_path = os.path.join(work_dir, args.mindir_path)

    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = 0
    context.ascend.precision_mode = "preferred_fp32"
    print(mindir_path)
    # exit()
    model_list = load_model(mindir_path, context, args.output_mode)
    model = model_list[args.output_mode]
    # model = mslite.Model()
    # model.build_from_file(mindir_path, mslite.ModelType.MINDIR, context)

    tokenizer = WordpieceTokenizer(args.vocab_path)
    os.makedirs(args.save_path, exist_ok=True)
    input = preprocess(batch_size, args.prompt, tokenizer)

    inputs = model.get_inputs()
    for i in range(len(inputs)):
        inputs[i].set_data_from_numpy(np.asarray(input[i], dtype=np.int32))

    # outputs = model.get_outputs()
    print("start predicting...")
    # model.predict(inputs, outputs)
    # model = model_list[args.output_mode]
    outputs = model.predict(inputs)
    print("start generating image...")
    images = postprocess(outputs[0].get_data_to_numpy(), args.save_path, args.output_mode)

    return images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', required=True,
                        default="", type=str, help='')
    parser.add_argument(
        '--mindir_path', default="./models/", type=str, help='')
    parser.add_argument(
        '--vocab_path', default="./config/vocab_zh.txt", type=str, help='')
    parser.add_argument(
        '--save_path', default="./output", type=str, help='')
    parser.add_argument(
        '--output_mode', default=1, type=int, help='0: 1280*720;'
                                                     '1: 720*1280;'
                                                     '2: 1200*900;'
                                                     '3: 900*1200;'
                                                     '4: 1000*1000')
    args = parser.parse_args()
    time1 = time.time()
    print("==================================")
    print(args.output_mode)
    # exit()
    main(args)
    time2 = time.time()
    print("============================================")
    print(f"time:{time2 - time1}")
    

