import dataclasses
from enum import auto, Enum
from typing import List, Any, Dict, Union, Tuple
import re
import base64
from io import BytesIO
from transformers import AutoTokenizer


class SeparatorStyle(Enum):
    """Different separator style."""

    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    CHATML = auto()
    LLAMA_2 = auto()
    LLAMA_3 = auto()
    QWEN = auto()
    GEMMA = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    tokenizer_id: str = ""
    tokenizer: Any = None
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0]
            # NOTE we allow the dialogue is not started with <image>
            # elif not init_msg.startswith("<image>"):
            # # TODO maybe this should be changed for interleaved data?
            # #     init_msg = init_msg.replace("<image>", "").strip()
            # #     messages[0] = (init_role, "<image>\n" + init_msg)
            # else:
            messages[0] = (init_role, init_msg)

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    # FIXED: Handle speech-only message format properly
                    if type(message) is tuple:
                        # Speech-only format: (text, audio) - extract text only
                        if len(message) >= 2:
                            message = message[0]  # Get text part
                        else:
                            message = str(message[0]) if message else ""
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"

        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    # FIXED: Handle speech-only message format properly
                    if type(message) is tuple:
                        # Speech-only format: (text, audio) - extract text only
                        if len(message) >= 2:
                            message = message[0]  # Get text part
                        else:
                            message = str(message[0]) if message else ""
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"

        elif self.sep_style == SeparatorStyle.CHATML:
            ret = "" if self.system == "" else self.system + self.sep + "\n"
            for role, message in messages:
                if message:
                    # FIXED: Handle speech-only message format properly
                    if type(message) is tuple:
                        # Speech-only format: (text, audio) - extract text only
                        if len(message) >= 2:
                            message = message[0]  # Get text part
                        else:
                            message = str(message[0]) if message else ""
                        # if "<image>" not in message:
                        #     message = "<image>" + len(images) + message
                    ret += role + "\n" + message + self.sep + "\n"
                else:
                    ret += role + "\n"

        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    # FIXED: Handle speech-only message format properly
                    if type(message) is tuple:
                        # Speech-only format: (text, audio) - extract text only
                        if len(message) >= 2:
                            message = message[0]  # Get text part
                        else:
                            message = str(message[0]) if message else ""
                    ret += role + message + self.sep
                else:
                    ret += role

        elif self.sep_style == SeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n"
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    # FIXED: Handle speech-only message format properly
                    if type(message) is tuple:
                        # Speech-only format: (text, audio) - extract text only
                        if len(message) >= 2:
                            message = message[0]  # Get text part
                        else:
                            message = str(message[0]) if message else ""
                    assert role == self.roles[0], "first message should come from user"
                    if self.system:
                        message = wrap_sys(self.system) + message
                    ret += wrap_inst(message)
                else:
                    # FIXED: Handle speech-only message format properly
                    if type(message) is tuple:
                        # Speech-only format: (text, audio) - extract text only
                        if len(message) >= 2:
                            message = message[0]  # Get text part
                        else:
                            message = str(message[0]) if message else ""
                    if message:
                        ret += " " + message + " </s><s>"
                    else:
                        ret += ""

        elif self.sep_style == SeparatorStyle.LLAMA_3:
            chat_template = self.tokenizer.chat_template
            if chat_template is not None:
                encodeds = []
                for i, (role, message) in enumerate(messages):
                    if message:
                        # FIXED: Handle speech-only message format properly
                        if type(message) is tuple:
                            # Speech-only format: (text, audio) - extract text only
                            if len(message) >= 2:
                                message = message[0]  # Get text part
                            else:
                                message = str(message[0]) if message else ""
                        encodeds.append({"role": role, "content": message})
                ret = self.tokenizer.apply_chat_template(encodeds, tokenize=False)
            else:
                ret = f"{self.system}\n\n"
                for i, (role, message) in enumerate(messages):
                    if message:
                        # FIXED: Handle speech-only message format properly
                        if type(message) is tuple:
                            # Speech-only format: (text, audio) - extract text only
                            if len(message) >= 2:
                                message = message[0]  # Get text part
                            else:
                                message = str(message[0]) if message else ""
                        ret += f"<|start_header_id|>{role}<|end_header_id|>\n\n{message}<|eot_id|>"
                    else:
                        ret += f"<|start_header_id|>{role}<|end_header_id|>\n\n"

        elif self.sep_style == SeparatorStyle.QWEN:
            ret = "" if self.system == "" else self.system + self.tokenizer.im_end + "\n"
            for i, (role, message) in enumerate(messages):
                if message:
                    # FIXED: Handle speech-only message format properly
                    if type(message) is tuple:
                        # Speech-only format: (text, audio) - extract text only
                        if len(message) >= 2:
                            message = message[0]  # Get text part
                        else:
                            message = str(message[0]) if message else ""
                    ret += self.tokenizer.im_start + role + "\n" + message + self.tokenizer.im_end + "\n"
                else:
                    ret += self.tokenizer.im_start + role + "\n"

        elif self.sep_style == SeparatorStyle.GEMMA:
            ret = ""
            for i, (role, message) in enumerate(messages):
                assert role in ["user", "model"], f"Unexpected role: {role}"
                if message:
                    # FIXED: Handle speech-only message format properly
                    if type(message) is tuple:
                        # Speech-only format: (text, audio) - extract text only
                        if len(message) >= 2:
                            message = message[0]  # Get text part
                        else:
                            message = str(message[0]) if message else ""
                    ret += f"<start_of_turn>{role}\n{message}<end_of_turn>\n"
                else:
                    ret += f"<start_of_turn>{role}\n"

        elif self.sep_style == SeparatorStyle.PLAIN:
            ret = ""
            for role, message in messages:
                if message:
                    # FIXED: Handle speech-only message format properly
                    if type(message) is tuple:
                        # Speech-only format: (text, audio) - extract text only
                        if len(message) >= 2:
                            message = message[0]  # Get text part
                        else:
                            message = str(message[0]) if message else ""
                    ret += message + self.sep
                else:
                    ret += ""
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def process_image(self, image, image_process_mode, return_pil=False, image_format='PNG', max_len=1344, min_len=672):
        # SPEECH-ONLY: Image processing removed
        # This method is kept for compatibility but does nothing
        return None

    def get_images(self, return_pil=False):
        # SPEECH-ONLY: No images in speech-only mode
        return []

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                # FIXED: Handle speech-only message format properly
                if type(msg) is tuple:
                    # Speech-only format: (text, audio) - extract text only
                    if len(msg) >= 2:
                        msg = msg[0]  # Get text part
                    else:
                        msg = str(msg[0]) if msg else ""
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version,
            tokenizer_id=self.tokenizer_id,
            tokenizer=self.tokenizer,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
            skip_next=self.skip_next,
        )

    def dict(self):
        # FIXED: Handle speech-only message format in serialization
        messages = []
        for role, msg in self.messages:
            if type(msg) is tuple:
                # Speech-only format: (text, audio) - serialize text only
                if len(msg) >= 2:
                    msg = msg[0]  # Get text part
                else:
                    msg = str(msg[0]) if msg else ""
            messages.append([role, msg])
            
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "version": self.version,
            "tokenizer_id": self.tokenizer_id,
            "stop_str": self.stop_str,
            "stop_token_ids": self.stop_token_ids,
            "skip_next": self.skip_next,
        }


# Speech-only conversation templates
conv_vicuna_v0 = Conversation(
    system="A chat between a curious human and an artificial assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_vicuna_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llama_2 = Conversation(
    system="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_llama_3 = Conversation(
    system="""You are a helpful assistant.""",
    roles=("user", "assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_3,
    sep="<s>",
    sep2="</s>",
)

conv_qwen = Conversation(
    system="You are a helpful assistant.",
    roles=("user", "assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.QWEN,
    sep="<s>",
    sep2="</s>",
)

conv_gemma_instruct = Conversation(
    system="",
    roles=("user", "model"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.GEMMA,
    sep="",
    sep2="",
)

conv_chatml_direct = Conversation(
    system="""<|im_start|>system
You are a helpful assistant.<|im_end|>""",
    roles=("<|im_start|>user", "<|im_start|>assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.CHATML,
    sep="<|im_end|>",
)

conv_mpt = Conversation(
    system="""<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_llava_plain = Conversation(
    system="",
    roles=("", ""),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
)

conv_llava_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_llava_v0_mmtag = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_llava_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llava_v1_mmtag = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

# Speech-only conversation templates
conv_templates = {
    "default": conv_vicuna_v0,
    "v0": conv_vicuna_v0,
    "v1": conv_vicuna_v1,
    "vicuna_v1": conv_vicuna_v1,
    "llama_2": conv_llama_2,
    "llama_3": conv_llama_3,
    "qwen": conv_qwen,
    "gemma_instruct": conv_gemma_instruct,
    "chatml_direct": conv_chatml_direct,
    "mistral_direct": conv_chatml_direct,
    "plain": conv_llava_plain,
    "llava_plain": conv_llava_plain,
    "llava_v0": conv_llava_v0,
    "llava_v0_mmtag": conv_llava_v0_mmtag,
    "llava_v1": conv_llava_v1,
    "llava_v1_mmtag": conv_llava_v1_mmtag,
    "mpt": conv_mpt,
}

# Default conversation for speech-only mode
default_conversation = conv_vicuna_v1


if __name__ == "__main__":
    print(default_conversation.get_prompt())

