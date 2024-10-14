import base64
import glob
import json
import os
import shutil
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from detection_sever import Yolov8DetectionModel, init_yolov8_model
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.messages import HumanMessage


def is_valid_url(server_url):
    try:
        result = urlparse(server_url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


class YoloV8LLM(LLM):
    """A custom chat model that echoes the first `n` characters of the input.

    When contributing an implementation to LangChain, carefully document
    the model including the initialization parameters, include
    an example of how to initialize the model and include any relevant
    links to the underlying models documentation or API.

    Example:

        .. code-block:: python

            model = CustomChatModel(n=2)
            result = model.invoke([HumanMessage(content="hello")])
            result = model.batch([[HumanMessage(content="hello")],
                                [HumanMessage(content="world")]])
    """

    def parse_prompt(self, prompt):
        txt_prompt, image_path = prompt.split("image_url=")
        return txt_prompt, image_path

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input.

        Override this method to implement the LLM logic.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            The model output as a string. Actual completions SHOULD NOT include the prompt.
        """
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        server = kwargs["server"]
        txt_prompt, image_path = self.parse_prompt(prompt)

        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

        data = {"image_url": image_data, "prompt": txt_prompt}

        json_data = json.dumps(data)

        if is_valid_url(server):
            response = requests.post(
                server, data=json_data, headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200:
                result = f"Successfully sent {image_path}\n. Server response: {response.json()}."
            else:
                result = f"Failed to send {image_path}\n. Server response: {response.text}."
        elif callable(server):
            response = server(image_path, txt_prompt)["answer_yes"]
            result = f"{image_path},{response}"
        else:
            raise ValueError("Invalid server.")

        return result

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": "CustomChatModel",
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "custom"


def demo(server):
    img_folder_default = "/home/stardust/Downloads/trash"
    judge_prompt_default = "Is there any object in the image?"
    img_save_folder = "/home/stardust/Downloads/trash_test_save"

    img_folder = img_folder_default
    judge_prompt = judge_prompt_default

    if os.path.isdir(img_folder):
        file_patterns = ["*.jpg", "*.jpeg", "*.png"]
        image_batch = []
        for file_pattern in file_patterns:
            image_batch.extend(glob.glob(os.path.join(img_folder, file_pattern)))
        image_batch = sorted(image_batch)
    elif os.path.isfile(img_folder):
        image_batch = [img_folder]

    llm = YoloV8LLM()
    print(llm)

    batch_messages = [
        [HumanMessage(content=judge_prompt + "image_url=" + image_path)]
        for image_path in image_batch
    ]

    llm_response = llm.batch(batch_messages, server=server)

    if os.path.exists(img_save_folder):
        shutil.rmtree(img_save_folder)
    os.makedirs(img_save_folder)
    for response_str in llm_response:
        path, answer = response_str.split(",")
        if answer == "True":
            shutil.copy(path, os.path.join(img_save_folder, os.path.basename(path)))


def main():
    # use callable server
    """
    Main entry point for the script. Creates a callable server and calls
    demo with it.

    This will run the demo using the callable server.
    """

    model_args = init_yolov8_model()
    yolov8 = Yolov8DetectionModel(model_args)
    demo(yolov8)


if __name__ == "__main__":
    main()
