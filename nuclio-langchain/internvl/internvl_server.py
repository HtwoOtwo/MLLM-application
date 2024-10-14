import base64
import io
from io import BytesIO
from urllib.request import urlopen

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class InternVL:
    def __init__(self, args):
        self.args = args
        self.model_path = args.model_path
        self.device = args.device

        self.build_transform(input_size=448)
        if args.max_new_tokens is not None:
            max_new_tokens_ = args.max_new_tokens
        else:
            max_new_tokens_ = 2

        self.model = (
            AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                load_in_8bit=args.load_in_8bit,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                mirror="https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models",
            )
            .eval()
            .to(self.device)
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.generation_config = dict(
            num_beams=1,
            max_new_tokens=max_new_tokens_,
            do_sample=False,
        )

    def load_image(self, image_file, input_size=448, max_num=6):
        if image_file.startswith(("data:", "http://", "https://")):
            with urlopen(image_file) as response:
                image_data = response.read()
                image_file = BytesIO(image_data)
                image = Image.open(image_file).convert("RGB")
        else:
            image_bytes = base64.b64decode(image_file)
            image = Image.open(io.BytesIO(image_bytes))

        images = InternVL.dynamic_preprocess(
            image, image_size=448, use_thumbnail=True, max_num=max_num
        )
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def build_transform(self, input_size):
        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ]
        )
        self.transform = transform

    @staticmethod
    def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = InternVL.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    @staticmethod
    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def set_prompt(self, question):
        self.question = question

    def inference(self, image_url, prompt):
        image = self.load_image(image_url).to(torch.bfloat16).to(self.device)

        self.set_prompt(prompt)
        with torch.inference_mode():
            response = self.model.chat(self.tokenizer, image, self.question, self.generation_config)

        # for boolean response
        if response.startswith(("yes", "Yes")):
            judge_result = True
        else:
            judge_result = False

        output = {"answer_yes": judge_result, "output": response}

        return output

    def __call__(self, image_url, prompt):
        return self.inference(image_url, prompt)


def init_model():
    # Define default values
    model_path = "OpenGVLab/InternVL2-1B"
    device = "cuda:0"
    max_new_tokens = 1000

    # Create a simple class to hold the arguments
    class Args:
        def __init__(self, model_path, device, max_new_tokens=None, load_in_8bit=False):
            self.model_path = model_path
            self.device = device
            self.max_new_tokens = max_new_tokens
            self.load_in_8bit = load_in_8bit

    # Return an instance of Args with the default values
    return Args(model_path, device, max_new_tokens)
