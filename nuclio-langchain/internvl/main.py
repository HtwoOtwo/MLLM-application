import json

from internvl_server import InternVL, init_model


# Initialize your model
def init_context(context):
    context.logger.info("Init context...  0%")
    model_args = init_model()
    internvl = InternVL(model_args)
    context.user_data.model_handler = internvl
    context.logger.info("Init context...100%")


# Inference endpoint
def handler(context, event):
    try:
        if isinstance(event.body, dict):
            data = event.body
        else:
            data = json.loads(event.body)
    except json.JSONDecodeError:
        return context.Response(
            body=json.dumps({"error": "Invalid JSON format"}),
            headers={},
            content_type="application/json",
            status_code=400,
        )

    if "image_url" not in data:
        return context.Response(
            body=json.dumps({"error": "No image URL provided"}),
            headers={},
            content_type="application/json",
            status_code=400,
        )

    image_path = data["image_url"]

    prompt = data.get("prompt")

    results = context.user_data.model_handler.inference(image_path, prompt)

    return context.Response(
        body=json.dumps(results),
        headers={},
        content_type="application/json",
        status_code=200,
    )
