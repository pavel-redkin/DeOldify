import io
import logging
import os
import tempfile
from pathlib import Path

import torch

_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

from flask import Flask, jsonify, request, send_file

from deoldify import device as device_settings
from deoldify.device_id import DeviceId
from deoldify.visualize import get_image_colorizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Model initialisation (runs once at startup)
# ---------------------------------------------------------------------------
gpu_enabled = os.environ.get("DEOLDIFY_GPU", "0") == "1"
if gpu_enabled:
    device_settings.set(device=DeviceId.GPU0)
else:
    device_settings.set(device=DeviceId.CPU)

artistic = os.environ.get("DEOLDIFY_ARTISTIC", "0") == "1"
render_factor_default = int(os.environ.get("DEOLDIFY_RENDER_FACTOR", "35"))

logger.info(
    "Loading model  artistic=%s  gpu=%s  default_render_factor=%s",
    artistic, gpu_enabled, render_factor_default,
)
colorizer = get_image_colorizer(artistic=artistic)
logger.info("Model loaded and ready")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/process", methods=["POST"])
def process():
    """Colorize an image supplied via a URL.

    JSON body:
        url           – direct link to a jpg/png image  (required)
        render_factor – integer 7‑45, default from env  (optional)
    """
    data = request.get_json(silent=True)
    if not data or "url" not in data:
        return jsonify({"error": "JSON body with 'url' field is required"}), 400

    url = data["url"]
    render_factor = data.get("render_factor", render_factor_default)

    try:
        render_factor = int(render_factor)
    except (TypeError, ValueError):
        return jsonify({"error": "render_factor must be an integer"}), 400

    if not (7 <= render_factor <= 45):
        return jsonify({"error": "render_factor must be between 7 and 45"}), 400

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "input.png"
            result_path = colorizer.plot_transformed_image_from_url(
                url=url,
                path=str(source_path),
                render_factor=render_factor,
                compare=False,
                watermarked=False,
            )
            return send_file(
                str(result_path),
                mimetype="image/png",
                download_name="colorized.png",
            )
    except Exception as exc:
        logger.exception("Colorization failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8181"))
    app.run(host="0.0.0.0", port=port)
