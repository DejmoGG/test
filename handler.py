import os, json, subprocess, uuid, shutil, tempfile, pathlib
from typing import Dict, Any
import runpod
import requests

WEIGHTS = os.environ.get("WEIGHTS_DIR", "/workspace/weights")
GEN_SCRIPT = "/workspace/generate_infinitetalk.py"

# --- utilities ---------------------------------------------------------------

def _fetch(url_or_path: str, dst_dir: str) -> str:
    """Download a URL or copy a local path into dst_dir, return local path."""
    dst_dir = pathlib.Path(dst_dir); dst_dir.mkdir(parents=True, exist_ok=True)
    if url_or_path.startswith(("http://", "https://")):
        fn = dst_dir / (uuid.uuid4().hex + "-" + url_or_path.split("/")[-1].split("?")[0])
        with requests.get(url_or_path, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(fn, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk: f.write(chunk)
        return str(fn)
    else:
        p = pathlib.Path(url_or_path)
        if not p.exists(): raise FileNotFoundError(f"{url_or_path} not found")
        dst = dst_dir / p.name
        if p.is_dir():
            if dst.exists(): shutil.rmtree(dst)
            shutil.copytree(p, dst)
        else:
            shutil.copy2(p, dst)
        return str(dst)

def _run_cmd(cmd: list[str], cwd: str | None = None):
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    logs = []
    for line in p.stdout:
        logs.append(line.rstrip())
    rc = p.wait()
    return rc, "\n".join(logs)

# --- RunPod handler ----------------------------------------------------------

def handler(event: Dict[str, Any]):
    """
    Inputs:
      {
        "mode": "streaming" | "clip",
        "size": "infinitetalk-480" | "infinitetalk-720",
        "sample_steps": 40,
        "motion_frame": 9,
        "audio_cfg": 4,
        "text_cfg": 5,
        "quant": null | "fp8",
        "lora": false,
        "input": {
          "type": "i2v" | "v2v",
          "image_url": "...",    # for i2v
          "video_url": "...",    # for v2v
          "audio_url": "..."
        }
      }
    Returns:
      { "output_video": "file:///out/<file>.mp4", "logs": "...tail..." }
    """
    params = event.get("input", {})
    mode = event.get("mode", "streaming")
    size = event.get("size", "infinitetalk-480")
    steps = int(event.get("sample_steps", 40))
    motion = int(event.get("motion_frame", 9))
    audio_cfg = event.get("audio_cfg", 4)
    text_cfg = event.get("text_cfg", 5)
    quant = event.get("quant")  # None or "fp8"
    use_lora = bool(event.get("lora", False))

    tmp = tempfile.mkdtemp(prefix="italk_")
    assets = {"image": None, "video": None, "audio": None}

    try:
        inp = params.get("type", "i2v")
        if inp == "i2v":
            assets["image"] = _fetch(params["image_url"], tmp)
        elif inp == "v2v":
            assets["video"] = _fetch(params["video_url"], tmp)
        else:
            raise ValueError("input.type must be 'i2v' or 'v2v'")
        assets["audio"] = _fetch(params["audio_url"], tmp)

        # Build input JSON for the upstream script
        example_json = {
            "input": [{
                "image_path": assets["image"],
                "video_path": assets["video"],
                "audio_path": assets["audio"],
            }]
        }
        jpath = os.path.join(tmp, "input.json")
        with open(jpath, "w") as f: json.dump(example_json, f)

        out_name = f"infinitetalk_{uuid.uuid4().hex}.mp4"
        out_path = f"/out/{out_name}"

        cmd = [
            "python", GEN_SCRIPT,
            "--ckpt_dir", f"{WEIGHTS}/Wan2.1-I2V-14B-480P",
            "--wav2vec_dir", f"{WEIGHTS}/chinese-wav2vec2-base",
            "--infinitetalk_dir", f"{WEIGHTS}/InfiniteTalk/single/infinitetalk.safetensors",
            "--input_json", jpath,
            "--size", size,
            "--sample_steps", str(steps),
            "--mode", mode,
            "--motion_frame", str(motion),
            "--save_file", out_path.replace(".mp4", ""),
            "--sample_text_guide_scale", str(text_cfg),
            "--sample_audio_guide_scale", str(audio_cfg),
            "--num_persistent_param_in_dit", "0"  # safer for 24GB
        ]

        if quant:
            cmd += ["--quant", quant,
                    "--quant_dir", f"{WEIGHTS}/InfiniteTalk/quant_models/infinitetalk_single_fp8.safetensors"]

        if use_lora:
            cmd += [
              "--lora_dir", f"{WEIGHTS}/Wan2.1_I2V_14B_FusionX_LoRA.safetensors",
              "--lora_scale", "1.0",
              "--sample_text_guide_scale", "1.0",
              "--sample_audio_guide_scale", "2.0",
              "--sample_steps", "8"
            ]

        rc, logs = _run_cmd(cmd, cwd="/workspace")
        if rc != 0:
            raise RuntimeError(f"Inference failed (exit {rc}). See logs.\n{logs}")

        # Upstream script writes <save_file>.mp4
        if not os.path.exists(out_path):
            # try  _res.mp4 naming
            candidates = list(pathlib.Path("/out").glob("infinitetalk_*.mp4"))
            if candidates:
                out_path = str(candidates[-1])
            else:
                raise FileNotFoundError("No output video was produced.")

        return { "output_video": f"file://{out_path}", "logs": logs[-2000:] }

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

runpod.serverless.start({"handler": handler})
