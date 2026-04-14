# HappyHorse 1.0 API: Python Wrapper for Alibaba's #1 AI Video Generator

[![PyPI version](https://img.shields.io/pypi/v/happyhorse-1-api.svg)](https://pypi.org/project/happyhorse-1-api/)
[![GitHub stars](https://img.shields.io/github/stars/Anil-matcha/HappyHorse-1.0-API.svg)](https://github.com/Anil-matcha/HappyHorse-1.0-API/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

The most comprehensive Python wrapper for the **HappyHorse 1.0 API** (developed by Alibaba's Taotian Group), delivered via [muapi.ai](https://muapi.ai). Generate cinematic, native 1080p AI videos with integrated audio from text prompts and static images — currently **ranked #1 on the Artificial Analysis Video Arena**.

Join subreddit [HappyHorseAI](https://www.reddit.com/r/HappyHorseAI_) for discussion

> 🌊 **Also explore these top AI video models:**
> - 🎬 [Seedance 2.0 API](https://github.com/Anil-matcha/Seedance-2.0-API) — ByteDance's cinematic 2K model with character sheets, omni-reference & video edit
> - 🎥 [Veo 4 API](https://github.com/Anil-matcha/Veo-4-API) — Google DeepMind's native 4K model with audio, character consistency & camera controls

## 🚀 Why Use HappyHorse 1.0 API?

HappyHorse 1.0 is Alibaba's state-of-the-art AI video generation model, built by the Future Life Lab team at Taotian Group. It debuted anonymously on April 7, 2026, instantly claiming the top spot in both Text-to-Video and Image-to-Video categories on the Artificial Analysis leaderboard.

- **#1 Ranked**: 1333 Elo in T2V, 1392 Elo in I2V — surpassing every competitor on the Artificial Analysis leaderboard.
- **Native 1080p HD**: Full HD output without upscaling, powered by a 15B-parameter 40-layer Transformer architecture.
- **Integrated Audio-Video**: Jointly generates video and audio in a single forward pass — no separate audio pipeline needed.
- **Blazing Fast**: ~10 seconds average generation time, one of the fastest available models.
- **Developer-First**: Simple Python SDK via MuAPI infrastructure.

## 🌟 Key Features of HappyHorse 1.0 API

- ✅ **HappyHorse 1.0 Text-to-Video (T2V)**: Transform descriptive prompts into stunning native 1080p HD video clips.
- ✅ **HappyHorse 1.0 Image-to-Video (I2V)**: Animate static images with precise motion control using `images_list`.
- ✅ **Integrated Audio-Video Generation**: Generate synchronized audio and video jointly in one Transformer pass — include sound cues like "rain pattering" or "crowd cheering" directly in your prompt.
- ✅ **Video Extension**: Seamlessly extend existing clips while maintaining consistent style and motion.
- ✅ **Video Edit**: Edit existing videos using natural language prompts and reference images.
- ✅ **File Upload**: Upload local images and videos directly via the `upload_file` method for use in generation tasks.
- ✅ **Flexible Aspect Ratios**: Optimized for `16:9`, `9:16` (TikTok/Reels), and `1:1`.
- ✅ **Quality Tiers**: `1080p` (default) and `4k` output support.

---

## 🛠 Installation

### Via Pip (Recommended)
```bash
pip install happyhorse-1-api
```

### From Source
```bash
# Clone the HappyHorse 1.0 API repository
git clone https://github.com/Anil-matcha/HappyHorse-1.0-API.git
cd HappyHorse-1.0-API

# Install required dependencies
pip install -r requirements.txt
```

### Configuration
Create a `.env` file in the root directory and add your [MuAPI](https://muapi.ai) API key:
```env
MUAPI_API_KEY=your_muapi_api_key_here
```

---

## 🤖 HappyHorse 1.0 MCP Server

Use HappyHorse 1.0 as an **MCP (Model Context Protocol)** server, allowing AI assistants like Claude Desktop or Cursor to directly invoke HappyHorse generation tools.

### Running the MCP Server
1. Ensure `MUAPI_API_KEY` is set in your environment.
2. Run the server:
   ```bash
   python3 mcp_server.py
   ```
3. To test with the MCP Inspector:
   ```bash
   npx -y @modelcontextprotocol/inspector python3 mcp_server.py
   ```

---

## 💻 Quick Start with HappyHorse 1.0 API (Python)

```python
from happyhorse_api import HappyHorseAPI

# Initialize the HappyHorse 1.0 client
api = HappyHorseAPI()

# 1. Generate Video from Text (T2V)
print("Generating AI Video using HappyHorse 1.0...")
submission = api.text_to_video(
    prompt="A cinematic aerial shot of a coastal city at golden hour, waves crashing against cliffs, birds flying, 1080p",
    aspect_ratio="16:9",
    duration=10,
    quality="1080p"
)

# 2. Wait for completion
result = api.wait_for_completion(submission['request_id'])
print(f"Success! View your HappyHorse 1.0 video here: {result['outputs'][0]}")
```

---

## 🎵 Audio-Video Generation

HappyHorse 1.0's standout feature: jointly generating video and audio in a single Transformer forward pass.

```python
from happyhorse_api import HappyHorseAPI

api = HappyHorseAPI()

# Generate video + audio together from text
submission = api.text_to_video_with_audio(
    prompt="A thunderstorm rolling over a mountain range, lightning flashing, thunder rumbling, rain pattering on leaves",
    aspect_ratio="16:9",
    duration=10,
    quality="1080p"
)

result = api.wait_for_completion(submission['request_id'])
print(f"Video with audio: {result['outputs'][0]}")

# Or use image-to-video with audio
submission = api.image_to_video_with_audio(
    prompt="@image1 comes alive — waves crashing, seagulls calling, ocean breeze",
    images_list=["https://example.com/beach.jpg"],
    aspect_ratio="16:9",
    duration=10,
)
result = api.wait_for_completion(submission['request_id'])
print(f"Animated image with audio: {result['outputs'][0]}")
```

> **Tip**: Include explicit sound cues in your prompt (e.g. "crowd cheering", "piano melody", "engine roaring") for richer, more accurate audio generation.

---

## 📡 API Endpoints & Reference

### 1. HappyHorse 1.0 Text-to-Video (T2V)
**Endpoint**: `POST https://api.muapi.ai/api/v1/happyhorse-1.0-t2v`

```bash
curl --location --request POST "https://api.muapi.ai/api/v1/happyhorse-1.0-t2v" \
  --header "Content-Type: application/json" \
  --header "x-api-key: YOUR_API_KEY" \
  --data-raw '{
      "prompt": "A majestic eagle soaring over snow-capped mountains at sunrise",
      "aspect_ratio": "16:9",
      "duration": 10,
      "quality": "1080p"
  }'
```

### 2. HappyHorse 1.0 Image-to-Video (I2V)
**Endpoint**: `POST https://api.muapi.ai/api/v1/happyhorse-1.0-i2v`

Reference images with `@image1`, `@image2`, etc. in the prompt.

```bash
curl --location --request POST "https://api.muapi.ai/api/v1/happyhorse-1.0-i2v" \
  --header "Content-Type: application/json" \
  --header "x-api-key: YOUR_API_KEY" \
  --data-raw '{
      "prompt": "@image1 — the clouds drift slowly, light shifts from golden to dusk",
      "images_list": ["https://example.com/landscape.jpg"],
      "aspect_ratio": "16:9",
      "duration": 10,
      "quality": "1080p"
  }'
```

### 3. HappyHorse 1.0 T2V with Audio
**Endpoint**: `POST https://api.muapi.ai/api/v1/happyhorse-1.0-t2v-audio`

Jointly generate video and synchronized audio in one pass.

```bash
curl --location --request POST "https://api.muapi.ai/api/v1/happyhorse-1.0-t2v-audio" \
  --header "Content-Type: application/json" \
  --header "x-api-key: YOUR_API_KEY" \
  --data-raw '{
      "prompt": "A busy Tokyo street at night, neon signs, rain, traffic noise, jazz music drifting from a bar",
      "aspect_ratio": "16:9",
      "duration": 10,
      "quality": "1080p"
  }'
```

### 4. HappyHorse 1.0 I2V with Audio
**Endpoint**: `POST https://api.muapi.ai/api/v1/happyhorse-1.0-i2v-audio`

Animate images with jointly generated audio.

```bash
curl --location --request POST "https://api.muapi.ai/api/v1/happyhorse-1.0-i2v-audio" \
  --header "Content-Type: application/json" \
  --header "x-api-key: YOUR_API_KEY" \
  --data-raw '{
      "prompt": "@image1 — waves begin to crash, seagulls cry in the distance, wind howling",
      "images_list": ["https://example.com/ocean.jpg"],
      "aspect_ratio": "16:9",
      "duration": 10,
      "quality": "1080p"
  }'
```

### 5. Video Extension
**Endpoint**: `POST https://api.muapi.ai/api/v1/happyhorse-1.0-extend`

```bash
curl --location --request POST "https://api.muapi.ai/api/v1/happyhorse-1.0-extend" \
  --header "Content-Type: application/json" \
  --header "x-api-key: YOUR_API_KEY" \
  --data-raw '{
      "request_id": "your-completed-request-id",
      "prompt": "The eagle lands on a mountain peak, surveying the valley",
      "duration": 10,
      "quality": "1080p"
  }'
```

### 6. Video Edit
**Endpoint**: `POST https://api.muapi.ai/api/v1/happyhorse-1.0-video-edit`

```bash
curl --location --request POST "https://api.muapi.ai/api/v1/happyhorse-1.0-video-edit" \
  --header "Content-Type: application/json" \
  --header "x-api-key: YOUR_API_KEY" \
  --data-raw '{
      "prompt": "Change the weather to a dramatic thunderstorm",
      "video_urls": ["https://example.com/video.mp4"],
      "aspect_ratio": "16:9",
      "quality": "1080p"
  }'
```

---

## 📖 API Method Reference

| Method | Parameters | Description |
| :--- | :--- | :--- |
| `text_to_video` | `prompt`, `aspect_ratio`, `duration`, `quality`, `with_audio` | Generate native 1080p video from text. |
| `image_to_video` | `prompt`, `images_list`, `aspect_ratio`, `duration`, `quality`, `with_audio` | Animate images. Reference with `@image1`, `@image2`, etc. |
| `text_to_video_with_audio` | `prompt`, `aspect_ratio`, `duration`, `quality` | T2V with jointly generated audio in one pass. |
| `image_to_video_with_audio` | `prompt`, `images_list`, `aspect_ratio`, `duration`, `quality` | I2V with jointly generated audio in one pass. |
| `extend_video` | `request_id`, `prompt`, `duration`, `quality` | Extend an existing HappyHorse video segment. |
| `video_edit` | `prompt`, `video_urls`, `images_list`, `aspect_ratio`, `quality` | Edit existing videos with natural language. |
| `upload_file` | `file_path` | Upload a local file (image or video) to MuAPI. |
| `get_result` | `request_id` | Check task status and retrieve outputs. |
| `wait_for_completion` | `request_id`, `poll_interval`, `timeout` | Blocking helper — polls until generation completes. |

---

## 🔗 Official Resources
- **API Provider**: [MuAPI.ai](https://muapi.ai)
- **Leaderboard**: [Artificial Analysis Video Arena](https://artificialanalysis.ai/video)

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Keywords**: HappyHorse 1.0 API, Alibaba HappyHorse, AI Video Generator, Text-to-Video AI, Image-to-Video API, HappyHorse Python SDK, Alibaba Video AI, Audio Video Generation, Integrated Audio Video, MuAPI, Video Generation API, Native 1080p AI Video, AI Video Creation, HappyHorse API Documentation, HappyHorse I2V, HappyHorse T2V, AI Movie Generator, Python Video API, HappyHorse Tutorial, #1 AI Video Model.
