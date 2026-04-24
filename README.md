# Awesome HappyHorse 1.0 — API & Prompts

[![Awesome](https://awesome.re/badge.svg)](https://github.com/sindresorhus/awesome)
[![PyPI version](https://img.shields.io/pypi/v/happyhorse-1-api.svg)](https://pypi.org/project/happyhorse-1-api/)
[![GitHub stars](https://img.shields.io/github/stars/Anil-matcha/Awesome-HappyHorse-1.0-API-and-Prompt.svg)](https://github.com/Anil-matcha/Awesome-HappyHorse-1.0-API-and-Prompt/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

A curated Python wrapper **and prompt library** for the **HappyHorse 1.0 API** (developed by Alibaba's Taotian Group), delivered via [muapi.ai](https://muapi.ai). Generate cinematic native-1080p AI videos from text prompts and static images — currently the **#1 ranked AI video generation model** — and use the bundled prompt pack of high-performing community examples to get great output on the first try.

Join subreddit [HappyHorseAI](https://www.reddit.com/r/HappyHorseAI_) for discussion.

Try for free [ArenaAI](https://arena.ai/video).

> 🌊 **Also explore these top AI video models:**
> - 🎬 [Seedance 2.0 API](https://github.com/Anil-matcha/Seedance-2.0-API) — ByteDance's cinematic 2K model with character sheets, omni-reference & video edit
> - 🎥 [Veo 4 API](https://github.com/Anil-matcha/Veo-4-API) — Google DeepMind's native 4K model with audio, character consistency & camera controls

---

## ⚠️ Closed beta — access requirements

HappyHorse 1.0 on muapi is currently in a **closed playground beta**. Today:

- 🔒 **Not yet available via API keys.** Submitting to either endpoint with an `x-api-key` header returns `403 { "detail": "Happy Horse 1.0 is in closed beta and is not yet available via API keys. Use the muapi playground to try it." }`
- 💳 **Pro or Business plan required.** Free-plan users get `403 { "detail": "Happy Horse 1.0 is available only on the Pro and Business plans. Upgrade at https://muapi.ai/topup#plans." }`
- 🎨 Use Happy Horse today directly inside the [muapi playground](https://muapi.ai) on a Pro ($20/mo) or Business ($100/mo) plan. Early access to new models is a paid-plan perk.

The Python SDK and MCP server in this repo mirror the final public surface so you can wire them up now; they'll start succeeding the moment the API goes GA.

---

## 📚 Contents

- [Why HappyHorse 1.0](#-why-use-happyhorse-10-api)
- [Key Features](#-key-features-of-happyhorse-10-api)
- [Installation](#-installation)
- [MCP Server](#-happyhorse-10-mcp-server)
- [Quick Start (Python)](#-quick-start-with-happyhorse-10-api-python)
- [API Endpoints & Reference](#-api-endpoints--reference)
- [API Method Reference](#-api-method-reference)
- [Pricing](#-pricing)
- [Prompt Library](#-prompt-library) ← popular community prompts
- [Official Resources](#-official-resources)
- [License](#-license)

## 🚀 Why Use HappyHorse 1.0 API?

HappyHorse 1.0 is Alibaba's state-of-the-art AI video generation model, built by the Future Life Lab team at Taotian Group. It debuted anonymously on April 7, 2026, instantly claiming the top spot in both Text-to-Video and Image-to-Video public benchmarks.

- **#1 Ranked**: 1333 Elo in T2V, 1392 Elo in I2V — surpassing every competitor in public benchmarks.
- **Native 1080p HD**: Full HD output without upscaling, powered by a 15B-parameter 40-layer Transformer architecture.
- **Blazing Fast**: ~10 seconds average generation time, one of the fastest available models.
- **Developer-First**: Simple Python SDK via [MuAPI](https://muapi.ai) infrastructure.

## 🌟 Key Features of HappyHorse 1.0 API

- ✅ **Text-to-Video (T2V)** — `POST /api/v1/happy-horse-1-text-to-video-1080p`. Turn a prompt into a 1080p clip.
- ✅ **Image-to-Video (I2V)** — `POST /api/v1/happy-horse-1-image-to-video-1080p`. Animate a starting image into a 1080p clip.
- ✅ **Flexible Aspect Ratios**: `16:9`, `9:16` (TikTok/Reels), `1:1`, `4:3`, `3:4`.
- ✅ **Duration**: 4–15 seconds per clip.
- ✅ **Fixed 1080p output** with Full HD quality on every request.

---

## 🛠 Installation

### Via Pip (Recommended)
```bash
pip install happyhorse-1-api
```

### From Source
```bash
# Clone the HappyHorse 1.0 API repository
git clone https://github.com/Anil-matcha/Awesome-HappyHorse-1.0-API-and-Prompt.git
cd Awesome-HappyHorse-1.0-API-and-Prompt

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
    prompt="A cinematic aerial shot of a coastal city at golden hour, waves crashing against cliffs, birds flying",
    aspect_ratio="16:9",
    duration=10,
)

# 2. Wait for completion
result = api.wait_for_completion(submission['request_id'])
print(f"Success! View your HappyHorse 1.0 video here: {result['outputs'][0]}")
```

---

## 📡 API Endpoints & Reference

**Base URL**: `https://api.muapi.ai/api/v1`
**Auth**: `x-api-key: <YOUR_API_KEY>` (API-key access is currently in closed beta — see the note at the top of the README)

All responses return `{"request_id": "<id>", "status": "processing"}`; poll `GET /api/v1/predictions/{request_id}/result` until `status` is `completed` and read `outputs[0]` for the final muapi-hosted video URL.

### 1. HappyHorse 1.0 Text-to-Video (1080p)
**Endpoint**: `POST https://api.muapi.ai/api/v1/happy-horse-1-text-to-video-1080p`

| Field | Type | Required | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `prompt` | string | yes | — | Scene description. |
| `aspect_ratio` | enum | no | `"16:9"` | One of `16:9`, `9:16`, `1:1`, `4:3`, `3:4`. |
| `duration` | int | no | `5` | Integer seconds, `4 <= duration <= 15`. |
| `webhook_url` | URL | no | — | Optional — a `POST` is fired to this URL on completion. |

```bash
curl --location --request POST "https://api.muapi.ai/api/v1/happy-horse-1-text-to-video-1080p" \
  --header "Content-Type: application/json" \
  --header "x-api-key: YOUR_API_KEY" \
  --data-raw '{
      "prompt": "A majestic eagle soaring over snow-capped mountains at sunrise",
      "aspect_ratio": "16:9",
      "duration": 10
  }'
```

### 2. HappyHorse 1.0 Image-to-Video (1080p)
**Endpoint**: `POST https://api.muapi.ai/api/v1/happy-horse-1-image-to-video-1080p`

The first image in `images_list` is used as the start frame and the generated clip animates outward from it.

| Field | Type | Required | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `prompt` | string | no | `""` | Optional — guides motion direction/mood. |
| `images_list` | string[] | yes | — | Single-element list: the start-frame image URL. |
| `aspect_ratio` | enum | no | `"16:9"` | Same set as above. Output is cropped/padded if ratio differs from the input image. |
| `duration` | int | no | `5` | `4 <= duration <= 15`. |
| `webhook_url` | URL | no | — | Optional completion webhook. |

```bash
curl --location --request POST "https://api.muapi.ai/api/v1/happy-horse-1-image-to-video-1080p" \
  --header "Content-Type: application/json" \
  --header "x-api-key: YOUR_API_KEY" \
  --data-raw '{
      "prompt": "The clouds drift slowly, light shifts from golden to dusk",
      "images_list": ["https://example.com/landscape.jpg"],
      "aspect_ratio": "16:9",
      "duration": 10
  }'
```

### 3. Poll a prediction
**Endpoint**: `GET https://api.muapi.ai/api/v1/predictions/{request_id}/result`

Returns the muapi standard prediction envelope — `status` transitions through `queued` → `processing` → `completed` (or `failed`). On completion the `outputs` array contains muapi-hosted video URLs; the top-level `video` key mirrors `outputs[0]` for convenience.

### 4. Upload a local file
**Endpoint**: `POST https://api.muapi.ai/api/v1/upload_file`

Multipart-form helper to host a local image on muapi so it can be referenced in `images_list`.

---

## 📖 API Method Reference

| Method | Parameters | Description |
| :--- | :--- | :--- |
| `text_to_video` | `prompt`, `aspect_ratio`, `duration` | Generate a native 1080p clip from text. |
| `image_to_video` | `prompt`, `images_list`, `aspect_ratio`, `duration` | Animate a starting image into a 1080p clip. |
| `upload_file` | `file_path` | Upload a local file (image or video) to MuAPI and get back its hosted URL. |
| `get_result` | `request_id` | Check task status and retrieve outputs. |
| `wait_for_completion` | `request_id`, `poll_interval`, `timeout` | Blocking helper — polls until generation completes. |

---

## 💰 Pricing

HappyHorse 1.0 bills flat per-second at 1080p:

| Output | Rate | 5 s clip | 10 s clip | 15 s clip |
| :--- | :--- | :--- | :--- | :--- |
| 1080p | **$0.5625 / sec** | $2.81 | $5.63 | $8.44 |

Pricing shown here matches the live cost-strategy on muapi.

---

## 🎨 Prompt Library

A curated pack of high-performing community prompts for HappyHorse 1.0, organized by use case. Drop any of these directly into `text_to_video(...)` or the `/happy-horse-1-text-to-video-1080p` endpoint. Prompts are adapted from the [ZeroLu/awesome-happy-horse](https://github.com/ZeroLu/awesome-happy-horse) collection (CC BY 4.0).

### 🎬 Film & Cinematic Storytelling

**Cave Flashlight Cinematic**
```text
A flashlight beam exploring a cave system, illuminating wet limestone formations. The light catches crystalline calcite deposits that glitter and flash. Where the beam passes through shallow standing water, it creates bright caustic patterns on the submerged floor. Stalactites cast long, swinging shadows as the flashlight moves.
```

**Flower Time-Lapse Continuity**
```text
A flower blooming and wilting over two weeks, one photo per day. Same vase, same window, same angle. Light changes with weather.
```

**Tracking Shot Street Escape**
```text
TRACKING SHOT follows her from behind as she runs through the street. Sari fabric flows and trails behind her, catching the wind. CLOSE-UP on bare feet hitting the ground. Fabric billowing. She glances back. Keeps running. Determined.
```

### 🛍️ Advertising & Product Storytelling

**Voice Assistant Day-In-The-Life**
```text
A time-lapse of a family using a home voice assistant throughout the day. From setting morning alarms, playing music, checking the weather, and controlling smart lights, the product integrates seamlessly into their daily life.
```

### 🎨 Animation & Stylized Visuals

**1990s Action Cartoon Firebending**
```text
1990s action cartoon style. A young martial artist performs a firebending kata. The flames are hand-drawn with thick outlines and bold orange-yellow gradients. Dynamic camera swoops around the character. The fighting stance shows anime influence while maintaining Western animation proportions. Smoke effects use the signature layered look of the era.
```

**Cyberpunk Android Repair Bay**
```text
Cyberpunk anime style. A female android sits in a maintenance chair as robotic arms repair her damaged arm. The skin panel is open, revealing intricate servos and fiber-optic cables beneath. Her eyes are blank and unfocused during the repair cycle. Neon city lights filter through rain-streaked windows. Cool blue and pink color palette with high contrast shadows.
```

### 📱 Social, Viral & UGC-Style Concepts

**Graduation Banner Chaos**
```text
A massive "CONGRATULATIONS GRADUATES" banner being unfurled across a university building by maintenance workers on the roof. The wind catches it mid-unfurl, turning it into a sail that nearly lifts one worker off their feet. Coworkers grab him, everyone laughs, and the banner finally drops into place. Below, students start taking selfies immediately.
```

### 💡 Prompt Engineering Tips for HappyHorse 1.0

- **Name the shot.** Tokens like `TRACKING SHOT`, `CLOSE-UP`, `WIDE ANGLE`, `TIME-LAPSE` meaningfully change camera behavior.
- **Specify the style bucket.** `1990s action cartoon style`, `cyberpunk anime style`, `cinematic 35mm film` — style tokens front-load the aesthetic.
- **Keep motion concrete.** "Wind catches it mid-unfurl" beats "it moves dramatically."
- **For I2V, describe the motion, not the subject.** The image already provides the subject — the prompt should describe what changes.

---

## 🔗 Official Resources
- **API Provider**: [MuAPI.ai](https://muapi.ai) — get your `MUAPI_API_KEY` and access the playground here
- **Upgrade plan**: [muapi.ai/topup#plans](https://muapi.ai/topup#plans) — Pro ($20/mo) or Business ($100/mo) unlocks early access to new models including Happy Horse 1.0
- **Prompt Attribution**: Community prompts in the library above are adapted from [ZeroLu/awesome-happy-horse](https://github.com/ZeroLu/awesome-happy-horse) (CC BY 4.0).

## 📄 License
The Python wrapper is MIT licensed — see the [LICENSE](LICENSE) file.
Prompt text in the [Prompt Library](#-prompt-library) is adapted from [ZeroLu/awesome-happy-horse](https://github.com/ZeroLu/awesome-happy-horse), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---

**Keywords**: HappyHorse 1.0 API, Awesome HappyHorse 1.0, HappyHorse 1.0 Prompts, Alibaba HappyHorse, AI Video Generator, Text-to-Video AI, Image-to-Video API, HappyHorse Python SDK, Alibaba Video AI, MuAPI, Video Generation API, Native 1080p AI Video, AI Video Creation, HappyHorse API Documentation, HappyHorse I2V, HappyHorse T2V, AI Movie Generator, Python Video API, HappyHorse Tutorial, #1 AI Video Model.
