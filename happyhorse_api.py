import os
import requests
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Supported aspect ratios on HappyHorse 1.0 (matches the muapi schema).
SUPPORTED_ASPECT_RATIOS = ("16:9", "9:16", "1:1", "4:3", "3:4")
SUPPORTED_RESOLUTIONS = ("1080p", "720p")
SUPPORTED_AUDIO_SETTINGS = ("auto", "origin")
MIN_DURATION = 4
MAX_DURATION = 15
MAX_REFERENCE_IMAGES = 9       # reference-to-video
MAX_EDIT_REFERENCE_IMAGES = 5  # video-edit


class HappyHorseAPI:
    """HappyHorse 1.0 API client for muapi.ai.

    NOTE: HappyHorse 1.0 on muapi is currently in a closed playground beta.
    API-key access will return 403 until it goes GA; Pro- or Business-plan
    users can try it today inside the muapi playground. See the project
    README for details.
    """

    def __init__(self, api_key=None):
        """
        Initialize the HappyHorse 1.0 API client.

        :param api_key: Your muapi.ai API key. Defaults to the MUAPI_API_KEY
                        environment variable.
        """
        self.api_key = api_key or os.getenv("MUAPI_API_KEY")
        if not self.api_key:
            raise ValueError("API Key is required. Set MUAPI_API_KEY in .env or pass it to the constructor.")

        self.base_url = "https://api.muapi.ai/api/v1"
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    @staticmethod
    def _validate_common(aspect_ratio, duration, resolution):
        if aspect_ratio not in SUPPORTED_ASPECT_RATIOS:
            raise ValueError(
                f"aspect_ratio must be one of {SUPPORTED_ASPECT_RATIOS}, got {aspect_ratio!r}"
            )
        if not (MIN_DURATION <= int(duration) <= MAX_DURATION):
            raise ValueError(
                f"duration must be between {MIN_DURATION} and {MAX_DURATION} seconds, got {duration}"
            )
        if resolution not in SUPPORTED_RESOLUTIONS:
            raise ValueError(
                f"resolution must be one of {SUPPORTED_RESOLUTIONS}, got {resolution!r}"
            )

    def text_to_video(self, prompt, aspect_ratio="16:9", duration=5, resolution="1080p"):
        """
        Submit a HappyHorse 1.0 Text-to-Video (T2V) task.

        Returns {"request_id": ..., "status": "processing"}; poll with get_result /
        wait_for_completion. Choose `resolution="720p"` for ~half the cost of 1080p.

        :param prompt: The text prompt describing the video.
        :param aspect_ratio: One of '16:9', '9:16', '1:1', '4:3', '3:4'.
        :param duration: Video duration in seconds (4-15, default 5).
        :param resolution: '1080p' (default) or '720p'. 720p costs ~half of 1080p.
        :return: JSON response with request_id.
        """
        self._validate_common(aspect_ratio, duration, resolution)
        endpoint = f"{self.base_url}/happy-horse-1-text-to-video-{resolution}"
        payload = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "duration": int(duration),
        }
        return self._post_request(endpoint, payload)

    def image_to_video(self, prompt, images_list, aspect_ratio="16:9", duration=5, resolution="1080p"):
        """
        Submit a HappyHorse 1.0 Image-to-Video (I2V) task.

        The first URL in images_list is used as the start frame; the generated
        clip animates outward from it. Choose `resolution="720p"` for ~half the
        cost of 1080p.

        :param prompt: Optional text prompt guiding the motion (can be empty).
        :param images_list: Single-element list with the start-frame image URL.
        :param aspect_ratio: One of '16:9', '9:16', '1:1', '4:3', '3:4'.
        :param duration: Video duration in seconds (4-15, default 5).
        :param resolution: '1080p' (default) or '720p'.
        :return: JSON response with request_id.
        """
        self._validate_common(aspect_ratio, duration, resolution)
        if not images_list:
            raise ValueError("images_list must contain the start-frame image URL.")
        endpoint = f"{self.base_url}/happy-horse-1-image-to-video-{resolution}"
        payload = {
            "prompt": prompt or "",
            "images_list": list(images_list)[:1],
            "aspect_ratio": aspect_ratio,
            "duration": int(duration),
        }
        return self._post_request(endpoint, payload)

    def reference_to_video(self, prompt, images_list, aspect_ratio="16:9", duration=5, resolution="1080p", seed=None):
        """
        Submit a HappyHorse 1.0 Reference-to-Video task.

        Generate a video from a text prompt and 1–9 reference images. Unlike
        Image-to-Video (which uses a single start frame), reference-to-video
        treats every supplied image as a *style/subject reference* — the model
        learns the look, characters, or environment from the references and
        renders the prompt accordingly. Choose `resolution="720p"` for ~half
        the cost of 1080p.

        :param prompt: Text describing the video content.
        :param images_list: 1–9 reference image URLs. JPEG/PNG/WEBP, ≥400px
                            shortest side, ≤10 MB each.
        :param aspect_ratio: One of '16:9', '9:16', '1:1', '4:3', '3:4'.
        :param duration: Video duration in seconds (4-15, default 5).
        :param resolution: '1080p' (default) or '720p'.
        :param seed: Optional integer seed for reproducibility.
        :return: JSON response with request_id.
        """
        self._validate_common(aspect_ratio, duration, resolution)
        if not images_list:
            raise ValueError("images_list must contain at least one reference image URL.")
        if len(images_list) > MAX_REFERENCE_IMAGES:
            raise ValueError(
                f"images_list must contain at most {MAX_REFERENCE_IMAGES} reference images, got {len(images_list)}."
            )
        endpoint = f"{self.base_url}/happy-horse-1-reference-to-video-{resolution}"
        payload = {
            "prompt": prompt,
            "images_list": list(images_list),
            "aspect_ratio": aspect_ratio,
            "duration": int(duration),
        }
        if seed is not None:
            payload["seed"] = int(seed)
        return self._post_request(endpoint, payload)

    def video_edit(self, prompt, video_url, images_list=None, audio_setting="auto", resolution="1080p", seed=None):
        """
        Submit a HappyHorse 1.0 Video Edit task.

        Transform an existing video using a natural-language edit instruction.
        Optionally supply 0–5 reference images to anchor characters, styles or
        elements that should appear in the edited output. Choose
        `resolution="720p"` for ~half the cost of 1080p.

        :param prompt: Edit instruction describing the change to apply.
        :param video_url: Source video URL. MP4 or MOV (H.264 recommended),
                          3–60 s, ≤100 MB, longer side ≤2160 px, shorter
                          side ≥320 px, frame rate >8 fps.
        :param images_list: Optional list of 0–5 reference image URLs.
                            JPEG/PNG/WEBP, ≥300 px each side, ≤10 MB each.
        :param audio_setting: 'auto' (regenerate audio to match the edit) or
                              'origin' (keep the source audio track).
        :param resolution: '1080p' (default) or '720p'.
        :param seed: Optional integer seed for reproducibility.
        :return: JSON response with request_id.
        """
        if resolution not in SUPPORTED_RESOLUTIONS:
            raise ValueError(
                f"resolution must be one of {SUPPORTED_RESOLUTIONS}, got {resolution!r}"
            )
        if audio_setting not in SUPPORTED_AUDIO_SETTINGS:
            raise ValueError(
                f"audio_setting must be one of {SUPPORTED_AUDIO_SETTINGS}, got {audio_setting!r}"
            )
        if not video_url:
            raise ValueError("video_url is required for video_edit.")
        if images_list and len(images_list) > MAX_EDIT_REFERENCE_IMAGES:
            raise ValueError(
                f"images_list must contain at most {MAX_EDIT_REFERENCE_IMAGES} reference images, got {len(images_list)}."
            )
        endpoint = f"{self.base_url}/happy-horse-1-video-edit-{resolution}"
        payload = {
            "prompt": prompt,
            "video_url": video_url,
            "audio_setting": audio_setting,
        }
        if images_list:
            payload["images_list"] = list(images_list)
        if seed is not None:
            payload["seed"] = int(seed)
        return self._post_request(endpoint, payload)

    def _post_request(self, endpoint, payload):
        response = requests.post(endpoint, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def upload_file(self, file_path):
        """
        Upload a local file (image or video) to muapi for use in generation tasks.

        :param file_path: Path to the local file to upload.
        :return: JSON response containing the URL of the uploaded file.
        """
        endpoint = f"{self.base_url}/upload_file"
        headers = {"x-api-key": self.api_key}
        with open(file_path, "rb") as file_data:
            files = {"file": file_data}
            response = requests.post(endpoint, headers=headers, files=files)
        response.raise_for_status()
        return response.json()

    def get_result(self, request_id):
        """
        Poll for the result of a HappyHorse generation task.

        :param request_id: The request_id returned from a generation call.
        :return: JSON response with status and outputs.
        """
        endpoint = f"{self.base_url}/predictions/{request_id}/result"
        response = requests.get(endpoint, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def wait_for_completion(self, request_id, poll_interval=5, timeout=600):
        """
        Block until a HappyHorse generation task completes and return the result.

        :param request_id: The request_id returned from a generation call.
        :param poll_interval: Seconds between status polls (default 5).
        :param timeout: Maximum seconds to wait before raising TimeoutError (default 600).
        :return: Completed result JSON with an 'outputs' list (muapi-hosted video URLs).
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            result = self.get_result(request_id)
            status = result.get("status")

            if status == "completed":
                return result
            elif status == "failed":
                raise Exception(f"Video generation failed: {result.get('error')}")

            print(f"Status: {status}. Waiting {poll_interval} seconds...")
            time.sleep(poll_interval)

        raise TimeoutError("Timed out waiting for HappyHorse video generation to complete.")


if __name__ == "__main__":
    try:
        api = HappyHorseAPI()
        prompt = "A cinematic aerial shot of a coastal city at golden hour, waves crashing against cliffs, birds flying"

        print(f"Submitting T2V task with prompt: {prompt}")
        submission = api.text_to_video(prompt=prompt, duration=10)
        request_id = submission.get("request_id")
        print(f"Task submitted. Request ID: {request_id}")

        print("Waiting for completion...")
        result = api.wait_for_completion(request_id)
        print(f"Generation completed! Video URL: {result.get('outputs', [None])[0]}")

    except Exception as e:
        print(f"Error: {e}")
