import json

from mcp.server.fastmcp import FastMCP

from happyhorse_api import HappyHorseAPI

# Initialize FastMCP server
mcp = FastMCP("HappyHorse 1.0 API Server")


def get_api():
    return HappyHorseAPI()


@mcp.tool()
def text_to_video(prompt: str, aspect_ratio: str = "16:9", duration: int = 5, resolution: str = "1080p") -> str:
    """Generate a HappyHorse 1.0 video from a text prompt.

    HappyHorse 1.0 is Alibaba's 15B-parameter Transformer video model, #1 on
    Artificial Analysis. Choose '1080p' (default) for native HD or '720p' for
    ~half the cost. Note that HappyHorse is currently a closed playground beta
    on muapi — API-key access will return 403 until it goes GA.

    :param prompt: Descriptive text prompt.
    :param aspect_ratio: One of '16:9', '9:16', '1:1', '4:3', '3:4'.
    :param duration: Duration in seconds (4-15, default 5).
    :param resolution: '1080p' (default) or '720p'.
    """
    api = get_api()
    result = api.text_to_video(prompt, aspect_ratio, duration, resolution)
    return json.dumps(result, indent=2)


@mcp.tool()
def image_to_video(prompt: str, images_list: list[str], aspect_ratio: str = "16:9", duration: int = 5, resolution: str = "1080p") -> str:
    """Animate a starting image into a HappyHorse 1.0 video.

    The first URL in images_list is used as the start frame; the generated
    clip animates outward from it. Choose '1080p' (default) for native HD or
    '720p' for ~half the cost.

    :param prompt: Optional text prompt guiding the motion (can be empty).
    :param images_list: Single-element list with the start-frame image URL.
    :param aspect_ratio: One of '16:9', '9:16', '1:1', '4:3', '3:4'.
    :param duration: Duration in seconds (4-15, default 5).
    :param resolution: '1080p' (default) or '720p'.
    """
    api = get_api()
    result = api.image_to_video(prompt, images_list, aspect_ratio, duration, resolution)
    return json.dumps(result, indent=2)


@mcp.tool()
def upload_file(file_path: str) -> str:
    """Upload a local file (image or video) to muapi for use in generation tasks.

    :param file_path: Local path to the file.
    """
    api = get_api()
    result = api.upload_file(file_path)
    return json.dumps(result, indent=2)


@mcp.tool()
def get_task_status(request_id: str) -> str:
    """Check the status and get results of a HappyHorse generation task.

    :param request_id: The ID returned from a generation tool call.
    """
    api = get_api()
    result = api.get_result(request_id)
    return json.dumps(result, indent=2)


if __name__ == "__main__":
    mcp.run()
