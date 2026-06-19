#!/usr/bin/env python3
"""Convert a Gradio JSON conversation log into a styled HTML page.

Edit ``FILENAME`` below to point at the log you want to convert.
Output is written to ``scripts/output/<FILENAME>.html``.
"""

import json
import html
import re
import sys
from pathlib import Path

from PIL import Image
import base64
from io import BytesIO

from pragmabot.utils import get_package_path

# --- Edit this to point at the log you want to convert ---
FILENAME = "20260317_145135_put_the_orange_on_the_plate"
INPUT_PATH = get_package_path() / "data" / "logs" / f"{FILENAME}.json"
OUTPUT_PATH = Path(__file__).parent / "output" / f"{FILENAME}.html"


def compress_base64_image(b64_str: str, max_size=(400, 300), quality=75) -> str:
    """
    Compress a base64-encoded image string.
    Supports both:
      - "data:image/png;base64,iVBOR..."
      - "image/png;base64,iVBOR..." (non-standard but used in your logs)
    Returns a new base64 string (usually smaller).
    """
    try:
        # Normalize: ensure it starts with "data:image/..."
        if b64_str.startswith("image/"):
            # Convert "image/png;base64,..." → "data:image/png;base64,..."
            b64_str = "data:" + b64_str

        if not b64_str.startswith("data:image/"):
            return b64_str  # Not a base64 image we can handle

        # Split header and data
        if "," not in b64_str:
            return b64_str
        header, data = b64_str.split(",", 1)

        # Decode image
        img_data = base64.b64decode(data)
        img = Image.open(BytesIO(img_data))

        # Handle transparency: convert RGBA to RGB with white background for JPEG
        if img.mode == "RGBA":
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
            img = background
            output_format = "JPEG"
        elif img.mode == "P" and "transparency" in img.info:
            # Palette mode with transparency → convert to RGBA then to RGB
            img = img.convert("RGBA")
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            img = background
            output_format = "JPEG"
        else:
            # For grayscale or RGB, prefer JPEG for compression unless it's clearly a mask
            output_format = "JPEG"

        # Resize if larger than max_size
        img.thumbnail(max_size, Image.Resampling.LANCZOS)

        # Save to buffer
        buffer = BytesIO()
        if output_format == "JPEG":
            img.save(buffer, format="JPEG", quality=quality, optimize=True)
        else:
            img.save(buffer, format="PNG", optimize=True)

        # Re-encode
        compressed_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        new_header = f"data:image/{output_format.lower()};base64,"
        return new_header + compressed_b64

    except Exception as e:
        print(f"⚠️ Warning: Image compression failed: {e}", file=sys.stderr)
        return b64_str  # Return original if anything goes wrong


def safe_format_content(content: str) -> str:
    """
    Render Markdown-like content safely in HTML:
    - Headings (#, ##, ###)
    - Images ![](url) → compress base64 images
    - Code blocks ```json ... ```
    - Escape everything else
    """
    # Step 1: Split into code blocks and non-code text
    parts = re.split(r"(```(?:\w*)?\n[\s\S]*?\n```)", content)

    result_parts = []
    for part in parts:
        if part.startswith("```") and part.endswith("```"):
            # This is a code block
            code_content = part[3:-3].strip()
            lang_match = re.match(r"^(\w+)\n(.*)", code_content, re.DOTALL)
            if lang_match:
                lang = lang_match.group(1).lower()
                code = lang_match.group(2)
            else:
                lang = ""
                code = code_content

            pretty_code = code

            is_json = lang == "json"
            if lang == "json" or (not lang and code.strip().startswith(("{", "["))):
                try:
                    parsed = json.loads(code)
                    pretty_code = json.dumps(parsed, indent=2)
                    is_json = True
                except (json.JSONDecodeError, ValueError):
                    pass

            safe_code = html.escape(pretty_code)
            lang_class = " language-json" if is_json else ""
            result_parts.append(f'<pre><code class="code-block{lang_class}">{safe_code}</code></pre>')
        else:
            # Non-code part: process headings, images, and escape
            lines = part.split("\n")
            processed_lines = []
            for line in lines:
                # Handle images: compress base64 ones
                def replace_img(match):
                    alt = match.group(1) or ""
                    url = match.group(2)
                    # Compress if it's a base64 image
                    if url.startswith("data:image/") or url.startswith("image/"):
                        compressed_url = compress_base64_image(url)
                        return f'\n<img src="{html.escape(compressed_url)}" alt="{html.escape(alt)}">'
                    else:
                        # Not a base64 image → escape as plain text
                        return html.escape(match.group(0))

                line = re.sub(r"!\[([^\]]*)]\(([^)]+)\)", replace_img, line)

                # Handle headings: shift down by 1 (e.g., # → h2, ## → h3, ### → h4)
                heading_match = re.match(r"^(#{1,3})\s+(.*)", line)
                if heading_match:
                    md_level = len(heading_match.group(1))  # 1, 2, or 3
                    html_level = md_level + 1  # → 2, 3, or 4
                    text = html.escape(heading_match.group(2))
                    line = f"<h{html_level}>{text}</h{html_level}>"
                else:
                    # Escape remaining line (but preserve <img>)
                    if "<img" in line:
                        subparts = re.split(r"(<img[^>]+>)", line)
                        line = "".join(sp if sp.startswith("<img") else html.escape(sp) for sp in subparts)
                    else:
                        line = html.escape(line)

                processed_lines.append(line)
            result_parts.append("\n".join(processed_lines))

    final_html = "\n".join(result_parts)

    # Strip physical newlines immediately preceding or following block elements
    # This prevents `pre-wrap` from double-stacking space next to CSS margins
    final_html = re.sub(r"\n+(<(?:h[1-6]|pre|img)[^>]*>)", r"\1", final_html)
    final_html = re.sub(r"(</(?:h[1-6]|pre)>)\n+", r"\1", final_html)

    return final_html


def format_gradio_log_to_html(log_str: str) -> str:
    log_data = json.loads(log_str)

    if not isinstance(log_data, list):
        raise ValueError("Log must be a JSON array")

    for i, msg in enumerate(log_data):
        if not (isinstance(msg, dict) and "role" in msg and "content" in msg):
            raise ValueError(f"Invalid message format at index {i}")

    # Merge consecutive messages from same role
    merged = []
    for msg in log_data:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] += "\n" + msg["content"]
        else:
            merged.append({"role": msg["role"], "content": msg["content"]})

    message_html = ""
    for msg in merged:
        role = msg["role"]
        content = msg["content"]

        formatted_content = safe_format_content(content)

        # Detect system messages
        is_system = (
            "### system prompt" in content or "you are a helpful assistant" in content.lower() or role == "system"
        )
        sys_class = " system-message" if is_system else ""

        # Add alignment class: user → right, others → left
        align_class = " user-align" if role in ["user", "system"] else " assistant-align"

        message_html += f"""
        <div class="message {role}-message{sys_class}{align_class}">
            <div class="role">{role.capitalize()}</div>
            <div class="content">{formatted_content}</div>
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Log</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: white;
            margin: 0;
            padding: 20px;
            color: #333;
        }}
        .messages {{
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        .message {{
            padding: 16px 20px;
            border-radius: 12px;
            max-width: 85%;
            width: fit-content;
            line-height: 1.6;
        }}
        .assistant-align {{
            align-self: flex-start;
            background-color: #f0f0f0;
            border-left: 3px solid #757575;
        }}
        .user-align {{
            align-self: flex-end;
            background-color: #fff8e1;
            border-left: 3px solid #ffa000 ;
        }}
        .system-message {{
            align-self: flex-end;
            background-color: #fff8e1 !important;
            border-left: 3px solid #ffa000 !important;
        }}
        .role {{
            font-weight: 600;
            margin-bottom: 8px;
            font-size: 0.9em;
            color: #555;
        }}
        .content {{
            white-space: pre-wrap;
            word-break: break-word;
        }}
        
        /* Controlled Spacing for Headings */
        .content h2, 
        .content h3, 
        .content h4, 
        .content h5, 
        .content h6 {{
            margin-top: 1.2em;
            margin-bottom: 0.5em;
            font-weight: 600;
            color: #1a1a1a;
        }}

        /* Clean up top/bottom edges of the message bubble */
        .content > *:first-child {{
            margin-top: 0;
        }}
        .content > *:last-child {{
            margin-bottom: 0;
        }}

        /* Improved Image Presentation */
        .content img {{
            max-width: 100%;
            border-radius: 8px;
            margin: 16px 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            display: block;
        }}

        pre {{
            background: #f8f8f8;
            padding: 12px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 12px 0;
            white-space: pre;
            border: 1px solid #e0e0e0;
        }}
        code {{
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
            font-size: 0.95em;
        }}
        code.language-json {{
            font-size: 0.85em;
        }}
    </style>
</head>
<body>
    <div class="messages">
        {message_html}
    </div>
</body>
</html>"""


if __name__ == "__main__":
    if not INPUT_PATH.exists():
        print(f"Error: Input file not found: {INPUT_PATH}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(INPUT_PATH, "r", encoding="utf-8") as f:
            log_str = f.read()

        html_output = format_gradio_log_to_html(log_str)

        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            f.write(html_output)

        print(f"✅ HTML saved to: {OUTPUT_PATH}")
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
