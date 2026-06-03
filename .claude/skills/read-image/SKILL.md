---
name: read-image
description: Analyze and understand images using Xiaomi MiMo-V2.5 Vision API. Use this skill whenever the user mentions reading, analyzing, describing, or asking questions about images, photos, screenshots, charts, diagrams, scans, or any visual content — even if they don't explicitly say "read image." Also use when the user asks "what's in this picture", "describe this photo", "explain this chart", "OCR this screenshot", or similar visual queries involving an image file path.
---

# Read Image

Analyze images using the Xiaomi MiMo-V2.5 Vision API. This skill bundles a Python
script that handles base64 encoding, API communication, and response parsing.

## When to use this skill

Invoke this skill whenever the user provides an image file path and wants to:
- Describe what's in the image
- Answer a question about visual content
- Extract text from a screenshot or scan (OCR)
- Interpret charts, graphs, or diagrams
- Compare or analyze visual elements

The skill calls the MiMo OpenAI-compatible chat completions endpoint with
`mimo-v2.5`, which has native multimodal support including a 1M context window.

## How it works

The bundled `scripts/read_image.py` script:
1. Reads the image file from disk and detects its MIME type
2. Encodes the image to base64 and wraps it as a `data:` URL
3. Sends it to the MiMo API as an `image_url` content part in a chat message
4. Returns the model's text response

The API format is OpenAI-compatible: the image goes into the `content` array
alongside the text prompt, using the standard `image_url` type with a base64
data URL.

## Usage

```
python <skill-dir>/scripts/read_image.py <image_path> [options]
```

### Required arguments

| Argument | Description |
|----------|-------------|
| `image` | Path to the image file |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--prompt`, `-p` | `"请详细描述这张图片的内容。"` | Question or instruction about the image |
| `--stream`, `-s` | off | Stream the response token-by-token |
| `--model` | `mimo-v2.5` | Model to use. See supported models below |
| `--temperature` | model default | Sampling temperature (0–1.5) |
| `--max-tokens` | 4096 | Max output tokens |
| `--system-prompt` | none | System message to set model behavior |

### Supported models

| Model | Notes |
|-------|-------|
| `mimo-v2.5` | Default. Native multimodal, 1M context, 32K max output |
| `mimo-v2.5-pro` | Enhanced reasoning, 131K max output |
| `mimo-v2-omni` | Previous-gen multimodal |
| `mimo-v2-flash` | Fast/lightweight, 65K max output |

All support image input. Audio/video/thinking features vary by model — check the
[docs](https://platform.xiaomimimo.com/docs/zh-CN/api/chat/openai-api) for details.

### Examples

Describe an image:
```
python scripts/read_image.py photo.jpg
```

Ask a question about a chart:
```
python scripts/read_image.py chart.png --prompt "What trend does this chart show?"
```

Extract text from a screenshot:
```
python scripts/read_image.py screenshot.png --prompt "OCR all visible text from this image. Output the text exactly as it appears."
```

Stream with a custom system prompt:
```
python scripts/read_image.py diagram.jpg \
  --system-prompt "You are a technical architect. Respond in Chinese." \
  --prompt "Explain this architecture diagram in detail" \
  --stream
```

## Supported image formats

PNG, JPEG, GIF, WebP, BMP. Other common image formats should also work — the
script uses Python's `mimetypes` module to detect the MIME type.

## Configuration

The script reads the API key from `xiaomi_api_key` in the project root `.env` file.
Set it like this:

```
xiaomi_api_key=tp-xxxxxxxxxxxxxxxx
```

Get an API key from the [MiMo Open Platform console](https://platform.xiaomimimo.com).

## Error handling

If the API key is missing, the script exits with a clear error message. If the
API returns an HTTP error, the script raises it with the response body for
debugging. For unexpected response structures, it prints the raw JSON so you
can inspect what went wrong.
