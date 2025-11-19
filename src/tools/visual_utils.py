from PIL import Image, ImageDraw, ImageFont

def draw_cursor(image: Image.Image, x: int, y: int, color: str = "red", radius: int = 10) -> Image.Image:
    """
    Draws a visual cursor on the image at coordinates (x, y) with label 'cursor'.
    Operates directly on the original image size.
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    w, h = img_copy.size
    
    # Clamp coordinates to current image dimensions
    x = max(0, min(x, w-1))
    y = max(0, min(y, h-1))
    
    # 1. Crosshair lines
    line_len = radius * 1.5
    draw.line([(x - line_len, y), (x + line_len, y)], fill=color, width=4)
    draw.line([(x, y - line_len), (x, y + line_len)], fill=color, width=4)
    
    # 2. Circle
    draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], outline=color, width=4)
    
    # 3. Text Label
    try:
        # Try to use a proportional font size based on image width
        font_size = max(15, int(w * 0.02)) 
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
        
    label = "cursor"
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    
    # Position text: Try top-right of cursor, fallback if out of bounds
    text_x = x + radius + 5
    text_y = y - radius - text_h
    
    if text_x + text_w > w: text_x = x - radius - text_w - 5
    if text_y < 0: text_y = y + radius + 5

    # Draw background and text
    draw.rectangle([text_x - 2, text_y - 2, text_x + text_w + 2, text_y + text_h + 2], fill="black")
    draw.text((text_x, text_y), label, font=font, fill="white")
    
    return img_copy

def visualize_trajectory(base_image: Image.Image, cursor_path: list, actions: list, gt_bbox: list, success: bool) -> Image.Image:
    """
    Draws the full execution path on a single image for Evaluation visualization.
    - Green Box: Ground Truth Target
    - Cyan Lines: Movement Path
    - Red X: End Point
    - Image Border: Green (Success) / Red (Fail)
    """
    # 1. Create a canvas (Dim original image slightly to make path pop)
    canvas = base_image.convert("RGBA")
    overlay = Image.new('RGBA', canvas.size, (0, 0, 0, 40)) # Darken slightly (alpha 40)
    canvas = Image.alpha_composite(canvas, overlay).convert("RGB")
    
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    
    # 2. Draw Ground Truth BBox (Green)
    if gt_bbox:
        # ScreenSpot format: [x1, y1, x2, y2]
        draw.rectangle(gt_bbox, outline="#00FF00", width=5)
        # Label GT
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        # Draw label background
        draw.rectangle([gt_bbox[0], gt_bbox[1]-20, gt_bbox[0]+60, gt_bbox[1]], fill="#00FF00")
        draw.text((gt_bbox[0]+5, gt_bbox[1]-18), "TARGET", fill="black", font=font)

    # 3. Draw Path (Cyan Lines)
    # cursor_path is a list of (x, y) tuples
    if len(cursor_path) > 1:
        draw.line(cursor_path, fill="cyan", width=3)

    # 4. Draw Key Points
    if cursor_path:
        # Start Point (Small White Circle)
        sx, sy = cursor_path[0]
        draw.ellipse([sx-6, sy-6, sx+6, sy+6], fill="white", outline="black", width=2)

        # End Point / Click Point (Red Crosshair)
        ex, ey = cursor_path[-1]
        r = 10
        draw.line([ex-r, ey-r, ex+r, ey+r], fill="red", width=4)
        draw.line([ex-r, ey+r, ex+r, ey-r], fill="red", width=4)
        draw.ellipse([ex-r, ey-r, ex+r, ey+r], outline="red", width=3)

    # 5. Add Border based on Success/Fail
    border_color = "#00FF00" if success else "#FF0000"
    border_width = 8
    draw.rectangle([0, 0, w-1, h-1], outline=border_color, width=border_width)
    
    return canvas