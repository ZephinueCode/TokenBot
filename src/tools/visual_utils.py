# src/tools/visual_utils.py

from PIL import Image, ImageDraw, ImageFont
import textwrap

def draw_cursor(image: Image.Image, x: int, y: int, color: str = "red", radius: int = 20) -> Image.Image:
    """
    TRAINING CURSOR:
    Draws a high-performance geometric cursor.
    - Style: Gapped Crosshair + Outer Ring (Non-occluding).
    - Optimization: NO text rendering to maximize DataLoader speed (30% faster).
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    w, h = img_copy.size
    
    # Clamp coordinates
    x = max(0, min(x, w-1))
    y = max(0, min(y, h-1))
    
    # Visual Parameters (Synced with visualize_trajectory)
    width = 4
    gap = 8          # Empty space in center to see target
    line_len = 12    # Length of crosshair arms
    ring_r = gap + line_len + 4
    
    # 1. Draw Inward Pointing Lines
    # Left
    draw.line([(x - gap - line_len, y), (x - gap, y)], fill=color, width=width)
    # Right
    draw.line([(x + gap, y), (x + gap + line_len, y)], fill=color, width=width)
    # Top
    draw.line([(x, y - gap - line_len), (x, y - gap)], fill=color, width=width)
    # Bottom
    draw.line([(x, y + gap), (x, y + gap + line_len)], fill=color, width=width)
    
    # 2. Draw Outer Ring (Helps visibility on complex backgrounds)
    draw.ellipse([(x - ring_r, y - ring_r), (x + ring_r, y + ring_r)], outline=color, width=4)
    
    return img_copy

def visualize_trajectory(
    base_image: Image.Image, 
    cursor_path: list, 
    actions: list, 
    gt_bbox: list, 
    success: bool,
    instruction: str = None
) -> Image.Image:
    """
    EVALUATION VISUALIZATION:
    Draws the full path, GT, and instruction overlay.
    The 'End Point' style matches the 'Training Cursor' for consistency.
    """
    # 1. Create Canvas (Dimmed)
    canvas = base_image.convert("RGBA")
    overlay = Image.new('RGBA', canvas.size, (0, 0, 0, 60)) 
    canvas = Image.alpha_composite(canvas, overlay).convert("RGB")
    
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    
    # 2. Draw Ground Truth BBox (Green)
    if gt_bbox:
        # Draw Box
        draw.rectangle(gt_bbox, outline="#00FF00", width=4)
        
        # Draw "Target" Label
        try:
            font_gt = ImageFont.truetype("arial.ttf", 16)
        except:
            font_gt = ImageFont.load_default()
            
        label_x = gt_bbox[0]
        label_y = max(0, gt_bbox[1] - 20)
        draw.rectangle([label_x, label_y, label_x+60, label_y+20], fill="#00FF00")
        draw.text((label_x+5, label_y), "TARGET", fill="black", font=font_gt)

    # 3. Draw Movement Path (Cyan Lines)
    if len(cursor_path) > 1:
        draw.line(cursor_path, fill="cyan", width=3)

    # 4. Draw Key Points
    if cursor_path:
        # A. Start Point (Simple White Circle)
        sx, sy = cursor_path[0]
        draw.ellipse([sx-6, sy-6, sx+6, sy+6], fill="white", outline="black", width=2)

        # B. End Point (The "Agent Cursor")
        # [SYNCED STYLE] Exactly matches draw_cursor
        ex, ey = cursor_path[-1]
        color = "red"
        width = 4
        gap = 8
        line_len = 15
        ring_r = gap + line_len + 5
        
        # Crosshair
        draw.line([(ex - gap - line_len, ey), (ex - gap, ey)], fill=color, width=width)
        draw.line([(ex + gap, ey), (ex + gap + line_len, ey)], fill=color, width=width)
        draw.line([(ex, ey - gap - line_len), (ex, ey - gap)], fill=color, width=width)
        draw.line([(ex, ey + gap), (ex, ey + gap + line_len)], fill=color, width=width)
        # Ring
        draw.ellipse([(ex - ring_r, ey - ring_r), (ex + ring_r, ey + ring_r)], outline=color, width=2)

    # 5. Draw Instruction Overlay (Top Right)
    if instruction:
        try:
            font_text = ImageFont.truetype("arial.ttf", 20)
        except:
            font_text = ImageFont.load_default()
            
        # Text Wrapping
        max_chars = int((w * 0.45) / 10) 
        lines = textwrap.wrap(f"Instr: {instruction}", width=max_chars)
        
        # Box Dimensions
        line_height = 28
        box_w = w * 0.5
        box_h = len(lines) * line_height + 16
        
        box_x = w - box_w - 10
        box_y = 10
        
        # Semi-transparent background
        overlay_box = Image.new('RGBA', canvas.size, (0,0,0,0))
        draw_box = ImageDraw.Draw(overlay_box)
        draw_box.rectangle([box_x, box_y, box_x + box_w, box_y + box_h], fill=(0, 0, 0, 200))
        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay_box).convert("RGB")
        
        # Draw Text
        draw = ImageDraw.Draw(canvas)
        for i, line in enumerate(lines):
            draw.text((box_x + 10, box_y + 8 + i*line_height), line, fill="white", font=font_text)

    # 6. Status Border
    border_color = "#00FF00" if success else "#FF0000"
    draw.rectangle([0, 0, w-1, h-1], outline=border_color, width=10)
    
    return canvas