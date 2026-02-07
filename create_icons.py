# create_icons.py - Generate simple icons for extension

from PIL import Image, ImageDraw, ImageFont
import os

os.makedirs('chrome-extension/icons', exist_ok=True)

def create_icon(size):
    # Create image
    img = Image.new('RGB', (size, size), color='#667eea')
    draw = ImageDraw.Draw(img)
    
    # Draw shield shape (simple rectangle for now)
    margin = size // 4
    draw.rectangle([margin, margin, size-margin, size-margin], 
                   fill='white', outline='#667eea', width=3)
    
    # Draw checkmark or R letter
    try:
        font = ImageFont.truetype("arial.ttf", size//2)
    except:
        font = ImageFont.load_default()
    
    text = "R"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ((size - text_width) // 2, (size - text_height) // 2 - size//10)
    
    draw.text(position, text, fill='#667eea', font=font)
    
    # Save
    img.save(f'chrome-extension/icons/icon{size}.png')
    print(f"✓ Created icon{size}.png")

# Create all sizes
create_icon(16)
create_icon(48)
create_icon(128)

print("✓ All icons created!")