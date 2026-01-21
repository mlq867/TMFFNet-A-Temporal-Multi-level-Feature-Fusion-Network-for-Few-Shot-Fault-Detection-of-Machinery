from PIL import Image

def vertical_concat_png(image_paths, save_path):
    """
    竖直拼接多张 PNG 图片（不缩放、不失真）

    :param image_paths: list[str]  图片路径列表
    :param save_path: str         输出PNG路径
    """
    images = [Image.open(p) for p in image_paths]

    widths = [img.width for img in images]
    heights = [img.height for img in images]

    max_width = max(widths)
    total_height = sum(heights)

    result = Image.new("RGB", (max_width, total_height), (255, 255, 255))

    y_offset = 0
    for img in images:
        result.paste(img, (0, y_offset))
        y_offset += img.height

    result.save(save_path, format="PNG")


image_paths = [
    r"../Result/Figure/JDPump_50.png",
    r"../Result/Figure/JDPump_100.png",
    r"../Result/Figure/JDPump_500.png",
    r"../Result/Figure/JDPump_1000.png"
]

vertical_concat_png(image_paths, "../Result/Figure/merged.png")

