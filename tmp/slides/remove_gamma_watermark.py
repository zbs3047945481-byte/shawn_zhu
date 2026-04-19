import shutil
import tempfile
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET


INPUT = Path("/Users/zhubingshuo/Desktop/桌面文件/毕设资料/中期报告ppt.pptx")
OUTPUT = Path("/Users/zhubingshuo/Desktop/桌面文件/毕设资料/中期报告ppt_去Gamma.pptx")

P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"

NS = {"p": P_NS, "a": A_NS}

ET.register_namespace("a", A_NS)
ET.register_namespace("r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships")
ET.register_namespace("p", P_NS)


def is_gamma_badge(pic):
    off = pic.find(".//a:xfrm/a:off", NS)
    ext = pic.find(".//a:xfrm/a:ext", NS)
    if off is None or ext is None:
        return False
    x = int(off.attrib["x"])
    y = int(off.attrib["y"])
    cx = int(ext.attrib["cx"])
    cy = int(ext.attrib["cy"])
    return x == 12839215 and y == 7749540 and cx == 1722605 and cy == 411480


with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)
    with zipfile.ZipFile(INPUT) as zf:
        zf.extractall(tmp)

    slide_layout_dir = tmp / "ppt" / "slideLayouts"
    removed = []

    for layout_path in sorted(slide_layout_dir.glob("slideLayout*.xml")):
        tree = ET.parse(layout_path)
        root = tree.getroot()
        sp_tree = root.find(".//p:spTree", NS)
        if sp_tree is None:
            continue
        pics = list(sp_tree.findall("p:pic", NS))
        changed = False
        for pic in pics:
            if is_gamma_badge(pic):
                sp_tree.remove(pic)
                changed = True
        if changed:
            tree.write(layout_path, encoding="utf-8", xml_declaration=True)
            removed.append(layout_path.name)

    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in tmp.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(tmp))

    print("removed_from:", ", ".join(removed))
    print("saved:", OUTPUT)
