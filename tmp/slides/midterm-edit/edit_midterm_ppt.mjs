import fs from "fs/promises";
import path from "path";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";
import JSZip from "jszip";

const INPUT = "/Users/zhubingshuo/Desktop/桌面文件/毕设资料/中期报告ppt.pptx";
const OUTPUT = "/Users/zhubingshuo/Desktop/桌面文件/毕设资料/中期报告_修改版.pptx";
const PREVIEW_DIR = "/Users/zhubingshuo/Desktop/plug-and-play-feature-distillation-fl/tmp/slides/midterm-edit/outputs/previews";

function setShapeText(slide, shapeIndex, text, position = null) {
  const shape = slide.shapes.items[shapeIndex];
  shape.text.set(text);
  shape.text.autoFit = "shrinkText";
  if (position) {
    shape.position = position;
  }
  return shape;
}

function setImagePosition(slide, imageIndex, position) {
  const image = slide.images.items[imageIndex];
  image.position = position;
  return image;
}

function removeShapesFrom(slide, startIndex) {
  const removable = slide.shapes.items.slice(startIndex);
  for (const shape of removable) {
    shape.delete();
  }
}

async function exportPreview(slide, filename) {
  const preview = await slide.export({ format: "png", scale: 1 });
  const buffer = Buffer.from(await preview.arrayBuffer());
  await fs.writeFile(path.join(PREVIEW_DIR, filename), buffer);
}

async function reorderSlidesInPptx(filePath, slideNumbers) {
  const buffer = await fs.readFile(filePath);
  const zip = await JSZip.loadAsync(buffer);
  const relsPath = "ppt/_rels/presentation.xml.rels";
  const presentationPath = "ppt/presentation.xml";
  const relsXml = await zip.file(relsPath).async("string");
  const presentationXml = await zip.file(presentationPath).async("string");

  const ridBySlideNumber = {};
  const relRegex = /<Relationship\b[^>]*Type="[^"]*\/slide"[^>]*Target="\/ppt\/slides\/slide(\d+)\.xml"[^>]*Id="([^"]+)"[^>]*\/>/g;
  let relMatch;
  while ((relMatch = relRegex.exec(relsXml)) !== null) {
    ridBySlideNumber[Number(relMatch[1])] = relMatch[2];
  }

  const tagByRid = {};
  const tagRegex = /<p:sldId\b[^>]*r:id="([^"]+)"[^>]*\/>/g;
  let tagMatch;
  while ((tagMatch = tagRegex.exec(presentationXml)) !== null) {
    tagByRid[tagMatch[1]] = tagMatch[0];
  }

  const orderedTags = slideNumbers.map((slideNumber) => {
    const rid = ridBySlideNumber[slideNumber];
    if (!rid || !tagByRid[rid]) {
      throw new Error(`Missing slide mapping for slide${slideNumber}.xml`);
    }
    return tagByRid[rid];
  });

  const updatedPresentationXml = presentationXml.replace(
    /<p:sldIdLst>[\s\S]*?<\/p:sldIdLst>/,
    `<p:sldIdLst>${orderedTags.join("")}</p:sldIdLst>`,
  );

  zip.file(presentationPath, updatedPresentationXml);
  const outBuffer = await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" });
  await fs.writeFile(filePath, outBuffer);
}

await fs.mkdir(PREVIEW_DIR, { recursive: true });

const pptx = await FileBlob.load(INPUT);
const presentation = await PresentationFile.importPptx(pptx);

const contentsSlide = presentation.slides.getItem(1).duplicate();
contentsSlide.setIndex(2);
setShapeText(contentsSlide, 0, "目录");
setShapeText(
  contentsSlide,
  1,
  "本次汇报围绕阶段目标、系统实现进展以及当前关键问题与解决思路三部分展开。",
);
setShapeText(contentsSlide, 4, "第一部分  工作目标与任务要求");
setShapeText(
  contentsSlide,
  5,
  "围绕 Non-IID 联邦场景、核心模块实现与工程规范化建设，明确本阶段需要完成的关键任务和验收重点。",
);
setShapeText(contentsSlide, 8, "第二部分  目前已完成任务情况");
setShapeText(
  contentsSlide,
  9,
  "展示联邦训练闭环、类别原型机制、特征蒸馏与插件化扩展等阶段性成果，说明系统已经具备可运行基础。",
);
setShapeText(contentsSlide, 12, "第三部分  存在的问题和拟解决方法");
setShapeText(
  contentsSlide,
  13,
  "重点说明 Non-IID 构造、即插即用设计、特征共享安全稳定性以及特征蒸馏约束四个核心问题的解决思路。",
);

const goalSlide = presentation.slides.getItem(1);
setShapeText(goalSlide, 0, "本阶段工作目标与任务要求");
setShapeText(
  goalSlide,
  1,
  "本阶段核心目标：完成从理论分析向系统实现的过渡，形成可复现实验框架、可插拔蒸馏模块与中期总结材料，并为后续对比实验与论文撰写打下基础。",
);
setShapeText(goalSlide, 5, "围绕 Non-IID 联邦场景，完成标签偏斜、数量偏斜与特征异构的数据划分设计，使实验设置既贴近真实分布，又保持训练可运行与结果可解释。");
setShapeText(goalSlide, 9, "实现中间特征低维投影、类别原型提取与特征蒸馏模块，并通过统一接口接入经典联邦学习基线，验证模块具备独立部署与复用能力。");
setShapeText(goalSlide, 13, "对系统结构、训练流程与接口边界进行规范化整理，保证客户端与服务端职责清晰，为后续消融实验、参数调优和结果复现提供稳定工程基础。");

const problem1Slide = presentation.slides.getItem(4);
setShapeText(problem1Slide, 1, "关键难点在于：简单随机划分不足以体现真实 Non-IID 挑战，而过强异构又会导致客户端样本失衡和训练波动。为此，系统将真实性、异构性与可训练性作为同时约束的三个目标，在划分阶段联合设计标签分布、样本规模与特征扰动机制。", { left: 83, top: 256, width: 1370, height: 88 });
setImagePosition(problem1Slide, 0, { left: 83, top: 350, width: 407, height: 260 });
setImagePosition(problem1Slide, 1, { left: 548, top: 350, width: 407, height: 260 });
setImagePosition(problem1Slide, 2, { left: 1013, top: 350, width: 407, height: 260 });
setShapeText(problem1Slide, 3, "使用 Dirichlet 分布分别控制客户端样本量和类别权重，使不同客户端同时表现出数量偏斜与标签偏斜。", { left: 83, top: 654, width: 437, height: 92 });
setShapeText(problem1Slide, 5, "进一步为不同客户端采样缩放、偏置和噪声参数，在输入侧构造统计差异，模拟多源设备与采集环境带来的特征异构。", { left: 548, top: 654, width: 437, height: 92 });
setShapeText(problem1Slide, 7, "引入最小样本数检查与重试机制，避免极端划分破坏本地训练稳定性，使异构强度与实验可运行性保持平衡。", { left: 1013, top: 654, width: 437, height: 92 });

const problem2Slide = presentation.slides.getItem(5);
setShapeText(problem2Slide, 1, "若将蒸馏逻辑直接硬编码进联邦训练主流程，系统会迅速演变为强耦合实现：算法切换困难，维护成本高，也不利于迁移到其他联邦框架。项目因此将低侵入接入和可扩展复用作为设计原则，采用插件化架构组织增强模块。", { left: 83, top: 214, width: 1370, height: 92 });
setImagePosition(problem2Slide, 0, { left: 83, top: 314, width: 407, height: 246 });
setImagePosition(problem2Slide, 1, { left: 548, top: 314, width: 407, height: 246 });
setImagePosition(problem2Slide, 2, { left: 1013, top: 314, width: 407, height: 246 });
setShapeText(problem2Slide, 3, "通过注册表维护插件名与实现类映射，再由工厂函数依据配置完成实例构建，使主干流程只依赖统一入口而不绑定具体算法实现。", { left: 83, top: 610, width: 437, height: 100 });
setShapeText(problem2Slide, 5, "服务器端在轮次开始前下发辅助信息，在常规模型聚合后再聚合客户端附加信息，从而把增强逻辑嵌入关键节点而非改写整条主流程。", { left: 548, top: 610, width: 437, height: 100 });
setShapeText(problem2Slide, 7, "客户端侧分别在轮次初始化、batch 训练和本地训练结束后三个阶段开放插件入口，保证增强训练、附加信息生成与上传过程都能按需启停。", { left: 1013, top: 610, width: 437, height: 112 });

const problem3Slide = presentation.slides.getItem(6);
setShapeText(problem3Slide, 1, "上传类别原型虽然比直接共享原始数据更安全，但中间特征仍携带类别语义与表示幅值信息：一方面，精细原型可能暴露局部数据特征；另一方面，不同客户端的特征尺度差异会放大聚合偏置，削弱全局原型的稳定性。因此，该环节必须同时处理隐私风险、异常值抑制和聚合鲁棒性。", { left: 82, top: 170, width: 1372, height: 108 });
setImagePosition(problem3Slide, 0, { left: 82, top: 290, width: 820, height: 458 });
setShapeText(problem3Slide, 2, "图7：共享前的规范化与扰动流程", { left: 930, top: 286, width: 330, height: 40 });
setShapeText(problem3Slide, 5, "L2 范数裁剪", { left: 960, top: 378, width: 310, height: 40 });
setShapeText(problem3Slide, 6, "先计算原型向量范数并执行上界裁剪，把共享表示约束在统一幅值范围内，抑制少数极端客户端对服务端加权聚合的主导效应。", { left: 960, top: 426, width: 470, height: 116 });
setShapeText(problem3Slide, 9, "高斯噪声扰动", { left: 960, top: 574, width: 310, height: 40 });
setShapeText(problem3Slide, 10, "在裁剪后的原型上叠加受控高斯噪声，在尽量保留类别结构信息的同时降低精确表示被反推或直接识别的风险。", { left: 960, top: 622, width: 470, height: 116 });

const problem4Slide = presentation.slides.getItem(7);
setShapeText(problem4Slide, 1, "在强 Non-IID 条件下，各客户端往往沿本地数据分布独立更新，导致局部表示空间逐步分叉，最终表现为客户端漂移与全局聚合效果下降。项目没有强行约束参数完全一致，而是转向更稳定的特征层对齐：以全局类别原型作为蒸馏目标，在保留本地判别能力的同时提升跨客户端表示一致性。", { left: 66, top: 132, width: 1404, height: 94 });
setImagePosition(problem4Slide, 0, { left: 66, top: 240, width: 640, height: 356 });
setImagePosition(problem4Slide, 1, { left: 830, top: 240, width: 640, height: 382 });
setShapeText(problem4Slide, 2, "图8：本地类别原型与全局原型的蒸馏损失计算", { left: 66, top: 612, width: 640, height: 34 });
setShapeText(problem4Slide, 3, "图9：分类损失与蒸馏损失的联合优化", { left: 830, top: 612, width: 640, height: 34 });
setShapeText(problem4Slide, 5, "训练时，本地 batch 先形成类别原型，再与服务器广播的全局原型计算蒸馏损失；若某类尚无可靠全局原型则自动跳过。最终以 L = L_cls + λL_distill 进行联合优化，使模型兼顾本地任务适应性与全局方向一致性。", { left: 110, top: 694, width: 1320, height: 96 });

const closingSlide = presentation.slides.getItem(0).duplicate();
closingSlide.setIndex(presentation.slides.count);
setShapeText(closingSlide, 0, "谢谢聆听", { left: 360, top: 232, width: 816, height: 86 });
setShapeText(closingSlide, 1, "请各位老师批评指正", { left: 288, top: 352, width: 960, height: 52 });
removeShapesFrom(closingSlide, 2);

await exportPreview(contentsSlide, "slide-02-contents.png");
await exportPreview(problem3Slide, "slide-07-problem3.png");

const outputBlob = await PresentationFile.exportPptx(presentation);
await outputBlob.save(OUTPUT);
await reorderSlidesInPptx(OUTPUT, [1, 4, 3, 5, 6, 7, 8, 9, 2]);

console.log(`saved: ${OUTPUT}`);
