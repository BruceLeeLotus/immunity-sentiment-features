# -*- coding: utf-8 -*-
"""期刊级论文架构图绘制组件库（matplotlib）。

为什么需要它
------------
参考样张（如 Fig01_Three_Region_Dashed.pdf）本身就是用 matplotlib 手写坐标画出来的。
GitHub 上现成的架构图项目（PlotNeuralNet 等）都是 LaTeX/TikZ 路线，与本机 MiKTeX
的 pgf 数学引擎不兼容，且风格偏旧。所以正确做法是把样张里的绘制原语抽成可复用组件，
换论文时只改内容不改画法。

提供的原语
----------
stage_divider  阶段分隔：虚线竖线 + 顶部粗体阶段名（横向管线的骨架）
cuboid         3D 张量块：正面 + 顶面 + 右侧面（样张里表示张量/矩阵的方块）
block_stack    堆叠块：一列小方块，表示多层/多通道
container      虚线圆角容器：把一组元素框成"模块"
box            普通圆角盒：流程节点
arrow          正交箭头：直角折线 + 实心三角
sparkline      内嵌迷你曲线：在框图里塞一条数据示意
bars           内嵌迷你柱状
formula_box    公式标注框
chip           图例芯片
legend_row     一行图例

用法
----
    from paper_figure_kit import Kit, PALETTE
    fig, ax = plt.subplots(figsize=(6.535, 3.4))
    kit = Kit(ax, xlim=(0, 100), ylim=(0, 52))
    kit.stage_divider(20, "Representation")
    kit.cuboid(4, 20, 10, 8, PALETTE["gold"])
    ...
    kit.finish("fig1.pdf")

坐标系
------
全部用"设计单位"，通过 xlim/ylim 映射到画布。换尺寸时不用改坐标，只改 figsize。
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np

# ---------------- 定性色板：Okabe-Ito（见 配色规范.md） ----------------
# 由 Okabe & Ito 于 2002 年为色觉异常可辨而设计，Bang Wong 在 Nature Methods 2011
# 专栏推荐（别名 Wong palette），Claus Wilke《Fundamentals of Data Visualization》默认。
# 八色亮度各不相同，灰度打印也可辨。定性配色一律用它，不要换。
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#56B4E9",
             "#D55E00", "#CC79A7", "#F0E442", "#000000"]

PALETTE = {
    "blue": "#0072B2",        # 本文方法 / 主色
    "orange": "#E69F00",      # 对照模型（BERT）
    "green": "#009E73",       # 正向标签 / 第三类别
    "sky": "#56B4E9",         # 张量块 / 次级高亮
    "vermillion": "#D55E00",  # 单点强调（只标一个英雄结果时用）
    "purple": "#CC79A7",      # 备用第四类别
    "yellow": "#F0E442",      # 备用（浅底上慎用）
    "grey": "#999999",        # 其他基线 / 无贡献 / 背景
    "gold": "#E69F00",        # 张量块（同 orange）
    "ink": "#1A2430",         # 正文黑
    "muted": "#5A6773",       # 次级文字
    "edge": "#B8C5CE",        # 描边
    "panel": "#F2F7FC",       # 分组容器底色
    "grid": "#E8ECF0",
}
# 张量块默认配色序列（正面, 侧面加深）
BLOCK_FACES = ["gold", "sky", "blue", "green", "orange", "purple", "grey"]


# ==================== 文字适配（防止排版溢出） ====================
# 手写坐标画框图最常见的毛病是"字比盒子宽"：靠肉眼估字号必然翻车，
# 因为 Arial 每字符宽度、粗体加宽量、mathtext 公式宽度都估不准。
# 正确做法是**先量后画**：拿 matplotlib 的 renderer 量出真实包围盒，
# 再等比缩字号（或折行）到能放下。下面三个函数就是这套机制。
# 注意：坐标系函数用"数据单位"，与 ax 的 xlim/ylim 一致。

def _renderer(fig):
    """取画布 renderer；Agg 后端可直接取，失败则先 draw 一次。"""
    canvas = fig.canvas
    get = getattr(canvas, "get_renderer", None)
    if get is not None:
        try:
            return get()
        except Exception:
            pass
    canvas.draw()
    return canvas.renderer


def measure_text(ax, s, fs, bold=False, style="normal", lsp=1.35):
    """量文本渲染后的尺寸，单位与 ax 数据坐标一致。返回 (宽, 高)。

    实现：在 (0,0) 放一个临时文本 → 取显示坐标包围盒 → transData 逆变换
    回数据坐标 → 删掉临时文本。多行取最宽一行为宽、含行距为高。

    !!! 必须把画布 dpi 设到 300 以上（plt.subplots(..., dpi=300)）!!!
    低 dpi 下 Agg 会把字形步进吸附到整数像素，6pt 小字的测量偏差可达
    -6%（英文）/-4%（中文），会使 fit_fontsize 误判为超宽而过度缩字。
    dpi=300 实测偏差 ±0.3%，dpi>=600 收敛到 0.7% 内，矢量 PDF 天然无此问题。
    """
    t = ax.text(0, 0, s, fontsize=fs, fontweight="bold" if bold else "normal",
                style=style, linespacing=lsp)
    bb = t.get_window_extent(renderer=_renderer(ax.figure))
    t.remove()
    inv = ax.transData.inverted()
    x0, y0 = inv.transform((bb.x0, bb.y0))
    x1, y1 = inv.transform((bb.x1, bb.y1))
    return abs(x1 - x0), abs(y1 - y0)


def fit_fontsize(ax, s, fs, avail_w, bold=False, style="normal", lsp=1.35,
                 min_fs=4.2, tol=1.002, max_iter=8):
    """把字号缩到文本宽度 <= avail_w。返回 (字号, 是否仍溢出)。

    avail_w 已够宽时原样返回 fs（不放大，保持全图字号一致性）。
    字号与宽度不是严格线性（字形 hinting/取整），单次按比例缩放常在
    目标值附近差一点点，故这里迭代收敛；缩到 min_fs 仍放不下才置 True。
    """
    if avail_w <= 0:
        return min_fs, True
    w, _ = measure_text(ax, s, fs, bold, style, lsp)
    if w <= avail_w * tol:
        return fs, False
    cur = fs
    for _ in range(max_iter):
        nxt = max(min_fs, cur * avail_w / w)
        w2, _ = measure_text(ax, s, nxt, bold, style, lsp)
        if w2 <= avail_w * tol:
            return nxt, False
        if nxt <= min_fs + 1e-6:
            return min_fs, True
        cur, w = nxt, w2
    return cur, True


def wrap_text(ax, s, fs, avail_w, bold=False, style="normal", lsp=1.35,
              max_lines=3):
    """按可用宽度折行，优先在空格处断。放不下时返回多行字符串。"""
    if measure_text(ax, s, fs, bold, style, lsp)[0] <= avail_w:
        return s
    lines, cur = [], ""
    for wd in s.split(" "):
        trial = (cur + " " + wd).strip()
        if cur and measure_text(ax, trial, fs, bold, style, lsp)[0] > avail_w:
            lines.append(cur)
            cur = wd
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return "\n".join(lines[:max_lines])


def _shade(hex_color: str, k: float) -> tuple:
    """把十六进制色按系数 k 调暗/调亮，返回 0-1 的 rgb（截断到合法区间）。"""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return (min(1.0, r * k), min(1.0, g * k), min(1.0, b * k))


class Kit:
    """架构图绘制工具。所有方法都画在当前 ax 上，坐标用设计单位。"""

    def __init__(self, ax, xlim=(0, 100), ylim=(0, 100),
                 font="Arial", base_fontsize=6.6, zorder=1):
        self.ax = ax
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.axis("off")
        self.base_fontsize = base_fontsize
        self.z = zorder
        plt.rcParams["font.sans-serif"] = [font, "Helvetica", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

    # ---------------- 阶段分隔 ----------------
    def stage_divider(self, x, title, color=None, y_top=None, y_bot=None, fs=None):
        """虚线竖线 + 顶部粗体阶段名。横向管线的骨架。"""
        c = color or PALETTE["edge"]
        yb = self.ax.get_ylim()[0] if y_bot is None else y_bot
        yt = self.ax.get_ylim()[1] if y_top is None else y_top
        self.ax.plot([x, x], [yb, yt], color=c, lw=0.9, ls=(0, (4, 3)),
                     zorder=self.z)
        self.ax.text(x, yt + (yt - yb) * 0.012, title, ha="left", va="bottom",
                     fontsize=fs or self.base_fontsize + 0.9, fontweight="bold",
                     color=PALETTE["ink"], zorder=self.z + 2)

    # ---------------- 3D 张量块 ----------------
    def cuboid(self, x, y, w, h, face="gold", depth=None, edge=None,
               label=None, label_color="white", fs=None, lw=0.6, alpha=1.0,
               zorder=None):
        """3D 张量块：正面 + 顶面 + 右侧面，样张里表示张量/矩阵的方块。

        zorder 默认 None 用 Kit 构造时的 zorder；块内各面按 z / z+1 / z+2 分层。
        需要把块画在某个盒子之上时必须显式给更大的 zorder —— 否则会被盒子盖住。
        """
        z = self.z if zorder is None else zorder
        d = depth if depth is not None else min(w, h) * 0.28
        f = PALETTE.get(face, face)
        top = _shade(f, 1.28)
        side = _shade(f, 0.72)
        ec = edge or _shade(f, 0.55)
        # 右侧面
        self.ax.add_patch(mp.Polygon(
            [(x + w, y), (x + w + d, y + d), (x + w + d, y + h + d), (x + w, y + h)],
            closed=True, facecolor=side, edgecolor=ec, lw=lw, zorder=z))
        # 顶面
        self.ax.add_patch(mp.Polygon(
            [(x, y + h), (x + d, y + h + d), (x + w + d, y + h + d), (x + w, y + h)],
            closed=True, facecolor=top, edgecolor=ec, lw=lw, zorder=z))
        # 正面
        self.ax.add_patch(mp.Rectangle(
            (x, y), w, h, facecolor=f, edgecolor=ec, lw=lw, alpha=alpha,
            zorder=z + 1))
        if label:
            self.ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                         fontsize=fs or self.base_fontsize, color=label_color,
                         zorder=z + 2, linespacing=1.2)

    def block_stack(self, x, y, w, h, n=3, face="gold", depth=None, gap=0.35,
                    lw=0.5):
        """一列小方块，表示多层/多通道（样张里 '256 units' 那种）。"""
        d = depth if depth is not None else w * 0.22
        step = (h - n * 0) / n
        for i in range(n):
            self.cuboid(x, y + i * step, w, step - gap, face=face, depth=d, lw=lw)

    def slab_stack(self, x, y, w, h, n=5, face="sky", total_depth=None,
                   plate_ratio=0.46, edge=None, lw=0.5, alpha=1.0,
                   label=None, fs=None, label_color="white", zorder=None):
        """PlotNeuralNet 风格堆叠块：n 片薄板沿深度轴错位堆叠成"一副牌"。

        与 cuboid 的区别：cuboid 是一个实心三维块，只表达"这里有张量"；
        slab_stack 把块拆成 n 片可见的薄板，用来表达**张量的维度/通道数**。
        本图用它区分 V 维稀疏 TF-IDF 向量（多片）与 300 维 SVD 向量（少片）。

        参数 n 为板数，total_depth 为整叠在深度方向的总跨度（默认 0.6w）。
        板厚取步长的 plate_ratio，板间留缝，使每片轮廓可辨。
        """
        D = total_depth if total_depth is not None else w * 0.6
        step = D / n
        for i in range(n - 1, -1, -1):           # 由后往前画，前面的板自然压住后面的
            self.cuboid(x + i * step, y + i * step, w, h, face=face,
                        depth=step * plate_ratio, edge=edge, lw=lw, alpha=alpha,
                        zorder=zorder)
        if label:
            self.ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                         fontsize=fs or self.base_fontsize, color=label_color,
                         zorder=self.z + 3, linespacing=1.2)

    # ---------------- 容器与盒子 ----------------
    def container(self, x, y, w, h, label=None, fc=None, ec=None, ls=(0, (4, 3)),
                  r=0.9, lw=0.8, label_fs=None, label_color=None):
        """虚线圆角容器：把一组元素框成"模块"。"""
        self.ax.add_patch(mp.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.25", linewidth=lw,
            edgecolor=ec or PALETTE["edge"], facecolor=fc or PALETTE["panel"],
            linestyle=ls, mutation_scale=0.9, zorder=self.z - 1))
        if label:
            self.ax.text(x + w / 2, y + h + 0.9, label, ha="center", va="bottom",
                         fontsize=label_fs or self.base_fontsize + 0.4,
                         fontweight="bold", color=label_color or PALETTE["ink"],
                         zorder=self.z + 2)

    def box(self, x, y, w, h, text, fc="white", ec=None, fs=None, bold=False,
            tc=None, lw=0.8, r=0.5, align="center"):
        """普通圆角盒：流程节点。"""
        self.ax.add_patch(mp.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.22", linewidth=lw,
            edgecolor=ec or PALETTE["edge"], facecolor=fc, mutation_scale=0.9,
            zorder=self.z))
        if text:
            tx = x + w / 2 if align == "center" else x + 1.2
            self.ax.text(tx, y + h / 2, text, ha=align, va="center",
                         fontsize=fs or self.base_fontsize,
                         fontweight="bold" if bold else "normal",
                         color=tc or PALETTE["ink"], zorder=self.z + 1,
                         linespacing=1.25)

    # ---------------- 连线 ----------------
    def arrow(self, pts, color=None, lw=0.9, head=2.0, ls="-", zorder=None):
        """正交折线 + 实心三角箭头。pts 为 [(x0,y0), (x1,y1), ...]。"""
        c = color or PALETTE["muted"]
        z = self.z if zorder is None else zorder
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        self.ax.plot(xs, ys, color=c, lw=lw, ls=ls, solid_capstyle="round",
                     solid_joinstyle="miter", zorder=z)
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        self.ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                         arrowprops=dict(arrowstyle="-|>", color=c, lw=lw,
                                         mutation_scale=head * 4.5,
                                         shrinkA=0, shrinkB=0), zorder=z + 1)

    # ---------------- 内嵌迷你图 ----------------
    def sparkline(self, x, y, w, h, data=None, color=None, lw=0.9, seed=0,
                  fill=True):
        """在框图里塞一条数据示意曲线（样张里输入序列那种）。"""
        c = color or PALETTE["blue"]
        if data is None:
            rng = np.random.default_rng(seed)
            t = np.linspace(0, 4 * np.pi, 90)
            data = (np.sin(t) * 0.35 + 0.5
                    + 0.14 * np.sin(3.1 * t + 0.7)
                    + rng.normal(0, 0.035, t.size))
            data = np.clip(data, 0, 1)
        d = np.asarray(data, dtype=float)
        d = (d - d.min()) / (d.max() - d.min() + 1e-9)
        xs = x + np.linspace(0, w, d.size)
        ys = y + d * h
        if fill:
            self.ax.fill_between(xs, y, ys, color=c, alpha=0.16, zorder=self.z)
        self.ax.plot(xs, ys, color=c, lw=lw, zorder=self.z + 1)

    def bars(self, x, y, w, h, n=12, color=None, seed=1, lw=0.4):
        """在框图里塞一组迷你柱（样张里 TF-IDF 向量那种）。"""
        c = color or PALETTE["blue"]
        rng = np.random.default_rng(seed)
        v = np.clip(rng.random(n) * 0.75 + 0.25, 0, 1)
        bw = w / n * 0.66
        for i, vv in enumerate(v):
            self.ax.add_patch(mp.Rectangle(
                (x + i * (w / n), y), bw, vv * h, facecolor=c, alpha=0.85,
                edgecolor="none", zorder=self.z))

    def grid_icon(self, x, y, w, h, rows=5, cols=5, color=None, lw=0.4, seed=2):
        """矩阵网格图标（表示嵌入矩阵/注意力图）。"""
        c = color or PALETTE["sky"]
        rng = np.random.default_rng(seed)
        for i in range(rows):
            for j in range(cols):
                a = 0.25 + 0.75 * rng.random()
                self.ax.add_patch(mp.Rectangle(
                    (x + j * w / cols, y + i * h / rows), w / cols * 0.86,
                    h / rows * 0.86, facecolor=c, alpha=float(a),
                    edgecolor="none", zorder=self.z))

    # ---------------- 标注 ----------------
    def formula_box(self, x, y, w, text, fc="#EAF1F8", ec=None, fs=None,
                    tc=None):
        """公式标注框：浅蓝底 + 等宽公式。"""
        lines = text.count("\n") + 1
        h = 1.9 * lines + 1.4
        self.ax.add_patch(mp.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.2", linewidth=0.6,
            edgecolor=ec or "#B9CDE0", facecolor=fc, mutation_scale=0.9,
            zorder=self.z))
        self.ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                     fontsize=fs or self.base_fontsize - 0.5,
                     color=tc or PALETTE["ink"], zorder=self.z + 1,
                     linespacing=1.35)
        return h

    def chip(self, x, y, w, h, label, fc, ec=None, fs=None, tc=None):
        """图例芯片：色块 + 文字。"""
        self.ax.add_patch(mp.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.15", linewidth=0.6,
            edgecolor=ec or PALETTE["edge"], facecolor=fc, mutation_scale=0.9,
            zorder=self.z))
        self.ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                     fontsize=fs or self.base_fontsize - 0.4,
                     color=tc or PALETTE["ink"], zorder=self.z + 1)

    def legend_row(self, y, items, x0=0.0, total_w=None, h=2.6, gap=1.2):
        """一行图例。items 为 [(文字, 颜色), ...]。"""
        total_w = total_w or self.ax.get_xlim()[1] - x0
        n = len(items)
        w = (total_w - gap * (n - 1)) / n
        for i, (label, color) in enumerate(items):
            self.chip(x0 + i * (w + gap), y, w, h, label,
                      fc=PALETTE.get(color, color), tc="white")

    # ---------------- 导出 ----------------
    def finish(self, path_pdf, path_png=None, dpi=300, tight=True, path_svg=None):
        """保存并关闭。path_svg 传了就额外出一份 SVG。

        SVG 用 svg.fonttype='none' 保存 —— matplotlib 默认是 'path'，会把所有文字
        转成矢量轮廓，导进 Inkscape 后能挪位置、改颜色，但**双击改不了字**。
        改成 'none' 后写的是真 <text> 元素，文字可编辑。
        代价：mathtext 公式会以字体名引用写出，若编辑器缺该字体（cmmi10 之类）
        会显示成替代字体，需要手动调。
        """
        import matplotlib.pyplot as plt
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["ps.fonttype"] = 42
        bb = "tight" if tight else None
        self.ax.figure.savefig(path_pdf, bbox_inches=bb)
        if path_png:
            self.ax.figure.savefig(path_png, dpi=dpi, bbox_inches=bb)
        if path_svg:
            with plt.rc_context({"svg.fonttype": "none"}):
                self.ax.figure.savefig(path_svg, bbox_inches=bb)
        plt.close(self.ax.figure)
        print("saved:", path_pdf, ("| " + path_png) if path_png else "",
              ("| " + path_svg) if path_svg else "")
