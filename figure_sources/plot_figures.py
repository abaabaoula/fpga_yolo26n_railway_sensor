"""Regenerate the manuscript's vector diagrams and plots from the packaged evidence."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PAPER = Path(__file__).resolve().parents[1]
DATA = PAPER / "source_data"
INK = "#172740"
MUTED = "#4b5f76"
GRID = "#dce4ed"
VIOLET = "#6244e8"
ORANGE = "#f15b32"
BLUE = "#2360e3"
GREEN = "#008f71"
PINK = "#d43877"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#8697a8",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "pdf.fonttype": 42,
    "mathtext.default": "regular",
})


def load_json(name):
    return json.loads((DATA / name).read_text(encoding="utf-8-sig"))


def save(figure, name, title, *, tight=False):
    figure.savefig(PAPER / name, bbox_inches="tight" if tight else None, pad_inches=.06, metadata={
        "Title": title, "Creator": "Manuscript figure source",
        "CreationDate": None, "ModDate": None,
    })
    plt.close(figure)


def draw_architecture():
    figure = plt.figure(figsize=(6.75, 4.15))
    ax = figure.add_axes([0, 0, 1, 1], xlim=(0, 100), ylim=(0, 64))
    ax.set_axis_off()

    def rect(x, y, width, height, fill, border, radius=1.1):
        ax.add_patch(FancyBboxPatch((x, y), width, height,
                                   boxstyle=f"round,pad=0.02,rounding_size={radius}",
                                   linewidth=0.8, edgecolor=border, facecolor=fill))

    def text(x, y, label, size=9.5, *, color=INK, weight="normal", ha="center"):
        ax.text(x, y, label, ha=ha, va="center", fontsize=size,
                color=color, fontweight=weight, linespacing=1.22)

    def link(x, y, x1, y1, *, color="#496985", two_way=False):
        ax.add_patch(FancyArrowPatch((x, y), (x1, y1),
                                    arrowstyle="<|-|>" if two_way else "-|>",
                                    mutation_scale=9, linewidth=1.10, color=color,
                                    shrinkA=0, shrinkB=0))

    def panel(x, y, letter, label):
        rect(x, y-1.8, 4.5, 3.6, INK, INK, .65)
        text(x+2.25, y, letter, 9, color="white", weight="bold")
        text(x+6, y, label, 12.1, weight="bold", ha="left")

    # Application-level data flow. The bypass retains full-image proposals for fusion.
    rect(1, 36.9, 98, 25.9, "#f4f8fd", "#d5e0ed", 1.55)
    panel(4, 58.9, "a", "Railway inference")
    nodes = [
        (4, 14, "Input frame", "BGR source", "#ffffff", "#9fb0c4"),
        (22.5, 15, "Global call", "384 × 640", "#e8f1ff", "#72a2ed"),
        (42.2, 15.3, "Select strip", "ARM routing", "#fff4e6", "#e8ac56"),
        (62.2, 15.1, "Refinement", "256 × 1280", "#e8f1ff", "#72a2ed"),
        (82, 14.0, "Map + fuse", "person boxes", "#e4f6ef", "#60a994"),
    ]
    for x, width, heading, description, fill, border in nodes:
        rect(x, 44.2, width, 9.65, fill, border)
        text(x+width/2, 50.6, heading, 9.15, weight="bold")
        text(x+width/2, 47.05, description, 8.75, color=MUTED)
    for left, right in zip(nodes, nodes[1:]):
        link(left[0]+left[1]+0.5, 49.0, right[0]-0.45, 49.0)
    ax.plot([30, 30, 89], [43.9, 40.1, 40.1], color="#496985", lw=.95)
    link(89, 40.1, 89, 43.9)
    text(59.5, 38.55, "Global proposals also go to fusion", 8.45, color=MUTED)

    text(50, 34.4, "Both detector calls use the same executor", 9.9,
         color="#224777", weight="bold")
    rect(1, 1.1, 98, 30.8, "#f5f9fb", "#d5e0ed", 1.55)
    panel(4, 28.2, "b", "Shared Zynq-7020 executor")

    # Host operations and ordinary external-memory transfers.
    rect(4, 5.7, 28.4, 18.75, "#edf2fa", "#a2b4cf")
    text(18.2, 21.65, "ARM Cortex-A9", 10, weight="bold")
    text(18.2, 17.65, "Input preparation and dispatch", 7.9)
    text(18.2, 14.35, "8 integer depthwise conv.", 7.9)
    text(18.2, 11.25, "FP32 non-convolution ops", 7.9)
    text(18.2, 8.2, "Ranking and decoding", 7.9)

    rect(38.4, 9.15, 18.75, 12.5, "#ffffff", "#9fb0c4")
    text(47.775, 17.7, "Off-chip DDR", 9.6, weight="bold")
    text(47.775, 13.65, "input and features", 8.5, color=MUTED)
    link(33.0, 15.4, 37.75, 15.4, two_way=True)
    link(57.8, 15.4, 66.25, 15.4, two_way=True)
    text(62.0, 18.8, "AXI DMA", 7.75, color=MUTED)
    text(62.0, 12.7, "HP0", 7.65, color=MUTED)

    # The cache is coupled to one sequential engine.
    rect(67, 4.05, 29.0, 20.4, "#e7f5f0", "#60a994")
    text(81.5, 22.05, "Programmable logic", 9.95, weight="bold")
    text(81.5, 18.35, "94 dense INT8 conv. per call", 8.1)
    text(81.5, 15.15, "Sequential · 128 MAC · 100 MHz", 7.85, color=MUTED)
    link(81.5, 13.7, 81.5, 10.65, color="#18745d", two_way=True)
    rect(70.0, 5.65, 23.0, 4.6, "#c7ebde", "#4e9c87", .65)
    text(81.5, 7.92, "128 KiB local cache", 8.45, weight="bold")
    save(figure, "Figure1.pdf", "Railway inference and the shared Zynq-7020 executor")


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=.7)
    ax.tick_params(length=3, labelsize=8.6)


def draw_mechanism():
    m = load_json("rectangular_mechanism_summary.json")
    review = load_json("result_review.json")
    with plt.rc_context(plt.rcParamsDefault):
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})

        def save(fig, name):
            assert name == 'rectangular_execution_mechanisms'
            fig.savefig(PAPER / "Figure2.pdf", dpi=300, bbox_inches='tight',
                        metadata={"CreationDate": None, "ModDate": None})
            plt.close(fig)

        # Archived Figure 2 geometry and author-selected data colours.
        def point(shape,options,resident):return next(p for p in m['points'] if p['shape']==shape and p['options']==options and p['resident']==resident)
        fig,axes=plt.subplots(2,2,figsize=(10.2,7.1));options=[4,4100,6276]
        for ax,shape in zip(axes[0],('384x640','256x1280')):
            for shift,resident,color in [(-.18,False,'#6C505B'),(.18,True,'#CA7E6E')]:
                values=[point(shape,o,resident)['ms'] for o in options]
                lo=[];hi=[]
                for o,value in zip(options,values):
                    row=next(v for v in review['cache_effects'] if v['shape']==shape and v['options']==o)
                    rounds=[v['on_ms' if resident else 'off_ms'] for v in row['rounds']]
                    lo.append(max(0.,value-min(rounds)));hi.append(max(0.,max(rounds)-value))
                bars=ax.bar(np.arange(3)+shift,values,.34,color=color,label='Resident on' if resident else 'Resident off',yerr=[lo,hi],capsize=3)
                for bar,value,upper in zip(bars,values,hi):ax.text(bar.get_x()+bar.get_width()/2,value+upper+15,f'{value:.0f}',ha='center',va='bottom',fontsize=8)
            ax.set_xticks(range(3),['flags4','flags4100','flags6276']);ax.set_ylim(0,max(point(shape,o,False)['ms'] for o in options)*1.2)
            ax.set_title(f'{shape} (height × width)',fontsize=10);ax.set_ylabel('Invocation latency (ms)')
            ax.legend(fontsize=8);ax.grid(axis='y',alpha=.15)

        ax=axes[1,0];x=np.arange(4);tick=[];reads=[];writes=[]
        for shape in ('384x640','256x1280'):
            for resident in (False,True):
                p=point(shape,6276,resident);reads.append(p['HP0_read_raw_counter']/1e6);writes.append(p['HP0_write_raw_counter']/1e6)
                tick.append(shape+'\n'+('on' if resident else 'off'))
        ax.bar(x,reads,color='#696F90',label='HP0 read bytes');ax.bar(x,writes,bottom=reads,color='#4395A7',label='HP0 write bytes')
        ax.set_xticks(x,tick,fontsize=8);ax.set_ylabel('Measured HP0 bytes per call (MB)');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.15)
        ax=axes[1,1];bottom=np.zeros(4)
        for key,label,color in [('pack_ms','Packing','#00C9BA'),('DMA_wait_ms','DMA wait (includes PL)','#C1FCF5'),('ARM_residual_including_norm_ms','ARM residual (includes norm/DW)','#49817B')]:
            values=[point(shape,6276,resident)[key] for shape in ('384x640','256x1280') for resident in (False,True)]
            ax.bar(x,values,bottom=bottom,color=color,label=label);bottom+=values
        ax.set_xticks(x,tick,fontsize=8);ax.set_ylabel('Native graph mean time (ms)');ax.legend(fontsize=7.3);ax.grid(axis='y',alpha=.15)
        fig.text(.5,.005,'Top: range of three round means. Same v3 library and firmware. Bottom: flags6276; diagnostics excluded.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.025,1,1));save(fig,'rectangular_execution_mechanisms')


def draw_tradeoff():
    with (DATA / "joint240_actual_ARM.csv").open(encoding="utf-8-sig", newline="") as stream:
        records = {record["method"]: record for record in csv.DictReader(stream)}
    styles = {
        "global_opt": (BLUE, "s", "ARM–FPGA global", (-1, -19), "left", "top"),
        "adaptive_opt": (GREEN, "D", "ARM–FPGA adaptive", (0, 11), "center", "bottom"),
        "ncnn_global640": (VIOLET, "^", "NCNN FP32 global", (8, -9), "left", "top"),
        "ncnn_adaptive": (ORANGE, "o", "NCNN FP32 adaptive", (0, 11), "center", "bottom"),
        "ncnn_int8_adaptive": (PINK, "P", "NCNN INT8 adaptive", (7, -8), "left", "top"),
        "ncnn_whole1280": ("#087fbb", "v", "NCNN FP32\nfull image (1280)", (0, 11), "center", "bottom"),
    }
    assert set(styles) == set(records)
    figure, ax = plt.subplots(figsize=(6.75, 3.65))
    figure.subplots_adjust(left=.12, right=.975, bottom=.16, top=.946)
    for method, (color, marker, label, offset, ha, va) in styles.items():
        record = records[method]
        assert int(record["images"]) == 240 and int(record["timings"]) == 720 and int(record["GT"]) == 479
        position = float(record["mean_ms"])/1000, float(record["recall"])*100
        ax.scatter(*position, color=color, marker=marker, s=94 if method == "adaptive_opt" else 74,
                   linewidths=.8, edgecolors=INK, zorder=3, label=label.replace("\n", " "))
        ax.annotate(label, position, xytext=offset, textcoords="offset points",
                    ha=ha, va=va, fontsize=8.85, color=INK)
    ax.set(xlim=(0, 10.35), ylim=(0, 60), xlabel="Mean application latency (s)",
           ylabel="Recall of provided person boxes (%)")
    ax.set_xticks(range(0, 11, 2))
    ax.set_yticks(range(0, 61, 10))
    style_axes(ax)
    ax.grid(axis="x", color=GRID, linewidth=.6)
    ax.legend(loc="lower right", ncols=2, frameon=True, framealpha=1,
              edgecolor=GRID, facecolor="white", fontsize=8.1, columnspacing=1.15,
              handletextpad=.4, markerscale=.82)
    save(figure, "Figure3.pdf", "Application latency and recall of provided person boxes")


def draw_external():
    record = load_json("external_second_evaluator.json")
    assert record["status"] == "PASS" and record["GT_instances"] == 271 and record["images"] == 86
    methods = ("old_adaptive", "roi_score", "ncnn_adaptive")
    labels = ("ARM–FPGA adaptive", "ARM–FPGA region-score", "NCNN FP32 adaptive")
    categories = (
        ("matched_primary", "IoU ≥ 0.50, score ≥ 0.25 (matched)", "#238b72"),
        ("localized_below_score", "IoU ≥ 0.50, score < 0.25", "#f2b744"),
        ("weak_localization", "0.10 ≤ IoU < 0.50", "#648db3"),
        ("no_person_candidate_IoU01", "IoU < 0.10", "#b6bdc6"),
        ("matching_competition", "IoU ≥ 0.50, score ≥ 0.25 (unmatched)", "#bb6b83"),
    )

    figure, ax = plt.subplots(figsize=(6.75, 3.05))
    figure.subplots_adjust(left=.266, right=.981, bottom=.25, top=.756)
    position = np.arange(len(methods))
    left = np.zeros(len(methods), dtype=int)
    for category, name, color in categories:
        counts = [record["methods"][method]["failure_categories"].get(category, 0) for method in methods]
        ax.barh(position, counts, left=left, color=color, label=name, height=.58,
                edgecolor="white", linewidth=.34)
        for y, count in enumerate(counts):
            center = left[y] + count / 2
            if count >= 14:
                ax.text(center, y, str(count), ha="center", va="center",
                        fontsize=9.5, color=INK)
            elif count:
                text_x = center + (10 if center < 271 / 2 else -10)
                ax.annotate(str(count), xy=(center, y + .28), xytext=(text_x, y + .49),
                            ha="center", va="center", fontsize=9.5, color=INK,
                            arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": .8,
                                        "shrinkA": 3, "shrinkB": 0})
        left += counts
    assert np.array_equal(left, np.full(3, 271))
    ax.set_yticks(position, labels)
    ax.tick_params(axis="y", length=0, labelsize=9.2)
    ax.tick_params(axis="x", length=3, labelsize=9)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(len(methods) - .33, -.45)
    ax.set_xlim(0, 271)
    ax.set_xlabel("Provided person boxes")
    handles, legend_labels = ax.get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(.515, .97),
                  ncols=2, frameon=False, fontsize=9.1, handlelength=1.2,
                  columnspacing=1.2, handletextpad=.4, labelspacing=.36)
    save(figure, "Figure5.pdf", "External evaluation of retained person candidates", tight=True)


if __name__ == "__main__":
    draw_mechanism()
    draw_tradeoff()
    draw_external()
